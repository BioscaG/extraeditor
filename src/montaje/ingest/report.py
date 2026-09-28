"""Ingest report: totals, devices, HDR mix, VFR, degraded assets, audio anomalies (§8 step 8)."""

from __future__ import annotations

from collections import Counter

from montaje.index.store import Store
from montaje.models.asset import Asset, DynamicRange
from montaje.workspace import Workspace, atomic_write_text

# Shared Albums historically capped videos at 720p; anything at or below this
# with an iPhone make is suspect (§7.2 mandatory quality check).
DEGRADED_MAX_EDGE = 1280


def detect_degraded(asset: Asset) -> str | None:
    v = asset.probe.video
    if v is None:
        return None
    long_edge = max(v.width, v.height)
    if asset.kind.value in ("video", "slomo") and long_edge <= DEGRADED_MAX_EDGE:
        return f"video long edge {long_edge}px ≤ {DEGRADED_MAX_EDGE}px (possible shared-album derivative)"
    return None


def build_report(ws: Workspace) -> str:
    store = Store(ws.db_path)
    assets = store.list_assets()

    kinds = Counter(a.kind.value for a in assets)
    devices = Counter(f"{a.probe.make or '?'} {a.probe.model or '?'}".strip() for a in assets)
    dr = Counter(
        a.probe.video.dynamic_range.value if a.probe.video else "n/a" for a in assets
    )
    total_dur = sum(a.duration_s for a in assets)
    vfr = [a for a in assets if a.probe.video and a.probe.video.vfr]
    no_audio = [a for a in assets if a.kind.value in ("video", "slomo") and not a.probe.audio]
    degraded = [(a, r) for a in assets if (r := detect_degraded(a))]
    failed = store.conn.execute(
        "SELECT asset_id, stage, error FROM status WHERE state = 'failed'"
    ).fetchall()

    lines = [
        "# Ingest report",
        "",
        f"- **Assets:** {len(assets)} — " + ", ".join(f"{k}: {n}" for k, n in kinds.most_common()),
        f"- **Total duration:** {total_dur / 60:.1f} min",
        "- **Dynamic range:** " + ", ".join(f"{k}: {n}" for k, n in dr.most_common()),
        "- **Devices:** " + ", ".join(f"{k}: {n}" for k, n in devices.most_common(10)),
        "",
    ]
    if any(k != DynamicRange.SDR.value for k in dr if k != "n/a") and dr.get("sdr"):
        lines.append("> ⚠️ Mixed HDR/SDR footage: color normalization (§15) will matter.\n")
    if vfr:
        lines.append(f"## VFR assets ({len(vfr)})\n")
        lines += [f"- `{a.asset_id}` {a.path.name} (avg {a.probe.video.avg_fps:.2f} fps)" for a in vfr]
        lines.append("")
    if degraded:
        lines.append(f"## Degraded assets ({len(degraded)})\n")
        lines += [f"- `{a.asset_id}` {a.path.name}: {r}" for a, r in degraded]
        lines.append("")
    if no_audio:
        lines.append(f"## Videos without audio ({len(no_audio)})\n")
        lines += [f"- `{a.asset_id}` {a.path.name}" for a in no_audio]
        lines.append("")
    if failed:
        lines.append(f"## Failed stages ({len(failed)})\n")
        lines += [f"- `{r['asset_id']}` {r['stage']}: {(r['error'] or '')[:200]}" for r in failed]
        lines.append("")

    store.close()
    report = "\n".join(lines)
    atomic_write_text(ws.root / "report.md", report)
    return report
