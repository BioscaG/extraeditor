"""Creative grade: the same look applied to every shot (§15 step 4).

Normalization and matching make shots agree with each other; the grade is what makes
them look like *one film*. It is deliberately identical for every shot — a per-shot
creative grade is what makes amateur edits feel incoherent.

Either a style LUT from `library/luts/` or a parametric grade, never both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from montaje.models.style import ColorPolicy


@dataclass(frozen=True)
class Grade:
    """A creative grade, expressed as an ffmpeg filter chain."""

    lut: Path | None = None
    contrast: float = 1.0  # 1.0 = unchanged
    saturation: float = 1.0
    shadow_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)  # split toning, -1..1 per channel
    highlight_tint: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grain: float = 0.0  # 0..1
    vignette: float = 0.0  # 0..1
    sharpen: float = 0.0  # 0..1

    def filters(self) -> list[str]:
        """Filter chain in application order: look first, then finishing (§15 step 5)."""
        chain: list[str] = []
        if self.lut is not None:
            # A LUT already encodes contrast and saturation, so parametric
            # adjustments are skipped to avoid applying the look twice.
            chain.append(f"lut3d=file='{self.lut}'")
        else:
            if abs(self.contrast - 1.0) > 1e-3 or abs(self.saturation - 1.0) > 1e-3:
                chain.append(f"eq=contrast={self.contrast:.4f}:saturation={self.saturation:.4f}")
            if any(abs(v) > 1e-3 for v in (*self.shadow_tint, *self.highlight_tint)):
                sr, sg, sb = self.shadow_tint
                hr, hg, hb = self.highlight_tint
                chain.append(
                    f"colorbalance=rs={sr:.3f}:gs={sg:.3f}:bs={sb:.3f}"
                    f":rh={hr:.3f}:gh={hg:.3f}:bh={hb:.3f}"
                )
        if self.vignette > 1e-3:
            # A subtle vignette focuses attention; angle scales with strength.
            angle = 0.4 + 0.9 * min(1.0, self.vignette)
            chain.append(f"vignette=angle=PI/{angle * 5:.3f}")
        if self.sharpen > 1e-3:
            chain.append(f"unsharp=5:5:{self.sharpen * 1.2:.3f}:5:5:0.0")
        if self.grain > 1e-3:
            # Grain last: it must sit on top of the graded image, and `noise` is
            # applied per-frame so it must not be sharpened or scaled afterwards.
            chain.append(f"noise=alls={self.grain * 20:.1f}:allf=t+u")
        return chain

    def to_filter(self) -> str | None:
        chain = self.filters()
        return ",".join(chain) if chain else None


def grade_from_style(policy: ColorPolicy, lut_dir: Path | None = None) -> Grade:
    """Build a Grade from a style's color policy, resolving the LUT path."""
    lut: Path | None = None
    if policy.lut and lut_dir is not None:
        candidate = lut_dir / policy.lut
        # A missing LUT falls back to no LUT rather than failing the render: the
        # style is data and may reference a LUT this install does not have.
        lut = candidate if candidate.is_file() else None
    return Grade(lut=lut, grain=policy.grain)
