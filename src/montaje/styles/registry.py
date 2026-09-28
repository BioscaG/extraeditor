"""Style loading and saving (§22)."""

from __future__ import annotations

from pathlib import Path

import yaml

from montaje.models.style import Style

REPO_ROOT = Path(__file__).resolve().parents[3]
STYLES_DIR = REPO_ROOT / "library" / "styles"

DEFAULT_STYLE = "modern-festival"


def list_styles() -> list[str]:
    if not STYLES_DIR.is_dir():
        return []
    return sorted(p.stem for p in STYLES_DIR.glob("*.yaml"))


def load_style(name: str | None) -> Style | None:
    """Load a style by name. None means no style — the plan then has no targets."""
    if name is None:
        return None
    path = STYLES_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    data.setdefault("name", name)
    return Style.model_validate(data)


def load_default() -> Style:
    style = load_style(DEFAULT_STYLE)
    if style is None:
        # A missing style file must not break a render; an empty style simply means
        # the rhythm report has no targets to check against.
        return Style(name="none")
    return style


def save_style(style: Style) -> Path:
    STYLES_DIR.mkdir(parents=True, exist_ok=True)
    path = STYLES_DIR / f"{style.name}.yaml"
    path.write_text(
        yaml.safe_dump(style.model_dump(mode="json", exclude_none=True), sort_keys=False,
                       allow_unicode=True)
    )
    return path
