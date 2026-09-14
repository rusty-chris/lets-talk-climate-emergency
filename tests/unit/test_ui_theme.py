"""Issue #398 — guards for the planet-Earth theme (Path A, aggressive theming).

These are structural/hygiene guards in the shape of the #18 shell-hygiene
tests (tests/unit/test_ui_shell_hygiene.py): they pin that the theme is wired
and, above all, that it stays SELF-CONTAINED. The product follows a strict
no-external-requests convention (the transparency pages / CSP); a Google Font
``<link>`` or a remote image URL sneaking into the injected CSS would break it
silently. The URL guard fails here, in unit tests, before it ever ships.

They do not assert a rendered pixel — the theme's *look* is judged from a
screenshot; these only pin that it is present, wired, and remote-asset-free.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = REPO_ROOT / "ui"
CONFIG_TOML = REPO_ROOT / ".streamlit" / "config.toml"

#: Anything that would trigger an outbound request. Protocol-relative ``//``
#: is matched only as a scheme-less host (``//host``), never CSS comments.
_EXTERNAL_URL = re.compile(r"https?:|(?<![:/])//[a-z0-9.\-]+\.[a-z]", re.IGNORECASE)


def _referenced_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _app_referenced_names() -> set[str]:
    return _referenced_names(ast.parse((UI_DIR / "app.py").read_text(encoding="utf-8")))


class TestThemeConfig:
    """`.streamlit/config.toml` carries the dark planet palette."""

    def test_config_exists_and_is_valid_toml(self) -> None:
        assert CONFIG_TOML.exists(), (
            ".streamlit/config.toml must exist — it sets the base dark palette "
            "Streamlit reads at startup (issue #398)"
        )
        tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))

    def test_theme_is_dark_with_a_full_palette(self) -> None:
        config = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
        theme = config.get("theme", {})
        assert theme.get("base") == "dark", "the planet theme is dark (issue #398)"
        for key in ("primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor"):
            value = theme.get(key, "")
            assert isinstance(value, str) and value.startswith("#"), (
                f"theme.{key} must be a hex colour so the palette is explicit (issue #398)"
            )

    def test_no_outbound_usage_ping(self) -> None:
        """Streamlit's usage-stats gather is an outbound request; the
        no-external-fetch convention keeps it off."""
        config = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
        assert config.get("browser", {}).get("gatherUsageStats") is False


class TestThemeModuleIsSelfContained:
    """The injected CSS/SVG references no remote host — the hard #398 rule."""

    def test_builders_return_markup(self) -> None:
        from ui.theme import globe_loader_html, hero_html, theme_style_block

        assert theme_style_block().startswith("<style>")
        assert "climate-globe" in globe_loader_html()
        assert "<svg" in globe_loader_html()  # the globe is an inline SVG, no glyph/fetch
        assert hero_html("Name", "Tagline").startswith("<div")

    @pytest.mark.parametrize("builder", ["theme_style_block", "globe_loader_html", "hero_html"])
    def test_no_external_url_in_any_builder(self, builder: str) -> None:
        import ui.theme as theme

        fn = getattr(theme, builder)
        markup = fn("Name", "Tagline") if builder == "hero_html" else fn()
        hit = _EXTERNAL_URL.search(markup)
        assert hit is None, (
            f"ui.theme.{builder}() emits an external reference ({hit.group(0)!r}); "
            "all CSS/SVG/fonts must be self-contained (gradients, inline SVG, hex "
            "colours) — no <link>/remote image (issue #398 no-external-requests rule)"
        )

    def test_module_source_has_no_external_url(self) -> None:
        source = (UI_DIR / "theme.py").read_text(encoding="utf-8")
        # Strip the docstring's prose mentions of the convention before scanning
        # (they name the rule; they are not markup).
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        assert _EXTERNAL_URL.search(code) is None, (
            "ui/theme.py references a remote host in code; the theme must be "
            "self-contained (issue #398)"
        )


class TestShellWiresTheTheme:
    """ui/app.py actually injects the theme, hero and globe loader."""

    def test_theme_is_injected_once(self) -> None:
        assert "inject_theme" in _app_referenced_names(), (
            "ui/app.py must call inject_theme() so the planet CSS reaches every "
            "rerun before widgets draw (issue #398)"
        )

    def test_landing_hero_is_rendered(self) -> None:
        assert "render_hero" in _app_referenced_names(), (
            "ui/app.py must render the Earth-from-space hero on the landing page (issue #398)"
        )

    def test_globe_loader_replaces_the_spinner_on_the_stream(self) -> None:
        assert "globe_loader_placeholder" in _app_referenced_names(), (
            "ui/app.py must show the spinning-globe loader while an answer "
            "streams, standing in for the default Streamlit spinner (issue #398)"
        )
