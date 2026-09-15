"""Issue #398 (+ post-launch polish) — the planet-Earth-from-space theme.

Owner feedback (2026-09-14): the site "looks like it's from the 90s" and
Streamlit "feels clunky". Path A is the aggressive-Streamlit-theming route
(fast, reversible, keeps the Python UI): the base palette lives in
``.streamlit/config.toml`` and is reinforced here with one block of injected,
self-contained CSS, a **spinning-Earth** loader that replaces the default
Streamlit spinner while an answer generates, a landing **hero** and a slim
branded **top bar** (Rusty Data mark + the transparency menu).

Post-launch owner iteration (2026-09-15): use a *real* Earth for the hero and
the loader (a NASA Blue-Marble texture, baked in — see
``static/earth_texture.SOURCE.txt`` for provenance/licence), warm the title
wordmark with an orange-red that sits against the blue, use the previously
blank header for branding + a menu, and move the page title into the header
once a chat has started.

Self-contained by construction — the no-external-requests convention holds:
this module references NO remote host. Colours are hex literals; the Earth is
a **data-URI** baked from a committed, public-domain texture (read at import,
never fetched — mirroring how ``ui.footer`` inlines the steward mark). The
NASA source URL is deliberately kept OUT of this source file (it lives in the
provenance sidecar) so the ``no external URL`` guard over this module holds.
``tests/unit/test_ui_theme.py`` guards that no ``http``/protocol-relative URL
leaks into the injected markup.

The split follows the #18 shell/core discipline: the string builders are pure
and import nothing, so they are testable without Streamlit; the thin
``inject_*`` / ``render_*`` helpers import ``streamlit`` locally.
"""

from __future__ import annotations

import base64
from pathlib import Path

# --- Palette (kept in lock-step with .streamlit/config.toml) ---------------
#: Deep space/ocean canvas, the lighter ocean-blue panel, the atmospheric
#: cyan accent, a softer teal, and the high-contrast off-white text. The
#: text-on-canvas pair clears WCAG AA for body copy in the dark.
SPACE = "#071522"
OCEAN = "#0f2c47"
OCEAN_DEEP = "#0a2138"
ATMOSPHERE = "#38bdf8"
TEAL = "#22d3ee"
LAND = "#2e7d5b"
TEXT = "#e8f2ff"
MUTED = "#9fb6cc"
#: The warm/hot accent (owner ask 2026-09-15): an orange-red for the title
#: wordmark that reads as heat against the cool blue — the climate tension in
#: two colours. Used only in the title gradient, never for body text.
WARM = "#ff6b4a"
WARM_DEEP = "#f4451f"

#: The Earth texture, baked to a data-URI at import from the committed
#: public-domain NASA Blue-Marble JPEG. Read once (like the footer mark);
#: missing-asset degrades to a CSS gradient rather than crashing import.
_EARTH_TEXTURE_PATH = Path(__file__).resolve().parent / "static" / "earth_texture.jpg"


def _earth_data_uri() -> str:
    """``data:`` URI for the baked Earth texture, or ``""`` if unavailable.

    Not an external reference — the bytes are committed in the repo and read
    from disk; nothing is fetched at runtime (the no-external-requests rule)."""
    try:
        raw = _EARTH_TEXTURE_PATH.read_bytes()
    except OSError:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


#: Computed once at import. Empty string ⇒ the CSS falls back to a gradient
#: sphere (see ``--climate-earth`` usage), so the theme never hard-depends on
#: the binary asset being present.
_EARTH_URI = _earth_data_uri()

#: The Earth background: the real texture when baked, else a cool gradient so
#: the sphere still reads as a planet. Both are self-contained.
_EARTH_BG = (
    f'url("{_EARTH_URI}")'
    if _EARTH_URI
    else f"radial-gradient(circle at 34% 30%, {TEAL} 0%, {ATMOSPHERE} 45%, {OCEAN_DEEP} 100%)"
)

#: One authoritative CSS block. Scoped to Streamlit's stable structural
#: selectors; everything degrades to the config.toml palette if a Streamlit
#: release renames a class, so a missed selector dulls polish but never breaks.
_CSS = f"""
:root {{ --climate-earth: {_EARTH_BG}; }}

/* Atmospheric canvas: a subtle "Earth limb" glow bottom-centre plus a faint
   high-atmosphere haze top-right, over the deep-space base. */
.stApp {{
  background:
    radial-gradient(120% 90% at 50% 118%, {ATMOSPHERE}26 0%, {ATMOSPHERE}0d 26%, transparent 55%),
    radial-gradient(90% 70% at 88% -10%, {WARM}14 0%, transparent 42%),
    linear-gradient(180deg, {SPACE} 0%, {OCEAN_DEEP} 100%) fixed;
  color: {TEXT};
}}

/* The previously blank Streamlit header now blends into the canvas instead of
   sitting there as a weird translucent strip (owner ask). The app's own top
   bar (below) carries the branding + menu; Streamlit's settings menu stays
   reachable on the right. */
header[data-testid="stHeader"] {{ background: transparent; }}

/* Title wordmark: a cool-blue → warm orange-red gradient (owner ask) — heat
   against the blue. Applies to the landing <h1> and the hero/top-bar titles. */
.stApp h1, .climate-title {{
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
  background: linear-gradient(95deg, {ATMOSPHERE} 0%, {TEAL} 24%, {WARM} 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.stApp h3 {{ color: {MUTED}; font-weight: 500; }}
.stApp h2, .stApp h4 {{ color: {TEXT}; }}

/* Slim branded top bar: Rusty Data mark + (optional) page title on the left,
   the transparency menu on the right. Uses the header space purposefully. */
.climate-topbar {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 1rem; flex-wrap: wrap;
  margin: 0 0 0.6rem; padding: 0.35rem 0 0.55rem;
  border-bottom: 1px solid {ATMOSPHERE}1f;
}}
.climate-brandwrap {{ display: flex; align-items: center; gap: 0.6rem; min-width: 0; }}
.climate-brand {{ display: inline-flex; align-items: center; gap: 0.4rem;
  color: {MUTED}; font-weight: 600; font-size: 0.9rem; white-space: nowrap; }}
.climate-brand img {{ display: inline-block; vertical-align: middle; }}
.climate-topbar .climate-title {{ font-size: 1.05rem; font-weight: 700;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.climate-nav {{ display: flex; align-items: center; gap: 0.85rem; flex-wrap: wrap; }}
.climate-nav a {{ color: {MUTED}; font-size: 0.85rem; font-weight: 500;
  text-decoration: none; }}
.climate-nav a:hover {{ color: {ATMOSPHERE}; }}

/* The Earth sphere — real NASA texture on a circle, with sphere shading and a
   soft atmospheric rim. Shared by the hero (static) and the loader (spinning).
   The equirectangular texture is 2:1, shown at 200% width so a hemisphere
   fills the disc; scrolling the background-x spins it. */
.climate-earth-sphere {{
  border-radius: 50%;
  background-image: var(--climate-earth);
  background-size: 200% 100%;
  background-repeat: repeat-x;
  background-position: 0% 50%;
  position: relative;
  box-shadow: inset -8px -8px 20px {SPACE}bf,
              0 0 0 1px {ATMOSPHERE}33, 0 0 22px {ATMOSPHERE}4d;
}}
.climate-earth-sphere::after {{
  content: ""; position: absolute; inset: 0; border-radius: 50%;
  background:
    radial-gradient(circle at 30% 26%, {TEXT}59 0%, transparent 34%),
    radial-gradient(circle at 50% 50%, transparent 56%, {SPACE}b3 100%);
}}
@keyframes climate-earth-rotate {{
  from {{ background-position: 0% 50%; }}
  to   {{ background-position: -200% 50%; }}
}}
.climate-earth-sphere.spin {{ animation: climate-earth-rotate 14s linear infinite; }}

/* Loader wrapper: the spinning Earth + a pulsing atmosphere ring + label. */
@keyframes climate-atmos-pulse {{
  0%,100% {{ opacity: .35; transform: scale(1); }}
  50%     {{ opacity: .75; transform: scale(1.06); }}
}}
.climate-globe-loader {{
  display: flex; align-items: center; gap: 0.9rem;
  padding: 0.4rem 0; color: {MUTED}; font-weight: 500;
}}
.climate-globe {{ position: relative; width: 46px; height: 46px; flex: 0 0 auto; }}
.climate-globe .atmos {{
  position: absolute; inset: -5px; border-radius: 50%;
  background: radial-gradient(circle, {ATMOSPHERE}00 55%, {ATMOSPHERE}59 72%, {ATMOSPHERE}00 82%);
  animation: climate-atmos-pulse 2.4s ease-in-out infinite;
}}
.climate-globe .climate-earth-sphere {{ position: absolute; inset: 0; width: 46px; height: 46px; }}
@media (prefers-reduced-motion: reduce) {{
  .climate-earth-sphere.spin {{ animation: none; }}
  .climate-globe .atmos {{ animation: none; opacity: .6; }}
}}

/* Inline chart answers (owner ask: "show me X" must render a graph): the SVG
   is inlined as a data-URI <img> on a light card so the dark theme doesn't
   swallow the white chart. */
.climate-chart {{
  width: 100%; height: auto; display: block;
  background: #ffffff; border-radius: 12px; padding: 10px;
  border: 1px solid {ATMOSPHERE}26;
}}

/* Starter buttons / actions: glassy ocean chips with a cyan hairline. */
.stButton > button {{
  background: linear-gradient(180deg, {OCEAN} 0%, {OCEAN_DEEP} 100%);
  color: {TEXT};
  border: 1px solid {ATMOSPHERE}59;
  border-radius: 12px;
  padding: 0.55rem 0.9rem;
  font-weight: 600;
  transition: border-color .15s ease, box-shadow .15s ease, transform .05s ease;
}}
.stButton > button:hover {{
  border-color: {ATMOSPHERE};
  box-shadow: 0 0 0 1px {ATMOSPHERE}59, 0 6px 20px {SPACE}b3;
  color: {TEXT};
}}
.stButton > button:active {{ transform: translateY(1px); }}

/* Bordered containers (the #402 footprint panel) & metrics: ocean glass. */
.stApp [data-testid="stMetric"],
.stApp div[data-testid="stExpander"] {{
  background: {OCEAN}80;
  border: 1px solid {ATMOSPHERE}26;
  border-radius: 14px;
}}
.stApp [data-testid="stMetric"] {{ padding: 0.75rem 1rem; }}
.stApp [data-testid="stMetricValue"] {{ color: {ATMOSPHERE}; }}
.stApp [data-testid="stExpander"] summary {{ color: {TEXT}; }}

/* Progress bar (footprint gauge): the atmospheric cyan fill. */
.stApp [data-testid="stProgress"] div[role="progressbar"] > div,
.stApp .stProgress > div > div > div > div {{
  background-image: linear-gradient(90deg, {ATMOSPHERE}, {TEAL});
}}

/* Chat bubbles & popovers: translucent ocean with a soft cyan edge. */
.stApp [data-testid="stChatMessage"] {{
  background: {OCEAN}66;
  border: 1px solid {ATMOSPHERE}1f;
  border-radius: 16px;
}}

/* Chat input: the "Ask anything" bar gets an atmospheric focus ring. */
.stApp [data-testid="stChatInput"] textarea:focus,
.stApp [data-testid="stChatInput"] > div:focus-within {{
  border-color: {ATMOSPHERE};
  box-shadow: 0 0 0 1px {ATMOSPHERE}59;
}}

/* Links keep the accent so citations/permalinks read as interactive. */
.stApp a {{ color: {ATMOSPHERE}; }}
.stApp a:hover {{ color: {TEAL}; }}
"""


def _earth_sphere_span(size_px: int, *, spin: bool, extra_style: str = "") -> str:
    """A single Earth-sphere ``<span>`` at ``size_px`` (pure; self-contained)."""
    spin_cls = " spin" if spin else ""
    return (
        f'<span class="climate-earth-sphere{spin_cls}" aria-hidden="true" '
        f'style="width:{size_px}px;height:{size_px}px;display:inline-block;'
        f'flex:0 0 auto;{extra_style}"></span>'
    )


def theme_style_block() -> str:
    """The whole theme CSS as a single ``<style>`` block (pure; self-contained).

    Carries the Earth texture as a ``data:`` URI in the ``--climate-earth``
    custom property — a baked, committed asset, not a remote reference."""
    return f"<style>{_CSS}</style>"


def globe_loader_html(message: str = "Consulting the evidence…") -> str:
    """The spinning-Earth 'generating' indicator markup (pure; self-contained).

    A real Earth texture on a shaded sphere, rotated by scrolling its
    background; a pulsing atmosphere ring sits behind it. Replaces the default
    Streamlit spinner while an answer streams. The label is app-authored copy.
    """
    return (
        '<div class="climate-globe-loader">'
        '<span class="climate-globe" role="img" aria-label="Generating">'
        '<span class="atmos"></span>'
        f"{_earth_sphere_span(46, spin=True)}"
        "</span>"
        f"<span>{message}</span></div>"
    )


def hero_html(name: str, tagline: str) -> str:
    """The landing hero: a real Earth beside the warm-gradient wordmark (pure).

    Opens the landing page on an Earth-from-space note — the baked NASA texture
    on a shaded sphere, left of the app name (cool-blue→orange-red gradient)
    and the muted tagline. Self-contained (data-URI texture, no fetch).
    """
    return (
        '<div style="display:flex;align-items:center;gap:1rem;margin:0.25rem 0 0.85rem;">'
        f"{_earth_sphere_span(66, spin=False)}"
        "<div>"
        f'<div class="climate-title" style="font-size:1.9rem;">{name}</div>'
        f'<div style="color:{MUTED};font-size:1.02rem;margin-top:0.15rem;">{tagline}</div>'
        "</div></div>"
    )


def top_bar_html(brand_mark: str, nav_items, page_title: str | None = None) -> str:
    """The slim branded top bar (pure; self-contained).

    ``brand_mark`` is a pre-built inline ``<img>`` tag (the Rusty Data steward
    mark, a data-URI — same asset as the footer). ``nav_items`` is a sequence
    of ``(label, href)`` for the transparency menu. When ``page_title`` is set
    (the chat view), the app title shows compactly in the bar beside the brand
    (owner ask: the title moves into the header once a chat has started).
    """
    title_html = f'<span class="climate-title">{page_title}</span>' if page_title else ""
    nav = "".join(f'<a href="{href}" target="_self">{label}</a>' for label, href in nav_items)
    return (
        '<div class="climate-topbar">'
        '<div class="climate-brandwrap">'
        f'<span class="climate-brand">{brand_mark}<span>Rusty Data</span></span>'
        f"{title_html}"
        "</div>"
        f'<nav class="climate-nav">{nav}</nav>'
        "</div>"
    )


def inject_theme() -> None:
    """Inject the theme CSS once per rerun (the only Streamlit touch here)."""
    import streamlit as st

    st.markdown(theme_style_block(), unsafe_allow_html=True)


def render_hero(name: str, tagline: str) -> None:
    """Draw the landing hero (real Earth + warm wordmark) via ``st.markdown``."""
    import streamlit as st

    st.markdown(hero_html(name, tagline), unsafe_allow_html=True)


def render_top_bar(brand_mark: str, nav_items, page_title: str | None = None) -> None:
    """Draw the branded top bar (brand + menu, + title on chat) via markdown."""
    import streamlit as st

    st.markdown(top_bar_html(brand_mark, nav_items, page_title), unsafe_allow_html=True)


def globe_loader_placeholder(message: str = "Consulting the evidence…"):
    """Show the spinning-Earth loader and return its placeholder.

    The caller shows it before streaming and clears it with ``.empty()`` once
    the answer has finished (the globe stands in for the default spinner during
    generation). Returning the placeholder keeps the *when-to-clear* decision
    with the shell, where the stream lifecycle lives.
    """
    import streamlit as st

    placeholder = st.empty()
    placeholder.markdown(globe_loader_html(message), unsafe_allow_html=True)
    return placeholder
