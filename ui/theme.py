"""Issue #398 — the planet-Earth-from-space visual theme (Path A).

Owner feedback (2026-09-14): the site "looks like it's from the 90s" and
Streamlit "feels clunky". Path A is the aggressive-Streamlit-theming route
(fast, reversible, keeps the Python UI): the base palette lives in
``.streamlit/config.toml`` and is reinforced here with one block of injected,
self-contained CSS plus a CSS/SVG **spinning-globe** loader that replaces the
default Streamlit spinner while an answer generates.

Self-contained by construction — the no-external-requests convention holds:
this module references NO remote host. Every colour is a hex literal, every
texture a CSS gradient, the globe an inline ``<svg>`` spun by a CSS
``@keyframes`` (mirroring how ``ui.footer`` inlines the steward mark as a
data-URI rather than fetching it). ``tests/unit/test_ui_theme.py`` guards
that no ``http``/protocol-relative URL ever leaks into the injected markup.

The split follows the #18 shell/core discipline: the string builders
(:func:`theme_style_block`, :func:`globe_loader_html`, :func:`hero_html`) are
pure and import nothing, so they are testable without Streamlit; the thin
``inject_*`` / ``render_*`` helpers import ``streamlit`` locally and are the
only Streamlit-touching code here.
"""

from __future__ import annotations

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

#: One authoritative CSS block. Scoped to Streamlit's stable structural
#: selectors (``stApp``/``stButton``/``stMetric`` …); everything degrades to
#: the config.toml palette if a Streamlit release renames a class, so a
#: missed selector dulls the polish but never breaks a feature.
_CSS = f"""
/* Atmospheric canvas: a subtle "Earth limb" glow bottom-centre plus a faint
   high-atmosphere haze top-right, painted with layered radial gradients over
   the deep-space base — no image is fetched. */
.stApp {{
  background:
    radial-gradient(120% 90% at 50% 118%, {ATMOSPHERE}26 0%, {ATMOSPHERE}0d 26%, transparent 55%),
    radial-gradient(90% 70% at 88% -10%, {TEAL}1f 0%, transparent 45%),
    linear-gradient(180deg, {SPACE} 0%, {OCEAN_DEEP} 100%) fixed;
  color: {TEXT};
}}

/* Hero title: the app name as a cyan→teal gradient wordmark, tagline muted.
   Applies to the landing <h1>/<h3> Streamlit renders from st.title/subheader. */
.stApp h1 {{
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
  background: linear-gradient(90deg, {TEXT} 0%, {ATMOSPHERE} 55%, {TEAL} 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.stApp h3 {{ color: {MUTED}; font-weight: 500; }}
.stApp h2, .stApp h4 {{ color: {TEXT}; }}

/* Starter buttons / actions: glassy ocean chips with a cyan hairline that
   lights up on hover — no longer flat default-Streamlit grey. */
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

/* Bordered containers (the #402 footprint panel) & metrics: ocean glass with
   a cyan-tinted edge so panels read as instrument readouts, not grey boxes. */
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

/* The globe loader — a CSS-spun inline-SVG Earth. Keyframes only; the SVG
   itself is emitted by globe_loader_html(). Two motions: the marble rotates,
   a faint atmosphere ring pulses. Honours reduced-motion. */
@keyframes climate-globe-spin {{ to {{ transform: rotate(360deg); }} }}
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
.climate-globe .marble {{
  position: absolute; inset: 0;
  animation: climate-globe-spin 3.2s linear infinite;
  transform-origin: 50% 50%;
}}
@media (prefers-reduced-motion: reduce) {{
  .climate-globe .marble {{ animation: none; }}
  .climate-globe .atmos {{ animation: none; opacity: .6; }}
}}
"""

#: The spinning marble as an inline SVG: ocean disc, a couple of abstract
#: land masses, a specular highlight, and a graticule — no glyph, no fetch, so
#: it renders identically on every platform (an emoji 🌍 would vary by OS).
_GLOBE_SVG = f"""
<span class="climate-globe" role="img" aria-label="Generating">
  <span class="atmos"></span>
  <svg class="marble" viewBox="0 0 100 100" width="46" height="46" aria-hidden="true">
    <defs>
      <radialGradient id="climate-ocean" cx="38%" cy="34%" r="75%">
        <stop offset="0%" stop-color="{TEAL}"/>
        <stop offset="55%" stop-color="{ATMOSPHERE}"/>
        <stop offset="100%" stop-color="{OCEAN_DEEP}"/>
      </radialGradient>
    </defs>
    <circle cx="50" cy="50" r="46" fill="url(#climate-ocean)"/>
    <g fill="{LAND}" opacity="0.9">
      <path d="M22 40 q10 -12 24 -6 q8 4 4 14 q-6 12 -20 8 q-14 -4 -8 -16 Z"/>
      <path d="M60 30 q12 -4 16 6 q3 10 -8 12 q-12 3 -12 -8 q0 -8 4 -10 Z"/>
      <path d="M52 60 q14 -2 16 10 q1 12 -12 12 q-14 0 -12 -14 q1 -6 8 -8 Z"/>
    </g>
    <circle cx="50" cy="50" r="46" fill="none"
      stroke="{SPACE}" stroke-opacity="0.25" stroke-width="0.8"/>
    <ellipse cx="50" cy="50" rx="46" ry="18" fill="none"
      stroke="{TEXT}" stroke-opacity="0.18" stroke-width="0.7"/>
    <line x1="50" y1="4" x2="50" y2="96"
      stroke="{TEXT}" stroke-opacity="0.14" stroke-width="0.7"/>
    <circle cx="34" cy="30" r="14" fill="{TEXT}" opacity="0.12"/>
  </svg>
</span>
"""


def theme_style_block() -> str:
    """The whole theme CSS as a single ``<style>`` block (pure; self-contained)."""
    return f"<style>{_CSS}</style>"


def globe_loader_html(message: str = "Consulting the evidence…") -> str:
    """The spinning-globe 'generating' indicator markup (pure; self-contained).

    Replaces the default Streamlit spinner while an answer streams. The label
    is app-authored copy, not model content.
    """
    return f'<div class="climate-globe-loader">{_GLOBE_SVG}<span>{message}</span></div>'


def hero_html(name: str, tagline: str) -> str:
    """A compact planet motif beside the landing title (pure; self-contained).

    A CSS radial-gradient 'planet' with an atmospheric halo — no image fetch —
    sitting left of the app name and tagline, so the landing page opens on an
    Earth-from-space note rather than a bare Streamlit heading.
    """
    return (
        '<div style="display:flex;align-items:center;gap:1rem;margin:0.25rem 0 0.75rem;">'
        '<div style="width:64px;height:64px;flex:0 0 auto;border-radius:50%;'
        "background:radial-gradient(circle at 34% 32%,"
        f" {TEAL} 0%, {ATMOSPHERE} 42%, {OCEAN_DEEP} 100%);"
        f"box-shadow:0 0 0 4px {ATMOSPHERE}26, 0 0 26px {ATMOSPHERE}59,"
        f' inset -8px -8px 18px {SPACE}99;"></div>'
        "<div>"
        f'<div style="font-size:1.9rem;font-weight:800;line-height:1.1;'
        f"background:linear-gradient(90deg,{TEXT} 0%,{ATMOSPHERE} 55%,{TEAL} 100%);"
        '-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;">'
        f"{name}</div>"
        f'<div style="color:{MUTED};font-size:1.02rem;margin-top:0.15rem;">{tagline}</div>'
        "</div></div>"
    )


def inject_theme() -> None:
    """Inject the theme CSS once per rerun (the only Streamlit touch here)."""
    import streamlit as st

    st.markdown(theme_style_block(), unsafe_allow_html=True)


def render_hero(name: str, tagline: str) -> None:
    """Draw the landing hero (planet motif + wordmark) via ``st.markdown``."""
    import streamlit as st

    st.markdown(hero_html(name, tagline), unsafe_allow_html=True)


def globe_loader_placeholder(message: str = "Consulting the evidence…"):
    """Show the spinning-globe loader and return its placeholder.

    The caller shows it before streaming and clears it with ``.empty()`` once
    the answer has finished (the globe stands in for the default spinner during
    generation). Returning the placeholder keeps the *when-to-clear* decision
    with the shell, where the stream lifecycle lives.
    """
    import streamlit as st

    placeholder = st.empty()
    placeholder.markdown(globe_loader_html(message), unsafe_allow_html=True)
    return placeholder
