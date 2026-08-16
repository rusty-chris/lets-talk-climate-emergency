"""Stub Streamlit application backing the `ui` docker-compose service.

This is scaffolding only (issue #1): a placeholder landing page. Streamlit's
built-in `/_stcore/health` endpoint (returns 200 "ok") is what docker-compose
and CI smoke tests probe. The real UI (chat, chip mapping, badges, starter
topics) lands in issue #20 per DESIGN.md §7 / IMPLEMENTATION.md §1.
"""

import streamlit as st

st.set_page_config(page_title="Let's Talk About the Climate Emergency")
st.title("Let's Talk About the Climate Emergency")
st.write("This is a scaffolding stub (issue #1). The real chat UI lands in issue #20.")
