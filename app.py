# -*- coding: utf-8 -*-
"""
Live demo: RNN vs GRU vs LSTM, tested side by side on the three tasks in
this repo. Run from the repo root:

    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from webapp import arabic, ner, next_word

st.set_page_config(
    page_title="Recurrent Architectures Benchmark",
    page_icon="🔁",
    layout="wide",
)

st.title("🔁 Recurrent Architectures Benchmark — Live Demo")
st.caption(
    "RNN, GRU, and LSTM, trained identically within each task, tested live side by side. "
    "See the [full write-up](https://github.com/mohamedtaha77/Recurrent-Architectures-Benchmark) "
    "for the benchmark results this app lets you probe interactively."
)

tab_ner, tab_nwp, tab_arabic = st.tabs([
    "🏷️ Named Entity Recognition",
    "✍️ Next-Word Prediction",
    "🇸🇦 Arabic Error Classification",
])

with tab_ner:
    ner.render_tab()

with tab_nwp:
    next_word.render_tab()

with tab_arabic:
    arabic.render_tab()
