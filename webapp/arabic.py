# -*- coding: utf-8 -*-
"""Live inference + Streamlit tab for the arabic-error-classification task."""
from __future__ import annotations

import random
import re

import streamlit as st
import torch

from webapp import samples
from webapp.common import DEVICE, load_checkpoint, load_task_module

_models_mod = load_task_module("arabic-error-classification", "models.py", "arabic_models_mod")
_qalb_mod = load_task_module("arabic-error-classification", "qalb_diff.py", "arabic_qalb_diff_mod")
SequenceClassifier = _models_mod.SequenceClassifier
tokenize = _qalb_mod.tokenize

ARCHS = ["rnn", "gru", "lstm"]
WINDOW = 6  # must match prep_data.py's WINDOW
EDIT_START, EDIT_END = "✁", "✂"  # must match prep_data.py's sentinels
MARK_RE = re.compile(r"\*\*(.+?)\*\*")


def build_window(text: str) -> str | None:
    """Text must wrap the suspected error span in **double asterisks**."""
    m = MARK_RE.search(text)
    if not m:
        return None
    before, span, after = tokenize(text[:m.start()]), tokenize(m.group(1)), tokenize(text[m.end():])
    if not span:
        return None
    marked = before[-WINDOW:] + [EDIT_START] + span + [EDIT_END] + after[:WINDOW]
    return " ".join(marked)


@st.cache_resource(show_spinner=False)
def load_model(arch: str):
    ckpt = load_checkpoint("arabic-error-classification", f"{arch}.pt")
    if ckpt is None:
        return None
    model = SequenceClassifier(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return {
        "model": model, "stoi": ckpt["stoi"], "labels": ckpt["labels"],
        "test_accuracy": ckpt.get("test_accuracy"),
    }


def predict(arch: str, window_text: str):
    bundle = load_model(arch)
    if bundle is None:
        return None
    model, stoi, labels = bundle["model"], bundle["stoi"], bundle["labels"]
    ids = [stoi.get(c, stoi.get("<unk>", 1)) for c in window_text] or [stoi.get("<unk>", 1)]
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, lengths)
    probs = torch.softmax(logits[0], dim=-1).cpu()
    order = probs.argsort(descending=True).tolist()
    return [(labels[i], probs[i].item()) for i in order]


def render_tab():
    st.subheader("Arabic Error-Type Classification")
    st.caption(
        "Character-level classification of the QALB grammar-error type inside a marked span "
        "(punctuation, hamza_alef, split, other, char_ins_del, ta_marbuta). "
        "Wrap the suspected error in **double asterisks**, e.g. هذا **الكتاب** جميل."
    )

    if "arabic_text" not in st.session_state:
        st.session_state.arabic_text = samples.ARABIC_SAMPLES[0]

    def _fill():
        st.session_state.arabic_text = random.choice(samples.ARABIC_SAMPLES)

    col1, col2 = st.columns([5, 1])
    with col1:
        text = st.text_input("Arabic text with **error** marked", key="arabic_text")
    with col2:
        st.write("")
        st.button("🎲 Random sample", key="arabic_random", on_click=_fill, use_container_width=True)

    available = [a for a in ARCHS if load_model(a) is not None]
    if not available:
        st.warning(
            "No trained checkpoints found yet — this task needs the QALB dataset (free "
            "registration required; its license forbids redistribution, see the main README). "
            "Once you have it under `arabic-error-classification/data/raw/qalb/`, run "
            "`python prep_data.py` then `python train.py --arch rnn` (and `gru`, `lstm`) inside "
            "`arabic-error-classification/`, and this tab will light up automatically."
        )
        return

    if not text.strip():
        st.info("Type Arabic text with the error wrapped in **double asterisks**, or click Random sample.")
        return

    window_text = build_window(text)
    if window_text is None:
        st.error("Mark the error span with **double asterisks** around it, e.g. هذا **الكتاب** جميل.")
        return

    st.caption(f"Model input window: `{window_text}`")

    cols = st.columns(3)
    for col, arch in zip(cols, ARCHS):
        with col:
            st.markdown(f"**{arch.upper()}**")
            bundle = load_model(arch)
            if bundle is None:
                st.caption("checkpoint not found")
                continue
            if bundle.get("test_accuracy") is not None:
                st.caption(f"test acc {bundle['test_accuracy']:.3f}")
            preds = predict(arch, window_text)
            for label, prob in preds[:3]:
                st.progress(min(prob, 1.0), text=f"{label}  ({prob:.1%})")
