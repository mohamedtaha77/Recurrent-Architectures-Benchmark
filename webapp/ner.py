# -*- coding: utf-8 -*-
"""Live inference + Streamlit tab for the named-entity-recognition task."""
from __future__ import annotations

import random
import re

import streamlit as st
import torch

from webapp import samples
from webapp.common import DEVICE, load_checkpoint, load_task_module

_models_mod = load_task_module("named-entity-recognition", "models.py", "ner_models_mod")
_dataset_mod = load_task_module("named-entity-recognition", "dataset.py", "ner_dataset_mod")
TokenTagger = _models_mod.TokenTagger
bio_to_spans = _dataset_mod.bio_to_spans

ARCHS = ["rnn", "gru", "lstm"]
ENTITY_COLORS = {"PER": "#f87171", "ORG": "#60a5fa", "LOC": "#34d399", "MISC": "#fbbf24"}
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


@st.cache_resource(show_spinner=False)
def load_model(arch: str):
    ckpt = load_checkpoint("named-entity-recognition", f"{arch}.pt")
    if ckpt is None:
        return None
    model = TokenTagger(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return {
        "model": model, "stoi": ckpt["stoi"], "tags": ckpt["tags"],
        "token_accuracy": ckpt.get("token_accuracy"), "entity_f1": ckpt.get("entity_f1"),
    }


def predict(arch: str, tokens: list[str]):
    bundle = load_model(arch)
    if bundle is None or not tokens:
        return None
    model, stoi, tags = bundle["model"], bundle["stoi"], bundle["tags"]
    ids = [stoi.get(t, stoi.get("<unk>", 1)) for t in tokens]
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    lengths = torch.tensor([len(ids)], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids, lengths)
    probs = torch.softmax(logits[0], dim=-1).cpu()
    pred_ids = probs.argmax(-1).tolist()
    pred_tags = [tags[i] for i in pred_ids]
    confidences = [probs[i, pred_ids[i]].item() for i in range(len(pred_ids))]
    return list(zip(tokens, pred_tags, confidences))


def render_html(tokens: list[str], tags: list[str]) -> str:
    spans = bio_to_spans(tags)
    span_at = {}
    for start, end, label in spans:
        for i in range(start, end):
            span_at[i] = (label, i == end - 1)
    parts = []
    for i, tok in enumerate(tokens):
        if i in span_at:
            label, is_end = span_at[i]
            color = ENTITY_COLORS.get(label, "#a78bfa")
            tag_html = f'<sup style="font-size:0.65em;color:{color};font-weight:600;">{label}</sup>' if is_end else ""
            parts.append(
                f'<span style="background:{color}22;border:1px solid {color};'
                f'border-radius:4px;padding:1px 4px;margin:0 1px;">{tok}{tag_html}</span>'
            )
        else:
            parts.append(tok)
    return " ".join(parts)


def render_tab():
    st.subheader("Named Entity Recognition")
    st.caption(
        "CoNLL-2003 word-level tagging into PER / ORG / LOC / MISC. "
        "Type a sentence and compare how RNN, GRU, and LSTM tag it, side by side."
    )

    if "ner_text" not in st.session_state:
        st.session_state.ner_text = samples.NER_SAMPLES[0]

    def _fill():
        st.session_state.ner_text = random.choice(samples.NER_SAMPLES)

    col1, col2 = st.columns([5, 1])
    with col1:
        text = st.text_area("Sentence", key="ner_text", height=80)
    with col2:
        st.write("")
        st.button("🎲 Random sample", key="ner_random", on_click=_fill, use_container_width=True)

    available = [a for a in ARCHS if load_model(a) is not None]
    if not available:
        st.warning(
            "No trained NER checkpoints found yet. From `named-entity-recognition/`, run "
            "`python train.py --arch rnn`, `--arch gru`, `--arch lstm`, then reload this page."
        )
        return

    if not text.strip():
        st.info("Type a sentence or click Random sample.")
        return

    tokens = tokenize(text)
    cols = st.columns(3)
    for col, arch in zip(cols, ARCHS):
        with col:
            st.markdown(f"**{arch.upper()}**")
            bundle = load_model(arch)
            if bundle is None:
                st.caption("checkpoint not found")
                continue
            if bundle.get("token_accuracy") is not None:
                st.caption(f"token acc {bundle['token_accuracy']:.3f} · entity F1 {bundle['entity_f1']:.3f}")
            preds = predict(arch, tokens)
            html = render_html([p[0] for p in preds], [p[1] for p in preds])
            st.markdown(html, unsafe_allow_html=True)
