# -*- coding: utf-8 -*-
"""Live inference + Streamlit tab for the next-word-prediction task."""
from __future__ import annotations

import random
import re

import streamlit as st
import torch

from webapp import samples
from webapp.common import DEVICE, load_checkpoint, load_task_module

_models_mod = load_task_module("next-word-prediction", "models.py", "nwp_models_mod")
WordLM = _models_mod.WordLM

ARCHS = ["rnn", "gru", "lstm"]
SEQ_LENS = [10, 25, 50, 100, 200]
PUNCT_RE = re.compile(r"([.,!?;:()\"])")
SHOWCASE_SUFFIX = "_wt103"  # trained on WikiText-103, more data + epochs than the benchmark


def tokenize(text: str) -> list[str]:
    text = PUNCT_RE.sub(r" \1 ", text)
    return text.split()


@st.cache_resource(show_spinner=False, max_entries=6)
def load_model(arch: str, seq_len: int):
    # max_entries=6 (two context lengths' worth of models) caps memory on
    # small deploy tiers, since a user clicking through all 5 lengths would
    # otherwise keep all 15 checkpoints (~500MB) resident at once.
    #
    # Prefers the WikiText-103 "showcase" checkpoint (bigger data, more
    # epochs, trained purely for generation quality) over the original
    # WikiText-2 benchmark checkpoint (the one results/report.md and the
    # README's numbers are based on), falling back to the benchmark
    # checkpoint if the showcase one hasn't been trained for this
    # (arch, seq_len) yet.
    ckpt = load_checkpoint("next-word-prediction", f"{arch}_{seq_len}{SHOWCASE_SUFFIX}.pt")
    variant = "showcase"
    if ckpt is None:
        ckpt = load_checkpoint("next-word-prediction", f"{arch}_{seq_len}.pt")
        variant = "benchmark"
    if ckpt is None:
        return None
    model = WordLM(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()
    return {
        "model": model, "stoi": ckpt["stoi"], "itos": ckpt["itos"],
        "test_perplexity": ckpt.get("test_perplexity"),
        "variant": variant,
    }


def predict_top_k(arch: str, seq_len: int, tokens: list[str], k: int = 5):
    bundle = load_model(arch, seq_len)
    if bundle is None or not tokens:
        return None
    model, stoi, itos = bundle["model"], bundle["stoi"], bundle["itos"]
    context = tokens[-seq_len:]
    ids = [stoi.get(t, stoi.get("<unk>", 1)) for t in context]
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(input_ids)
    probs = torch.softmax(logits[0], dim=-1).cpu()
    top_probs, top_ids = probs.topk(min(k, probs.numel()))
    return [(itos[i], p) for i, p in zip(top_ids.tolist(), top_probs.tolist())]


def generate(arch: str, seq_len: int, tokens: list[str], n_words: int = 10,
             temperature: float = 0.7, top_k: int = 8,
             repetition_penalty: float = 1.3, no_repeat_ngram_size: int = 3):
    """
    Autoregressive multi-word continuation: feed the model's own prediction
    back in as context, one word at a time, up to n_words.

    Sampled from only the top_k most likely words at each step (not the
    full ~33k-word vocab, and never <unk>), so one unlucky low-probability
    pick can't derail the rest of the continuation the way plain
    full-vocabulary sampling did. temperature reshapes the distribution
    *within* that shortlist: low = closer to always picking the single most
    likely word, high = closer to uniform among the shortlist.

    Small word-level LMs love to fall into loops ("United States United
    States United States"), same failure mode real LLM sampling guards
    against, so two standard inference-time fixes, no retraining needed:
    - repetition_penalty: softly discourages re-picking any word already
      generated (Keskar et al. CTRL-style: divide/multiply its logit).
    - no_repeat_ngram_size: hard-blocks any word that would recreate an
      n-gram that already appeared earlier in this continuation.
    """
    bundle = load_model(arch, seq_len)
    if bundle is None or not tokens:
        return None
    model, stoi, itos = bundle["model"], bundle["stoi"], bundle["itos"]
    unk_id = stoi.get("<unk>")
    generated = list(tokens)
    generated_ids = [stoi.get(t, stoi.get("<unk>", 1)) for t in generated]
    n = no_repeat_ngram_size
    for _ in range(n_words):
        context = generated[-seq_len:]
        ids = [stoi.get(t, stoi.get("<unk>", 1)) for t in context]
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = model(input_ids)[0]
        logits = logits.clone()
        if unk_id is not None:
            logits[unk_id] = float("-inf")

        if repetition_penalty and repetition_penalty != 1.0:
            for prev_id in set(generated_ids):
                if logits[prev_id] > 0:
                    logits[prev_id] /= repetition_penalty
                else:
                    logits[prev_id] *= repetition_penalty

        if n and len(generated_ids) >= n - 1:
            prefix = tuple(generated_ids[-(n - 1):])
            banned = {generated_ids[i + n - 1] for i in range(len(generated_ids) - n + 1)
                      if tuple(generated_ids[i:i + n - 1]) == prefix}
            for b in banned:
                logits[b] = float("-inf")

        k = min(top_k, logits.numel())
        top_logits, top_ids = logits.topk(k)
        probs = torch.softmax(top_logits / max(temperature, 1e-4), dim=-1)
        choice = int(torch.multinomial(probs, 1))
        next_id = int(top_ids[choice])
        generated.append(itos[next_id])
        generated_ids.append(next_id)
    return generated[len(tokens):]


def render_tab():
    st.subheader("Next-Word Prediction")
    st.caption(
        "WikiText-2 language modeling, trained at five context lengths. Pick a context length, "
        "type (or autofill) a prompt, and watch each model write a continuation, word by word, "
        "feeding its own output back in as context."
    )

    if "nwp_text" not in st.session_state:
        st.session_state.nwp_text = samples.NWP_SAMPLES[0]

    def _fill():
        st.session_state.nwp_text = random.choice(samples.NWP_SAMPLES)

    setting_cols = st.columns(4)
    with setting_cols[0]:
        seq_len = st.select_slider("Context length (checkpoint)", options=SEQ_LENS, value=50)
    with setting_cols[1]:
        n_words = st.slider("Words to generate", min_value=3, max_value=25, value=10)
    with setting_cols[2]:
        top_k = st.slider(
            "Word choices per step", min_value=1, max_value=30, value=6,
            help="Only sample from the model's top-K most likely next words at each step "
                 "(never the full ~33k vocab). 1 = always the single most likely word "
                 "(deterministic, safest, most repetitive). Higher = more variety, more risk "
                 "of a weird pick derailing the rest of the sentence.",
        )
    with setting_cols[3]:
        temperature = st.slider(
            "Sampling temperature", min_value=0.3, max_value=1.5, value=0.7, step=0.1,
            help="Reshapes probabilities within that shortlist. Lower = closer to always "
                 "picking the top choice. Higher = closer to uniform among the shortlist.",
        )

    col1, col2 = st.columns([5, 1])
    with col1:
        text = st.text_input("Prompt — each model continues from here", key="nwp_text")
    with col2:
        st.write("")
        c1, c2 = st.columns(2)
        c1.button("🎲", key="nwp_random", on_click=_fill, use_container_width=True, help="Random sample prompt")
        c2.button("🔁", key="nwp_regen", use_container_width=True, help="Regenerate (re-sample) with the same prompt")

    available = [a for a in ARCHS if load_model(a, seq_len) is not None]
    if not available:
        st.warning(
            f"No trained checkpoints found for context length {seq_len} yet. From "
            f"`next-word-prediction/`, run `python train.py --arch rnn --seq-len {seq_len}` "
            f"(and `gru`, `lstm`), then reload this page."
        )
        return

    if not text.strip():
        st.info("Type a prompt or click 🎲 for a random sample.")
        return

    tokens = tokenize(text)
    cols = st.columns(3)
    for col, arch in zip(cols, ARCHS):
        with col:
            st.markdown(f"**{arch.upper()}**")
            bundle = load_model(arch, seq_len)
            if bundle is None:
                st.caption("checkpoint not found")
                continue
            variant_label = "WikiText-103 showcase" if bundle.get("variant") == "showcase" else "WikiText-2 benchmark"
            if bundle.get("test_perplexity") is not None:
                st.caption(f"test perplexity {bundle['test_perplexity']:.1f} · {variant_label}")
            else:
                st.caption(variant_label)
            continuation = generate(arch, seq_len, tokens, n_words=n_words,
                                     temperature=temperature, top_k=top_k)
            if not continuation:
                st.caption("(empty context)")
                continue
            st.markdown(f"{text} **{' '.join(continuation)}**")
            with st.expander("Next single word, top-5"):
                top5 = predict_top_k(arch, seq_len, tokens, k=5)
                for word, prob in top5:
                    st.progress(min(prob, 1.0), text=f"{word}  ({prob:.1%})")
