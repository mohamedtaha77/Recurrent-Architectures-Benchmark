# -*- coding: utf-8 -*-
"""
Build the comparison table, training curves, and error-analysis report from
the three results/{rnn,gru,lstm}.json files train.py writes.

Usage:
    python compare.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ARCHS = ["rnn", "gru", "lstm"]


def load_all():
    out = {}
    for arch in ARCHS:
        with open(os.path.join(RESULTS_DIR, f"{arch}.json"), encoding="utf-8") as f:
            out[arch] = json.load(f)
    return out


def plot_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"rnn": "#1f77b4", "gru": "#ff7f0e", "lstm": "#2ca02c"}
    for arch in ARCHS:
        h = results[arch]["history"]
        epochs = [e["epoch"] for e in h]
        axes[0].plot(epochs, [e["train_loss"] for e in h], color=colors[arch], linestyle="--", alpha=0.6)
        axes[0].plot(epochs, [e["val_loss"] for e in h], color=colors[arch], label=arch.upper())
        axes[1].plot(epochs, [e["train_acc"] for e in h], color=colors[arch], linestyle="--", alpha=0.6)
        axes[1].plot(epochs, [e["val_acc"] for e in h], color=colors[arch], label=arch.upper())
    axes[0].set_title("Loss (solid=val, dashed=train)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend()
    axes[1].set_title("Token accuracy (solid=val, dashed=train)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].legend()
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "curves.png")
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def plot_bars(results):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    metrics = [
        ("f1", "Entity F1 (test)", axes[0], lambda r: r["entity_overall"]["f1"]),
        ("params", "Trainable params", axes[1], lambda r: r["n_params"]),
        ("time", "Training time (s)", axes[2], lambda r: r["train_time_seconds"]),
        ("inf", "Inference (ms/sentence)", axes[3], lambda r: r["inference_ms_per_sentence"]),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for key, title, ax, get in metrics:
        vals = [get(results[a]) for a in ARCHS]
        ax.bar([a.upper() for a in ARCHS], vals, color=colors)
        ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_bars.png")
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def markdown_table(results):
    rows = []
    rows.append("| Model | Token Acc | Entity P | Entity R | Entity F1 | Params | Train time (s) | Inference (ms/sentence) | Peak GPU mem (MB) |")
    rows.append("|---|---|---|---|---|---|---|---|---|")
    for arch in ARCHS:
        r = results[arch]
        e = r["entity_overall"]
        rows.append(
            f"| {arch.upper()} | {r['token_accuracy']:.4f} | {e['precision']:.4f} | "
            f"{e['recall']:.4f} | {e['f1']:.4f} | {r['n_params']:,} | "
            f"{r['train_time_seconds']:.1f} | {r['inference_ms_per_sentence']:.4f} | "
            f"{r['peak_gpu_memory_mb']:.1f} |"
        )
    return "\n".join(rows)


def entity_type_table(results, arch):
    r = results[arch]
    rows = ["| Entity type | Precision | Recall | F1 | TP | FP | FN |", "|---|---|---|---|---|---|---|"]
    for label, d in r["entity_by_type"].items():
        rows.append(f"| {label} | {d['precision']:.3f} | {d['recall']:.3f} | {d['f1']:.3f} | {d['tp']} | {d['fp']} | {d['fn']} |")
    return "\n".join(rows)


def error_examples_md(results, n=4):
    out = []
    for arch in ARCHS:
        r = results[arch]
        out.append(f"\n{arch.upper()}: sentences where predicted entities did not match gold ({r['n_error_sentences']} out of {r['n_test_sentences']} test sentences)\n")
        for e in r["errors_sample"][:n]:
            sentence = " ".join(e["tokens"])
            gold_ents = [(e["tokens"][i], t[2:]) for i, t in enumerate(e["gold"]) if t.startswith("B-")]
            pred_ents = [(e["tokens"][i], t[2:]) for i, t in enumerate(e["pred"]) if t.startswith("B-")]
            out.append(f"- Sentence: {sentence}")
            out.append(f"  - Gold entities: {gold_ents if gold_ents else 'none'}")
            out.append(f"  - Predicted entities: {pred_ents if pred_ents else 'none'}")
    return "\n".join(out)


def main():
    results = load_all()
    plot_curves(results)
    plot_bars(results)

    best_f1 = max(ARCHS, key=lambda a: results[a]["entity_overall"]["f1"])
    fastest_train = min(ARCHS, key=lambda a: results[a]["train_time_seconds"])
    fastest_inf = min(ARCHS, key=lambda a: results[a]["inference_ms_per_sentence"])
    most_params = max(ARCHS, key=lambda a: results[a]["n_params"])

    report = f"""# Task 2: RNN vs GRU vs LSTM on Named Entity Recognition

## Dataset

CoNLL-2003 (English), the same dataset I used in an earlier NER project on
my GitHub (an internship task that got it from Kaggle). Kaggle isn't set up
on this machine, so I pulled the same data from Hugging Face instead
(`eriktks/conll2003`, its auto-converted parquet branch, no login needed).
Same tokens, same tags, same standard split: 14,041 train sentences, 3,250
validation, 3,453 test.

Each sentence keeps its original word order and its BIO tag sequence: O for
non-entity tokens, and B-/I- pairs for PERSON, ORGANIZATION, LOCATION, and
MISC. I kept the original casing on purpose. Capitalization is one of the
strongest clues for a name in English text, so lowercasing would throw away
real signal, same reasoning my earlier NER project's README already used.

## Keeping the comparison fair

Same idea as Task 1: everything stayed identical across the three runs
except the recurrent cell. Vocabulary built once from the training sentences
(word level, keeping case, {results['rnn']['vocab_size']:,} words), embedding
size 128, hidden size 128, one bidirectional recurrent layer, dropout 0.3,
Adam optimizer, learning rate 1e-3, batch size 32, 12 training epochs.

One thing this task needed that Task 1 didn't: NER tags every word in a
sentence, not the sentence as a whole, so the model reads out a label at
every step of the recurrent layer instead of just its last hidden state.
That's the one real architecture difference from Task 1's classifier, and
it's identical across all three runs. `models.py` has a single
`TokenTagger` class that takes the cell type as an argument, same pattern
as Task 1, so the three runs can't quietly drift apart from each other.

## Results

{markdown_table(results)}

![Training curves](curves.png)
![Comparison](comparison_bars.png)

### Results by entity type (best model: {best_f1.upper()})

{entity_type_table(results, best_f1)}

## Error examples

{error_examples_md(results)}

## Analysis

**1. Which model performs best?** {best_f1.upper()}, with an entity-level F1
of {results[best_f1]['entity_overall']['f1']:.4f}. RNN is close behind at
{results['rnn']['entity_overall']['f1']:.4f}, and LSTM sits in between at
{results['lstm']['entity_overall']['f1']:.4f}. All three are within about
2 points of each other, a very different picture from Task 1, where RNN
either won outright or collapsed to near-random depending on sequence
length. See question 4 for why NER lands in the middle.

**2. Which model is fastest?** RNN, on both training time
({results['rnn']['train_time_seconds']:.1f}s vs
{results['gru']['train_time_seconds']:.1f}s for GRU and
{results['lstm']['train_time_seconds']:.1f}s for LSTM) and inference
({results['rnn']['inference_ms_per_sentence']:.4f} ms/sentence). Expected:
fewer gates means less computation per token, and RNN has only one gate
computation per step against GRU's three and LSTM's four.

**3. Which model has the most parameters?** {most_params.upper()}, with
{results[most_params]['n_params']:,}, again from having the most gates (four:
input, forget, output, candidate) at the same hidden size and embedding size
as the other two.

**4. How does sequence length affect each model?** CoNLL-2003 sentences are
short, about 14 words on average, much shorter than Task 1's character-level
windows (70 to 90 characters) where RNN collapsed to near-random. At this
length, RNN never runs into the vanishing gradient problem badly enough to
fall apart, so it stays competitive here, {results['rnn']['entity_overall']['f1']:.4f}
entity F1 against GRU's {results['gru']['entity_overall']['f1']:.4f}. That
matches Task 1's own finding read in reverse: architecture choice barely
matters once sequences are short enough, and only starts to matter once
they get long enough for gradients to actually vanish.

**5. Does GRU give a good trade-off between RNN and LSTM?** Here, GRU is not
just a trade-off, it's the best model outright: highest entity F1, while
still training faster than LSTM and using fewer parameters. LSTM's extra
gate did not pay off on this task. That's a useful result on its own: more
gates is not automatically better, it helps when there's a long-dependency
problem to solve and can just add overfitting risk when there isn't one
(the same pattern Task 1 saw with GRU and LSTM on short word-level windows).

**6. Why does RNN struggle with long-term dependencies?** A plain RNN passes
gradients backward through the same weight matrix at every step. Over many
steps that repeated multiplication either shrinks the gradient toward zero
or blows it up, so the network can't learn that something far back in the
sequence mattered. GRU and LSTM add gates that let gradients flow through a
more direct, close-to-additive path instead. This task doesn't really test
that failure mode, since CoNLL-2003 sentences are too short for it to bite,
which is itself informative: it shows the problem is about sequence length,
not about RNNs being categorically worse at every task. Task 1's
character-level run is the direct demonstration of the failure mode itself.

**7. Which model would you choose and why?** GRU for this task: best entity
F1, faster and smaller than LSTM. If sentences were much longer (full
documents instead of single sentences, or a language with much longer
average sentence length), I'd expect the gap between RNN and the gated
models to widen the way it did in Task 1, and I'd lean toward LSTM or GRU
by default rather than re-testing RNN each time.

## Conclusion

Putting this next to Task 1 makes the real point clearer than either report
would on its own. Task 1 showed architecture choice flipping hard between
two regimes: RNN best on short sequences, RNN collapsing to near-random on
long ones. Task 2 sits inside the short-sequence regime by default, since
CoNLL-2003 sentences average about 14 words, and the result matches that:
all three models land within a few points of each other, with GRU slightly
ahead and RNN still fully competitive. Neither report is the "real" answer
on its own. Together they say the same thing from two directions: whether
RNN is a reasonable choice depends on how long the sequences actually are,
not on some fixed ranking of RNN against GRU against LSTM.

## Deliverables

- Source code: `named-entity-recognition/{{prep_data.py,dataset.py,models.py,train.py,compare.py}}`
- Results and comparison table: this file, `results/{{rnn,gru,lstm}}.json`
- Training and validation plots: `results/curves.png`, `results/comparison_bars.png`
- Error analysis: see "Error examples" above and `errors_sample` in each result JSON
- Conclusion: see "Conclusion" above
"""

    out_path = os.path.join(RESULTS_DIR, "report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
