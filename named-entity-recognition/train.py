# -*- coding: utf-8 -*-
"""
Train one of {rnn, gru, lstm} on CoNLL-2003 NER and report performance
metrics, entity-level F1, parameter count, training time, inference time,
memory usage, train/val curves, and error examples, the same set of things
arabic-error-classification's train.py reports.

Everything but --arch stays fixed across all three runs (same data, same
vocab-building rule, embedding dim, hidden size, optimizer, learning rate,
batch size, epoch count, dropout), so the only thing that changes between
runs is the recurrent cell.

Usage (from this folder):
    python train.py --arch rnn
    python train.py --arch gru
    python train.py --arch lstm
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import NERDataset, build_vocab, collate, entity_scores, read_jsonl, read_tags
from models import TokenTagger

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def run_epoch(model, loader, device, criterion, tag2idx, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n_tokens, correct = 0.0, 0, 0
    for input_ids, tags, lengths, _, _ in loader:
        input_ids, tags = input_ids.to(device), tags.to(device)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            logits = model(input_ids, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1))
            if train_mode:
                loss.backward()
                optimizer.step()
        mask = tags != tag2idx["PAD"]
        preds = logits.argmax(-1)
        correct += ((preds == tags) & mask).sum().item()
        n_tokens += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()
    return total_loss / n_tokens, correct / n_tokens


def predict_tags(model, loader, device, idx2tag):
    model.eval()
    all_gold, all_pred, all_tokens = [], [], []
    with torch.no_grad():
        for input_ids, tags, lengths, tokens, gold_tags in loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids, lengths)
            preds = logits.argmax(-1).cpu()
            for i, length in enumerate(lengths.tolist()):
                pred_tags = [idx2tag[t] for t in preds[i, :length].tolist()]
                all_pred.append(pred_tags)
                all_gold.append(list(gold_tags[i]))
                all_tokens.append(list(tokens[i]))
    return all_gold, all_pred, all_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["rnn", "gru", "lstm"])
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tags = read_tags()
    tag2idx = {t: i for i, t in enumerate(tags)}
    tag2idx["PAD"] = len(tags)
    idx2tag = {i: t for t, i in tag2idx.items()}

    train_rows, val_rows, test_rows = read_jsonl("train"), read_jsonl("val"), read_jsonl("test")
    stoi, itos = build_vocab(train_rows, min_freq=a.min_freq)
    print(f"[{a.arch}] vocab size {len(itos)}, train/val/test sentences = "
          f"{len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

    collate_fn = partial(collate, pad_tag_idx=tag2idx["PAD"])
    train_dl = DataLoader(NERDataset(train_rows, stoi, tag2idx), batch_size=a.batch_size,
                           shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(NERDataset(val_rows, stoi, tag2idx), batch_size=a.batch_size,
                         shuffle=False, collate_fn=collate_fn)
    test_dl = DataLoader(NERDataset(test_rows, stoi, tag2idx), batch_size=a.batch_size,
                          shuffle=False, collate_fn=collate_fn)

    model = TokenTagger(
        vocab_size=len(itos), num_tags=len(tags), cell=a.arch,
        emb_dim=a.emb_dim, hidden_dim=a.hidden_dim, num_layers=a.num_layers,
        dropout=a.dropout, bidirectional=a.bidirectional,
    ).to(device)
    n_params = model.num_params()
    print(f"[{a.arch}] {n_params:,} trainable params")

    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tag2idx["PAD"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        ep_t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_dl, device, criterion, tag2idx, optimizer)
        val_loss, val_acc = run_epoch(model, val_dl, device, criterion, tag2idx, optimizer=None)
        ep_time = time.perf_counter() - ep_t0
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc, "epoch_seconds": ep_time})
        print(f"[{a.arch}] epoch {epoch}/{a.epochs}  train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({ep_time:.1f}s)")
    train_time = time.perf_counter() - t0

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                   if device.type == "cuda" else None)

    # ---- inference timing + predictions on the held-out test set ----
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_t0 = time.perf_counter()
    gold_tags, pred_tags, tokens = predict_tags(model, test_dl, device, idx2tag)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_time = time.perf_counter() - inf_t0
    n_test_tokens = sum(len(g) for g in gold_tags)
    per_example_ms = 1000 * inf_time / len(test_rows)

    # ---- token-level metrics ----
    from sklearn.metrics import (accuracy_score, classification_report,
                                  precision_recall_fscore_support)

    flat_gold = [t for seq in gold_tags for t in seq]
    flat_pred = [t for seq in pred_tags for t in seq]
    token_acc = accuracy_score(flat_gold, flat_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        flat_gold, flat_pred, average="macro", zero_division=0, labels=tags)
    token_report = classification_report(flat_gold, flat_pred, labels=tags,
                                          zero_division=0, output_dict=True)

    # ---- entity-level metrics ----
    entity_overall, entity_by_type = entity_scores(gold_tags, pred_tags)

    # ---- error examples: sentences where predicted entities differ from gold ----
    errors = []
    for toks, g, p in zip(tokens, gold_tags, pred_tags):
        if g != p:
            errors.append({"tokens": toks, "gold": g, "pred": p})

    result = {
        "arch": a.arch,
        "hyperparams": vars(a),
        "vocab_size": len(itos),
        "n_params": n_params,
        "device": str(device),
        "history": history,
        "train_time_seconds": train_time,
        "peak_gpu_memory_mb": peak_mem_mb,
        "inference_time_seconds_full_test_set": inf_time,
        "inference_ms_per_sentence": per_example_ms,
        "n_test_sentences": len(test_rows),
        "n_test_tokens": n_test_tokens,
        "token_accuracy": token_acc,
        "token_precision_macro": p_macro,
        "token_recall_macro": r_macro,
        "token_f1_macro": f1_macro,
        "token_report": token_report,
        "entity_overall": entity_overall,
        "entity_by_type": entity_by_type,
        "tags": tags,
        "errors_sample": errors[:40],
        "n_error_sentences": len(errors),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{a.arch}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{a.arch}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "arch": a.arch,
        "stoi": stoi,
        "itos": itos,
        "tags": tags,
        "model_kwargs": {
            "vocab_size": len(itos), "num_tags": len(tags), "cell": a.arch,
            "emb_dim": a.emb_dim, "hidden_dim": a.hidden_dim,
            "num_layers": a.num_layers, "dropout": a.dropout,
            "bidirectional": a.bidirectional,
        },
        "token_accuracy": token_acc,
        "entity_f1": entity_overall["f1"],
    }, ckpt_path)
    print(f"[{a.arch}] wrote {ckpt_path}")

    print(f"\n[{a.arch}] token_acc={token_acc:.4f} token_macro_F1={f1_macro:.4f}  "
          f"entity_P={entity_overall['precision']:.4f} entity_R={entity_overall['recall']:.4f} "
          f"entity_F1={entity_overall['f1']:.4f}  train_time={train_time:.1f}s "
          f"inference={per_example_ms:.3f}ms/sentence  params={n_params:,} "
          f"peak_mem={peak_mem_mb}")
    print(f"[{a.arch}] wrote {out_path}")


if __name__ == "__main__":
    main()
