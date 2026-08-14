# -*- coding: utf-8 -*-
"""
Train one of {rnn, gru, lstm} on the Task 1 dataset and report everything
the assignment's "Required Comparison" section asks for: performance
metrics, parameter count, training time, inference time, memory usage,
train/val curves, and error examples.

Everything but --arch is a fixed default shared by all three runs (same
data, vocab-building rule, embedding dim, hidden size, optimizer, lr, batch
size, epochs, dropout) so the only thing that differs between runs is the
recurrent cell.

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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ClassificationDataset, build_vocab, collate, read_labels, read_tsv
from models import SequenceClassifier

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def run_epoch(model, loader, device, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n, correct = 0.0, 0, 0
    for input_ids, lengths, labels, _ in loader:
        input_ids, lengths, labels = input_ids.to(device), lengths, labels.to(device)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["rnn", "gru", "lstm"])
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = read_labels()
    label2idx = {l: i for i, l in enumerate(labels)}
    train_rows, val_rows, test_rows = read_tsv("train"), read_tsv("val"), read_tsv("test")

    stoi, itos = build_vocab(train_rows, min_freq=a.min_freq)
    print(f"[{a.arch}] vocab size {len(itos)}, train/val/test = "
          f"{len(train_rows)}/{len(val_rows)}/{len(test_rows)}")

    train_ds = ClassificationDataset(train_rows, stoi, label2idx)
    val_ds = ClassificationDataset(val_rows, stoi, label2idx)
    test_ds = ClassificationDataset(test_rows, stoi, label2idx)
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate)

    model = SequenceClassifier(
        vocab_size=len(itos), num_classes=len(labels), cell=a.arch,
        emb_dim=a.emb_dim, hidden_dim=a.hidden_dim, num_layers=a.num_layers,
        dropout=a.dropout, bidirectional=a.bidirectional,
    ).to(device)
    n_params = model.num_params()
    print(f"[{a.arch}] {n_params:,} trainable params")

    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    t0 = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        ep_t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_dl, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_dl, device, criterion, optimizer=None)
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
    model.eval()
    all_preds, all_labels, all_texts = [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_t0 = time.perf_counter()
    with torch.no_grad():
        for input_ids, lengths, labels_b, texts in test_dl:
            input_ids = input_ids.to(device)
            logits = model(input_ids, lengths)
            preds = logits.argmax(-1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels_b.tolist())
            all_texts.extend(texts)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_time = time.perf_counter() - inf_t0
    per_example_ms = 1000 * inf_time / len(test_ds)

    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix, precision_recall_fscore_support)

    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=labels,
                                     zero_division=0, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    errors = []
    for text, true_i, pred_i in zip(all_texts, all_labels, all_preds):
        if true_i != pred_i:
            errors.append({"text": text, "true": labels[true_i], "pred": labels[pred_i]})

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
        "inference_ms_per_example": per_example_ms,
        "test_size": len(test_ds),
        "test_accuracy": acc,
        "test_precision_macro": p_macro,
        "test_recall_macro": r_macro,
        "test_f1_macro": f1_macro,
        "per_class_report": report,
        "confusion_matrix": cm,
        "labels": labels,
        "errors_sample": errors[:40],
        "n_errors": len(errors),
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
        "labels": labels,
        "model_kwargs": {
            "vocab_size": len(itos), "num_classes": len(labels), "cell": a.arch,
            "emb_dim": a.emb_dim, "hidden_dim": a.hidden_dim,
            "num_layers": a.num_layers, "dropout": a.dropout,
            "bidirectional": a.bidirectional,
        },
        "test_accuracy": acc,
        "test_f1_macro": f1_macro,
    }, ckpt_path)
    print(f"[{a.arch}] wrote {ckpt_path}")

    print(f"\n[{a.arch}] test_acc={acc:.4f} macro_P={p_macro:.4f} macro_R={r_macro:.4f} "
          f"macro_F1={f1_macro:.4f}  train_time={train_time:.1f}s "
          f"inference={per_example_ms:.3f}ms/ex  params={n_params:,} "
          f"peak_mem={peak_mem_mb}")
    print(f"[{a.arch}] wrote {out_path}")


if __name__ == "__main__":
    main()
