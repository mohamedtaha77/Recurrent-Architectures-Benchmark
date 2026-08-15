# -*- coding: utf-8 -*-
"""
Train one of {rnn, gru, lstm} at one context length on WikiText-2 next-word
prediction, and report loss, perplexity, next-word accuracy, parameter
count, training time, inference time, memory usage, train/val curves, and
error examples.

Everything but --arch and --seq-len stays fixed across all 15 runs (same
data, same vocab-building rule, embedding dim, hidden size, optimizer,
learning rate, batch size, epoch count, dropout). Sequence length is the
one variable the assignment wants swept on purpose, so it's the only other
thing that changes between runs.

Usage (from this folder):
    python train.py --arch rnn --seq-len 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (WindowDataset, build_vocab_from_text, collate, encode_text, read_text,
                     stride_for_target_count)
from models import WordLM

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def run_epoch(model, loader, device, criterion, optimizer=None, clip_grad=0.0):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, n, correct = 0.0, 0, 0
    for x, y, _, _ in loader:
        x, y = x.to(device), y.to(device)
        if train_mode:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["rnn", "gru", "lstm"])
    ap.add_argument("--seq-len", required=True, type=int)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-train", type=int, default=20000,
                     help="window stride is chosen so every seq_len gets about this many "
                          "training examples, keeping data volume comparable across lengths")
    ap.add_argument("--target-eval", type=int, default=3000)
    ap.add_argument("--out-suffix", default="",
                     help="Appended to the results/checkpoint filename, e.g. _wt103, so a "
                          "differently-configured run (bigger dataset, more epochs, ...) "
                          "doesn't overwrite the original benchmark's results/checkpoint.")
    ap.add_argument("--tie-weights", action="store_true", default=False,
                     help="Share the embedding and output-projection matrix (requires "
                          "emb_dim == hidden_dim). Standard in GPT-2 and most modern LMs; "
                          "cuts the dominant parameter cost roughly in half.")
    ap.add_argument("--clip-grad", type=float, default=0.0,
                     help="Max gradient norm; 0 disables clipping.")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                     help="Stop after this many epochs with no val_loss improvement; 0 "
                          "disables early stopping and always runs the full --epochs. "
                          "Either way, the checkpoint saved is the best val_loss epoch "
                          "seen, not necessarily the last one.")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_text, val_text, test_text = read_text("train"), read_text("val"), read_text("test")
    stoi, itos = build_vocab_from_text(train_text, min_freq=a.min_freq)
    tag = f"{a.arch}_{a.seq_len}{a.out_suffix}"
    print(f"[{tag}] vocab size {len(itos)}")

    train_ids = encode_text(train_text, stoi)
    val_ids = encode_text(val_text, stoi)
    test_ids = encode_text(test_text, stoi)
    del train_text, val_text, test_text  # free the raw text now that it's encoded to ids

    train_stride = stride_for_target_count(len(train_ids), a.seq_len, a.target_train)
    val_stride = stride_for_target_count(len(val_ids), a.seq_len, a.target_eval)
    test_stride = stride_for_target_count(len(test_ids), a.seq_len, a.target_eval)
    train_ds = WindowDataset(train_ids, itos, a.seq_len, stride=train_stride)
    val_ds = WindowDataset(val_ids, itos, a.seq_len, stride=val_stride)
    test_ds = WindowDataset(test_ids, itos, a.seq_len, stride=test_stride)
    print(f"[{tag}] train/val/test windows = {len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
          f"(strides {train_stride}/{val_stride}/{test_stride})")

    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=a.batch_size, shuffle=False, collate_fn=collate)

    model = WordLM(
        vocab_size=len(itos), cell=a.arch, emb_dim=a.emb_dim,
        hidden_dim=a.hidden_dim, num_layers=a.num_layers, dropout=a.dropout,
        tie_weights=a.tie_weights,
    ).to(device)
    n_params = model.num_params()
    print(f"[{tag}] {n_params:,} trainable params")

    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    t0 = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        ep_t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_dl, device, criterion, optimizer,
                                           clip_grad=a.clip_grad)
        val_loss, val_acc = run_epoch(model, val_dl, device, criterion, optimizer=None)
        ep_time = time.perf_counter() - ep_t0
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "train_ppl": math.exp(min(train_loss, 20)),
                         "val_loss": val_loss, "val_acc": val_acc,
                         "val_ppl": math.exp(min(val_loss, 20)), "epoch_seconds": ep_time})
        print(f"[{tag}] epoch {epoch}/{a.epochs}  train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_ppl={history[-1]['val_ppl']:.1f} "
              f"({ep_time:.1f}s)")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if a.early_stop_patience and epochs_without_improvement >= a.early_stop_patience:
            print(f"[{tag}] no val_loss improvement for {a.early_stop_patience} epochs, "
                  f"stopping early at epoch {epoch} (best was epoch {best_epoch})")
            break
    train_time = time.perf_counter() - t0

    if best_state is not None and best_epoch != epoch:
        print(f"[{tag}] restoring epoch {best_epoch}'s weights (best val_loss={best_val_loss:.4f}) "
              f"over epoch {epoch}'s")
        model.load_state_dict(best_state)

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                   if device.type == "cuda" else None)

    # ---- inference timing + predictions on the held-out test set ----
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_t0 = time.perf_counter()
    total_loss, correct, n = 0.0, 0, 0
    errors = []
    with torch.no_grad():
        for x, y, contexts, targets in test_dl:
            x_dev, y_dev = x.to(device), y.to(device)
            logits = model(x_dev)
            loss = criterion(logits, y_dev)
            preds = logits.argmax(-1)
            total_loss += loss.item() * y.size(0)
            correct += (preds.cpu() == y).sum().item()
            n += y.size(0)
            if len(errors) < 40:
                pred_words = [itos[p] for p in preds.cpu().tolist()]
                for ctx, tgt, pred in zip(contexts, targets, pred_words):
                    if tgt != pred and len(errors) < 40:
                        errors.append({"context": " ".join(ctx), "target": tgt, "predicted": pred})
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_time = time.perf_counter() - inf_t0

    test_loss = total_loss / n
    test_acc = correct / n
    test_ppl = math.exp(min(test_loss, 20))
    per_example_ms = 1000 * inf_time / n

    result = {
        "arch": a.arch,
        "seq_len": a.seq_len,
        "hyperparams": vars(a),
        "vocab_size": len(itos),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "train_stride": train_stride,
        "n_params": n_params,
        "device": str(device),
        "history": history,
        "train_time_seconds": train_time,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "peak_gpu_memory_mb": peak_mem_mb,
        "inference_time_seconds_full_test_set": inf_time,
        "inference_ms_per_example": per_example_ms,
        "n_test_examples": n,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_perplexity": test_ppl,
        "errors_sample": errors,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{tag}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "arch": a.arch,
        "seq_len": a.seq_len,
        "stoi": stoi,
        "itos": itos,
        "model_kwargs": {
            "vocab_size": len(itos), "cell": a.arch, "emb_dim": a.emb_dim,
            "hidden_dim": a.hidden_dim, "num_layers": a.num_layers,
            "dropout": a.dropout, "tie_weights": a.tie_weights,
        },
        "test_perplexity": test_ppl,
        "test_accuracy": test_acc,
    }, ckpt_path)
    print(f"[{tag}] wrote {ckpt_path}")

    print(f"\n[{tag}] test_loss={test_loss:.4f} test_ppl={test_ppl:.1f} test_acc={test_acc:.4f} "
          f"train_time={train_time:.1f}s inference={per_example_ms:.4f}ms/example "
          f"params={n_params:,} peak_mem={peak_mem_mb}")
    print(f"[{tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
