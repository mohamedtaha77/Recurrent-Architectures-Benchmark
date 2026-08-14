# -*- coding: utf-8 -*-
"""
Shared plumbing for the live demo app: loads each task's models.py /
dataset.py / qalb_diff.py by file path (they live in sibling folders with
duplicate module names like "models.py", so plain imports would collide),
and loads the .pt checkpoints train.py now writes.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_task_module(task_dir: str, filename: str, unique_name: str):
    path = os.path.join(REPO_ROOT, task_dir, filename)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(task_dir: str, checkpoint_filename: str):
    path = os.path.join(REPO_ROOT, task_dir, "checkpoints", checkpoint_filename)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=DEVICE, weights_only=False)
