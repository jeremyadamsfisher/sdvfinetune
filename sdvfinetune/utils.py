import os
import re
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pytorch_lightning as L
import wandb
from gpt import PROJECT_ID, VERSION
from gpt.config import Config
from loguru import logger


def restore_config(node):
    """Restore the config from a checkpoint, since it gets mangled by PyTorch Lightning."""
    if isinstance(node, dict):
        return {k: restore_config(v) for k, v in node.items()}
    elif isinstance(node, (list, tuple)):
        return [restore_config(x) for x in node]
    else:
        # See: https://github.com/omry/omegaconf/blob/master/omegaconf/nodes.py#L21
        return getattr(node, "_val", node)


def get_run_name(load_from: Optional[str]):
    """Generate a run name for wandb. If load_from is provided, use the
    run name which is the parent directory of the checkpoint."""

    if load_from:
        return Path(load_from).parent.name
    else:
        return f"run-v{VERSION}-{uuid4()}"


def get_rank_zero_or_single_gpu():
    """Return whether the current process is the rank zero process."""
    return os.environ.get("LOCAL_RANK", "0") == "0"


def rank_zero_only(f):
    if get_rank_zero_or_single_gpu():
        return f
    else:
        return lambda *args, **kwargs: None


@contextmanager
def run_manager(disable_wandb, load_from):
    """Return a context manager for running the model and determining
    the run name.

    Args:
        disable_wandb: whether to disable wandb
        load_from: path to a checkpoint to load from
    """
    name = get_run_name(load_from)
    if get_rank_zero_or_single_gpu():
        ctx = (
            nullcontext
            if disable_wandb
            else lambda: wandb.init(project=PROJECT_ID, name=name)
        )
        with ctx():
            yield name
    else:
        yield name
