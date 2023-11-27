from typing import List
from dataclasses import dataclass


@dataclass
class Config:
    """Config class for sdvfinetune."""

    audio_featurizer: str
    model_uri: str
    lr: float
    accumulate_grad_batches: int
    batch_size: int
    num_workers: int
    adam_betas: List[float]

    profile: bool
    disable_wandb: bool
    load_from: str
    save_to: str
    compile: bool
    distributed: bool
