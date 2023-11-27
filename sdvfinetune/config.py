from dataclasses import dataclass


@dataclass
class Config:
    """Config class for sdvfinetune."""

    scheduler: str
    audio_featurizer: str
    model_uri: str
    lr: float
    accumulate_grad_batches: int

    profile: bool
    disable_wandb: bool
    load_from: str
    save_to: str
    compile: bool
    distributed: bool
