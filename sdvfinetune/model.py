from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from transformers import Wav2Vec2Processor
from transformers import logging as tsmrs_logging


tsmrs_logging.set_verbosity_error()


VAE_TO_UNET_SCALING_FACTOR = 0.18215


class AudioAdapter(nn.Module):
    """A simple audio adapter that takes in an audio signal outputs something
    like the CLIP text encoder."""
    def __init__(self, input_channels, output_channels, n_middle_layers=1, time_downsample_rate=0.1):
        super().__init__()
        self.conv1 = nn.Linear(input_channels, output_channels)
        self.relu1 = nn.ReLU()
        middle = []
        for _ in range(n_middle_layers):
            middle.extend((nn.Linear(output_channels, output_channels), nn.ReLU()))
        self.middle = nn.Sequential(*middle)
        self.pool = nn.MaxPool2d(kernel_size=(int(1/time_downsample_rate), 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.middle(x)
        x = self.pool(x)
        return x


class SVDLightning(pl.LightningModule):
    """A PyTorch Lightning module for training a Stable Diffusion Video model."""
    def __init__(
        self,
        audio_featurizer: Wav2Vec2Processor,
        scheduler: Any,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        adaptor=None,
        lr=None,
        betas=None,
    ):
        super().__init__()
        self.audio_featurizer = audio_featurizer
        self.scheduler = scheduler
        self.unet = unet
        self.vae = vae
        self.adaptor = adaptor
        self.lr = lr
        self.betas = betas

    @classmethod
    def from_config(cls, config):
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        return cls(
            audio_featurizer=Wav2Vec2Processor.from_pretrained(config.audio_featurizer),
            scheduler=pipe.scheduler,
            unet=pipe.unet,
            vae=pipe.vae,
            lr=config.lr,
            betas=config.adam_betas,
        )

    def _step(self, batch):
        # Get an arbitrary video frame, until SVD is released
        video, audio = batch
        BS, C, H, W, T = video.shape
        frames = video[..., 0]
        assert frames.shape == (BS, C, H, W)

        # Encode the video frames
        with torch.no_grad():
            frames = frames * 2 - 1
            latents = self.vae.encode(frames).latent_dist.sample()
            latents *= VAE_TO_UNET_SCALING_FACTOR

        # Encode the audio
        with torch.no_grad():
            # TODO: This should be in the datamodule collate function
            audio_features = self.audio_featurizer(audio, sampling_rate=16_000, return_tensors="pt")

        # Resize the audio features to match the CLIP encoding
        if self.adaptor is not None:
            audio_features = self.adaptor(audio_features)

        # Add some noise
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (BS,),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.rand_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise
        noise_pred = self.unet(noisy_latents, timesteps, audio_features).sample

        # Compute the loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.unet.parameters(), lr=self.lr, betas=self.betas)
