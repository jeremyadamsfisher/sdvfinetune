from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from torchvision import transforms as tfms
from transformers import Wav2Vec2Processor
from transformers import logging as tsmrs_logging

tsmrs_logging.set_verbosity_error()


to_tensor = tfms.ToTensor()


VAE_TO_UNET_SCALING_FACTOR = 0.18215


class StableDiffusionVideoFinetuneLightningModule(pl.LightningModule):
    def __init__(
        self,
        audio_featurizer: Wav2Vec2Processor,
        scheduler: Any,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        lr=None,
    ):
        super().__init__()
        self.audio_featurizer = audio_featurizer
        self.scheduler = scheduler
        self.unet = unet
        self.vae = vae
        self.lr = lr

    def training_step(self, batch, batch_idx):
        return self._step(batch)

    def _step(self, batch):
        # Get an arbitrary video frame, until SVD is released
        video, audio = batch
        BS, T, H, W, C = video.shape
        frames = batch[:, 0, :, :, :]
        assert frames.shape == (BS, H, W, C)

        # Encode the video frames
        with torch.no_grad():
            frames = frames * 2 - 1
            latents = (
                self.vae.encode(frames).latent_dist.sample()
                * VAE_TO_UNET_SCALING_FACTOR
            )

        # Encode the audio
        audio_features = self.audio_featurizer(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )

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
        loss = F.mse_loss(noise_pred, latents)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.unet, lr=self.lr)
