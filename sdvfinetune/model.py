from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from torchvision import transforms as tfms
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as tsmrs_logging

tsmrs_logging.set_verbosity_error()


to_tensor = tfms.ToTensor()


VAE_TO_UNET_SCALING_FACTOR = 0.18215


class StableDiffusion(pl.LightningModule):
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        scheduler: Any,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        lr=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.unet = unet
        self.vae = vae
        self.lr = lr

    def training_step(self, batch, batch_idx):
        return self._step(batch)

    def _step(self, batch):
        # Get an arbitrary video frame, until SVD is released
        audio, video = batch["audio"], batch["video"]
        BS, T, H, W, C = video.shape
        frames = batch[:, 0, :, :, :]  # BTHWC -> BHWC

        # Encode the video frames
        with torch.no_grad():
            frames = frames * 2 - 1
            latents = (
                self.vae.encode(frames).latent_dist.sample()
                * VAE_TO_UNET_SCALING_FACTOR
            )

        # Embed the prompt; to be replaced by the wav2vec embeddings
        prompt_embedding = self.embed_prompt("A person is speaking")

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
        noise_pred = self.unet(noisy_latents, timesteps, prompt_embedding).sample

        # Compute the loss
        loss = F.mse_loss(noise_pred, latents)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.unet, lr=self.lr)
