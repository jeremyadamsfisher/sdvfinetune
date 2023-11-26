from typing import Any

import pytorch_lightning as pl
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image
from torchvision import transforms as tfms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as tsmrs_logging

tsmrs_logging.set_verbosity_error()


to_tensor = tfms.ToTensor()


VAE_TO_UNET_SCALING_FACTOR = 0.18215


def compress(
    img: Image.Image,  # Input image
    vae: torch.nn.Module,  # VAE
):
    """Project pixels into latent space"""
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    img = to_tensor(img).unsqueeze(0).to(vae.device)
    img = img * 2 - 1  # Note scaling
    with torch.no_grad():
        latents = vae.encode(img)
    return VAE_TO_UNET_SCALING_FACTOR * latents.latent_dist.sample()


def decompress(
    latents: torch.Tensor,  # VAE latents
    vae: torch.nn.Module,  # VAE
    as_pil=True,  # Return a PIL image
    no_grad=True,  # Discard forward gradientss
):
    """Project latents into pixel space"""
    if no_grad:
        with torch.no_grad():
            img = vae.decode(latents / VAE_TO_UNET_SCALING_FACTOR).sample
    else:
        img = vae.decode(latents / VAE_TO_UNET_SCALING_FACTOR).sample
    img = (img / 2 + 0.5).clamp(0, 1)
    # color dimension goes last for matplotlib
    img = img.permute(0, 2, 3, 1)
    if as_pil:
        img = img.cpu().numpy().squeeze()
        img = (img * 255).round().astype("uint8")
        img = Image.fromarray(img)
    return img


class StableDiffusion(pl.LightningModule):
    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        scheduler: Any,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        lr=None,
        rng=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.unet = unet
        self.vae = vae
        if rng is None:
            rng = torch.manual_seed(42)
        self.rng = rng
        self.lr = lr

    def embed_prompt(self, prompt: str) -> torch.tensor:
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))
            text_embeddings = text_embeddings[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )
            uncond_embeddings = uncond_embeddings[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def pred_noise(self, prompt_embedding, l, t, guidance_scale):
        latent_model_input = torch.cat([l] * 2)
        # Scale the initial noise by the variance required by the scheduler
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        with torch.no_grad():
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=prompt_embedding
            ).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred

    def denoise(
        self, prompt_embedding, l, t, guidance_scale, i, return_noise_pred=False
    ):
        noise_pred = self.pred_noise(prompt_embedding, l, t, guidance_scale)
        l = self.scheduler.step(noise_pred, t, l).prev_sample
        if return_noise_pred:
            return l, noise_pred
        return l

    def init_latents(self):
        l = torch.randn((1, self.unet.config.in_channels, 64, 64), generator=self.rng)
        l = l.to(self.device)
        l *= self.scheduler.init_noise_sigma
        return l

    def init_schedule(self, n_inference_steps):
        self.scheduler.set_timesteps(n_inference_steps)
        # workaround for ARM Macs where float64's are not supported
        self.scheduler.timesteps = self.scheduler.timesteps.to(torch.float32)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
        return self.scheduler

    def __call__(self, prompt, guidance_scale=7.5, n_inference_steps=30, as_pil=False):
        prompt_embedding = self.embed_prompt(prompt)
        l = self.init_latents()
        self.init_schedule(n_inference_steps)
        # Note that the time steps aren't neccesarily 1, 2, 3, etc
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=n_inference_steps):
            # workaround for ARM Macs where float64's are not supported
            t = t.to(torch.float32).to(self.device)
            l = self.denoise(prompt_embedding, l, t, guidance_scale, i)
        return decompress(l, self.vae, as_pil=as_pil)

    def configure_optimizers(self):
        return torch.optim.SGD(self.unet, lr=self.lr)
