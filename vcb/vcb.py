import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPProcessor

from .utils import is_torch2_available

if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .ip_adapter import ImageProjModel, IPAdapter
from .resampler import Resampler


class VCB(IPAdapter):
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        super().__init__(sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens)
    def generate_image_vcb(
        self,
        pil_image=None,
        clip_image_embeds=None,
        prompt=None,
        ref_images: List[Image.Image]=None,   # ← now a list of refs
        negative_prompt=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        common=True,
        theta=0.4,
        depth_scale=0.0,
        width=512,
        height=512,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        # source image
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )

        # reference images
        if not ref_images or len(ref_images) < 2:
            raise ValueError("You must pass at least two reference images in ref_images list.")
        ref_embeds = []
        uncond_ref_embeds = []
        for ref in ref_images:
            e, ue = self.get_image_embeds(pil_image=ref, clip_image_embeds=None)
            ref_embeds.append(e)
            uncond_ref_embeds.append(ue)

        # stack along new dim=0: shape = [n_refs, batch, seq_len, dim]
        stacked = torch.stack(ref_embeds, dim=0)
        uncond_stacked = torch.stack(uncond_ref_embeds, dim=0)

        # compute mean across refs: shape [batch, seq_len, dim]
        avg_ref = stacked.mean(dim=0)
        avg_uncond_ref = uncond_stacked.mean(dim=0)
        
        # compute absolute difference of each ref to the mean, then take the max
        # diff_from_mean: [batch, seq_len, dim]
        diff_from_mean, _ = (stacked - avg_ref).abs().max(dim=0)

        # build a similarity mask where locations with diff < theta are “common”
        embeds_similarity = torch.where(diff_from_mean < theta, 1.0, 0.0)

        # mix
        if common:
            mixed_embeds = (
                (1.0 - embeds_similarity) * image_prompt_embeds
                + embeds_similarity * avg_ref
            )
            mixed_uncond = (
                (1.0 - embeds_similarity) * uncond_image_prompt_embeds
                + embeds_similarity * avg_uncond_ref
            )
        else:
            mixed_embeds = (
                embeds_similarity * image_prompt_embeds
                + (1.0 - embeds_similarity) * ref_embeds[0]
            )
            mixed_uncond = (
                embeds_similarity * uncond_image_prompt_embeds
                + (1.0 - embeds_similarity) * uncond_ref_embeds[0]
            )

        # reshape for classifier-free guidance
        bs, seq_len, dim = mixed_embeds.shape
        mixed_embeds = mixed_embeds.repeat(1, num_samples, 1).view(bs * num_samples, seq_len, dim)
        mixed_uncond = mixed_uncond.repeat(1, num_samples, 1).view(bs * num_samples, seq_len, dim)

        # encode text prompts
        with torch.inference_mode():
            txt_embeds, txt_uncond = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([txt_embeds, mixed_embeds], dim=1)
            negative_prompt_embeds = torch.cat([txt_uncond, mixed_uncond], dim=1)

        # generate
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        outputs = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=depth_scale,
            width=width,
            height=height,
            **kwargs,
        )
        return outputs.images
