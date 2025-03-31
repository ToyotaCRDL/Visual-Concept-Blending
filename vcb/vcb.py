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
        ref_image1=None,
        ref_image2=None,
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

        # ref image1
        ref_image1_prompt_embeds, uncond_ref_image1_prompt_embeds = self.get_image_embeds(
            pil_image=ref_image1, clip_image_embeds=None
        )
        
        # ref image2
        ref_image2_prompt_embeds, uncond_ref_image2_prompt_embeds = self.get_image_embeds(
            pil_image=ref_image2, clip_image_embeds=None
        )

        # tensor operation between ref1 and ref2
        diff_embeds = torch.abs(ref_image1_prompt_embeds - ref_image2_prompt_embeds)
        embeds_similarity = torch.where(diff_embeds < theta, 1, 0)
        if common:
            # image mix
            mixed_image_prompt_embeds = \
                image_prompt_embeds * (1.0 - embeds_similarity) \
                + ref_image1_prompt_embeds * 0.5 * embeds_similarity \
                + ref_image2_prompt_embeds * 0.5 * embeds_similarity
            uncond_mixed_image_prompt_embeds = \
                uncond_image_prompt_embeds * (1.0 - embeds_similarity) \
                + uncond_ref_image1_prompt_embeds * 0.5 * embeds_similarity \
                + uncond_ref_image2_prompt_embeds * 0.5 * embeds_similarity
        else:            
            # image mix
            mixed_image_prompt_embeds = \
                image_prompt_embeds * embeds_similarity \
                + ref_image1_prompt_embeds * (1.0 - embeds_similarity)
            uncond_mixed_image_prompt_embeds = \
                uncond_image_prompt_embeds * embeds_similarity \
                + uncond_ref_image1_prompt_embeds * (1.0 - embeds_similarity)

        # generate images
        bs_embed, seq_len, _ = mixed_image_prompt_embeds.shape
        mixed_image_prompt_embeds = mixed_image_prompt_embeds.repeat(1, num_samples, 1)
        mixed_image_prompt_embeds = mixed_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_mixed_image_prompt_embeds = uncond_mixed_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_mixed_image_prompt_embeds = uncond_mixed_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, mixed_image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_mixed_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            controlnet_conditioning_scale=depth_scale,
            width=width,
            height=height,
            **kwargs,
        ).images

        return images
