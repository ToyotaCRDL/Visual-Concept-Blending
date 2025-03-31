import argparse
import os
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from diffusers import (
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    AutoencoderKL,
    ControlNetModel
)
from vcb import VCB

def generate_depth_map_if_none(img, depth_map=None):
    if depth_map is not None:
        return depth_map
    
    model_zoe_n = torch.hub.load("ZoeDepth", "ZoeD_N", source="local", pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_n.to(device)

    estimated_map = zoe.infer_pil(img, output_type="pil")
    depth_array = np.array(estimated_map)
    depth_array_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    inverted_map = ImageOps.invert(Image.fromarray(depth_array_norm).convert("RGB"))
    return inverted_map


class VisualConceptBlending():
    def __init__(self, common=True):
        self.common = common # transfer the common properties of the two reference images to the key image

        base_model_path = "runwayml/stable-diffusion-v1-5"
        vae_model_path = "stabilityai/sd-vae-ft-mse"
        image_encoder_path = "models/image_encoder"
        ip_ckpt = "models/ip-adapter_sd15.bin"
        device = "cuda"

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

        # load controlnet
        controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_path, 
            torch_dtype=torch.float16
        )

        # load SD pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

        # load ip-adapter
        self.ip_model = VCB(
            pipe, 
            image_encoder_path, 
            ip_ckpt, 
            device
        )
    
    def run(
        self, 
        img, 
        ref_image1, 
        ref_image2, 
        seed=None, 
        theta=0.4, 
        num_samples=1, 
        depth_map=None, 
        depth_scale=1.0
    ):
        """
        Generate images by transferring features from two reference images to the key image.
        Uses a depth map as a ControlNet condition.

        Args:
            img (PIL.Image):                        The source image.
            ref_image1 (PIL.Image):                 The first reference image.
            ref_image2 (PIL.Image):                 The second reference image.
            seed (int, optional):                   Random seed for reproducibility. Defaults to None.
            theta (float, optional):                Threshold for extracting reference features. Defaults to 0.5.
            num_samples (int, optional):            Number of images to generate. Defaults to 5.
            depth_map (PIL.Image or None, optional):Depth map to be used by ControlNet. 
                                                        If None, it will be automatically generated using ZoeDepth. Defaults to None.
            depth_scale (float, optional):          Strength of the ControlNet guidance (depth).
                                                    Defaults to 1.0.

        Returns:
            Tuple[List[PIL.Image], PIL.Image]:
                - A list of generated images.
                - The depth map used (auto-generated if none was provided).
        """

        # Generate depth map if none is provided
        depth_map = generate_depth_map_if_none(img, depth_map)

        # Determine the image size
        width, height = img.size
        print(f"width: {width}, height: {height}")

        # Generate images using IP-Adapter's mix function
        images = self.ip_model.generate_image_vcb(
            pil_image=img, 
            image=depth_map, 
            num_samples=num_samples, 
            guidance_scale=7.5,
            num_inference_steps=30, 
            seed=seed,
            ref_image1=ref_image1, 
            ref_image2=ref_image2, 
            scale=0.7, 
            common=self.common, 
            theta=theta,
            depth_scale=depth_scale,
            width=width,
            height=height
        )
        return images, depth_map

    
def main():
    parser = argparse.ArgumentParser(description="Run Image Concept Transfer from the command line.")

    # required arguments
    parser.add_argument("-k", "--key_img_path", required=True, help="Path to the key image.")
    parser.add_argument("-r1", "--ref_img1_path", required=True, help="Path to the first reference image.")
    parser.add_argument("-r2", "--ref_img2_path", required=True, help="Path to the second reference image.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save result images.")

    # optional arguments
    parser.add_argument("-t", "--theta", type=float, default=0.4, help="Threshold for extracting reference features (default: 0.4).")
    parser.add_argument("-d", "--depth_scale", type=float, default=0.0, help="Strength of shape constraint (default: 0.0).")
    parser.add_argument("--seed", type=int, default=168, help="Random seed (default: 168).")
    parser.add_argument("--common", dest="common", action="store_true", help="Use 'common' concept transfer (default: True).")
    parser.add_argument("--distinctive", dest="common", action="store_false", help="Disable 'common' and use 'distictive' concept transfer.")
    parser.set_defaults(common=True)

    # parse arguments
    args = parser.parse_args()
    key_img_path = args.key_img_path
    ref_img1_path = args.ref_img1_path
    ref_img2_path = args.ref_img2_path
    output_dir = args.output_dir

    SEED = args.seed
    common = args.common
    theta = args.theta
    depth_scale = args.depth_scale

    key_img_name = os.path.basename(key_img_path).split('.')[0]
    ref_img1_name = os.path.basename(ref_img1_path).split('.')[0]
    ref_img2_name = os.path.basename(ref_img2_path).split('.')[0]
    common_or_distinct = 'common' if common else 'distinct'

    # load images    
    key_img = Image.open(key_img_path).convert('RGB').resize((512, 512))
    ref_img1 = Image.open(ref_img1_path).convert('RGB').resize((512, 512))
    ref_img2 = Image.open(ref_img2_path).convert('RGB').resize((512, 512))

    # generation
    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    ip = VisualConceptBlending(common=common)
    output_img, depth_map = ip.run(
        key_img, ref_img1, ref_img2, seed=SEED, theta=theta, num_samples=1, depth_scale=depth_scale
    )
    output_img = output_img[0].resize((512, 512))
    output_img.save(
        f'{output_dir}/{cur_time}_depth_{common_or_distinct}_key_{key_img_name}_ref_{ref_img1_name}_{ref_img2_name}_theta_{theta}.png'.replace('\n', ' ')
    )
    depth_map.save(
        f'{output_dir}/{cur_time}_depth_{common_or_distinct}_key_{key_img_name}_depth_map.png'.replace('\n', ' ')
    )


if __name__ == '__main__':  
    main()
