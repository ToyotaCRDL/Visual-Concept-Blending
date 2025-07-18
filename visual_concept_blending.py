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
        self.common = common # transfer the common properties of the two reference images to the source image

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
        ref_images,
        seed=None, 
        theta=0.4, 
        num_samples=1, 
        depth_map=None, 
        depth_scale=1.0
    ):
        """
        Generate images by transferring features from multiple reference images to the source image.
        Uses a depth map as a ControlNet condition.

        Args:
            img (PIL.Image):               The source image.
            ref_images (List[PIL.Image]):   A list of reference images.
            seed (int, optional):           Random seed for reproducibility.
            theta (float, optional):        Threshold for extracting reference features.
            num_samples (int, optional):    Number of images to generate.
            depth_map (PIL.Image or None):  Depth map to be used by ControlNet.
            depth_scale (float, optional):  Strength of the ControlNet guidance.
        Returns:
            Tuple[List[PIL.Image], PIL.Image]:
                - A list of generated images.
                - The depth map used (auto-generated if none was provided).
        """

        depth_map = generate_depth_map_if_none(img, depth_map)
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
            ref_images=ref_images,
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
    parser.add_argument("-s", "--src_img_path", required=True, help="Path to the source image.")
    parser.add_argument("-r", "--ref_img_paths", nargs="+", required=True, help="Paths to two or more reference images (space separated).")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save result images.")

    # optional arguments
    parser.add_argument("-t", "--theta", type=float, default=0.3, help="Threshold for extracting reference features (default: 0.3).")
    parser.add_argument("-d", "--depth_scale", type=float, default=0.0, help="Strength of shape constraint (default: 0.0).")
    parser.add_argument("--seed", type=int, default=168, help="Random seed (default: 168).")
    parser.add_argument("--common", dest="common", action="store_true", help="Use 'common' concept transfer (default: True).")
    parser.add_argument("--distinctive", dest="common", action="store_false", help="Disable 'common' and use 'distictive' concept transfer.")
    parser.set_defaults(common=True)

    # parse arguments
    args = parser.parse_args()
    if len(args.ref_img_paths) < 2:
        parser.error("You must specify at least two reference images.")
    src_img_path = args.src_img_path
    ref_img_paths = args.ref_img_paths
    output_dir = args.output_dir

    SEED = args.seed
    common = args.common
    theta = args.theta
    depth_scale = args.depth_scale

    src_img_name = os.path.basename(src_img_path).split('.')[0]
    ref_img_names = [os.path.basename(p).split('.')[0] for p in ref_img_paths]
    ref_names_joined = "_".join(ref_img_names)
    common_or_distinct = 'common' if common else 'distinct'

    # load images    
    src_img = Image.open(src_img_path).convert('RGB').resize((512, 512))
    ref_imgs = [
        Image.open(p).convert('RGB').resize((512, 512))
        for p in ref_img_paths
    ]

    # generate
    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    ip = VisualConceptBlending(common=common)
    output_imgs, depth_map = ip.run(
        src_img,
        ref_imgs,
        seed=SEED,
        theta=theta,
        num_samples=1,
        depth_scale=depth_scale
    )

    # save
    out_img = output_imgs[0].resize((512, 512))
    out_img.save(
        f'{output_dir}/{cur_time}_{common_or_distinct}_src_{src_img_name}_refs_{ref_names_joined}_theta_{theta}.png'
    )
    depth_map.save(
        f'{output_dir}/{cur_time}_{common_or_distinct}_src_{src_img_name}_depth_map.png'
    )


if __name__ == '__main__':
    main()
