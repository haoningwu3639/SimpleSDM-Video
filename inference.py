import os
import torch
import random
import argparse
from typing import Optional
from accelerate import Accelerator
from accelerate.logging import get_logger

from model.unet_3d_condition import UNet3DConditionModel
from model.pipeline import TextToVideoSDPipeline
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from utils.util import get_time_string, export_to_video, export_to_gif
from einops import rearrange

logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./inference", type=str)
    parser.add_argument('--ckpt', default='./ckpt/zeroscope_v2_576w/', type=str)
    parser.add_argument('--prompt', default="A white cat is running in the rain.", type=str)    
    parser.add_argument('--num_inference_steps', default=40, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--guidance_scale', default=9.0, type=float)
    return parser

def inference(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    num_inference_steps: int = 40,
    num_frames: int = 16,
    guidance_scale: float = 9.0,
    mixed_precision: Optional[str] = "no",   # "fp16" 
):
    
    time_string = get_time_string()
    logdir += f"_{time_string}"
    
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    pipeline = TextToVideoSDPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        )

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed" f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("SimpleSDM-Video")
    
    vae.eval()
    text_encoder.eval()
    unet.eval()

    # memory optimization
    pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
    pipeline.enable_vae_slicing()

    sample_seed = random.randint(0, 100000)
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(sample_seed)
    shape = (1, 4, num_frames, 40, 72) # Init latents
    noise_latents = torch.randn(shape, generator=generator, device=accelerator.device, dtype=weight_dtype).to(accelerator.device)

    video_frames = pipeline(
        prompt = prompt,
        height = 320,
        width = 576,
        num_frames = num_frames,
        latents = noise_latents,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
    ).frames
    
    # visualize noise and image
    b, c, f, h, w = noise_latents.shape
    noise_latents = noise_latents / vae.config.scaling_factor
    noise_latents = rearrange(noise_latents, "b c f h w -> (b f) c h w")
    noise = vae.decode(noise_latents).sample
    noise = noise.clamp(-1, 1)
    noise = (noise / 2 + 0.5).clamp(0, 1)
    noise = rearrange(noise, "(b f) c h w -> b f h w c", b=b)
    noise = noise.squeeze(0).detach().cpu().float().numpy()
    noise = [(frame * 255).astype("uint8") for frame in noise]
    export_to_gif(noise, os.path.join(logdir, "sample_noise.gif"))
    
    export_to_gif(video_frames, os.path.join(logdir, 'output.gif'))
    # video_path = export_to_video(video_frames, os.path.join(logdir, 'output.mp4'))
    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pretrained_model_path = args.ckpt
    logdir = args.logdir
    prompt = args.prompt
    num_inference_steps = args.num_inference_steps
    num_frames = args.num_frames
    guidance_scale = args.guidance_scale
    mixed_precision = "fp16" # "no"

    inference(pretrained_model_path, logdir, prompt, num_inference_steps, num_frames, guidance_scale, mixed_precision)

# CUDA_VISIBLE_DEVICES=5 accelerate launch test.py