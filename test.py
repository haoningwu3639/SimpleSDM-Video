import os
import torch
from tqdm import tqdm
from typing import Optional
from accelerate import Accelerator
from accelerate.logging import get_logger

from model.unet_3d_condition import UNet3DConditionModel
from model.pipeline import TextToVideoSDPipeline
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from utils.util import get_time_string, export_to_video, export_to_gif

logger = get_logger(__name__)

def test(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    num_inference_steps: int = 40,
    num_frames: int = 16,
    guidance_scale: float = 9.0,
    num_sample_per_prompt: int = 1,
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

    sample_seeds = torch.randint(0, 100000, (num_sample_per_prompt,))
    sample_seeds = sorted(sample_seeds.numpy().tolist())
    
    for sample_seed in tqdm(sample_seeds):
        # sample_seed = random.randint(0, 100000)
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(sample_seed)
        shape = (1, 4, num_frames, 40, 72) # Init latents
        noise_latents = torch.randn(shape, generator=generator, device=accelerator.device, dtype=weight_dtype).to(accelerator.device)

        output = pipeline(
            prompt = prompt,
            height = 320,
            width = 576,
            num_frames = num_frames,
            latents = noise_latents,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
        )
        
        video_frames = output.frames
        export_to_gif(video_frames, os.path.join(logdir, 'output.gif'))
        # video_path = export_to_video(video_frames, os.path.join(logdir, 'output.mp4'))
        

if __name__ == "__main__":
    pretrained_model_path = "./ckpt/zeroscope_v2_576w/"
    logdir = "./test"
    num_inference_steps = 40
    num_frames = 24
    guidance_scale = 9.0
    num_sample_per_prompt = 1
    mixed_precision = "fp16" # "fp16",
    prompt = "A white cat is running in the rain."

    test(pretrained_model_path, logdir, prompt, num_inference_steps, num_frames, guidance_scale, num_sample_per_prompt, mixed_precision)

# CUDA_VISIBLE_DEVICES=5 accelerate launch test.py