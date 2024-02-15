# SimpleSDM-Video
 A simple and flexible PyTorch implementation of Video Generation StableDiffusion (ZeroScope_v2) based on diffusers.

<div align="center">
   <img src="inference/output.gif">
</div>
<div align="center">
   A white cat is running in the rain.
</div>


## Prepartion
- You should download the checkpoints of ZeroScope_v2, from [ZeroScope_v2](https://huggingface.co/cerspense/zeroscope_v2_576w/tree/main), including scheduler, text_encoder, tokenizer, unet, and vae. Then put it in the ckpt folder.
- ZeroScope_v2 is a watermark-free [Modelscope-based](https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis) video model optimized for producing high-quality 16:9 compositions and a smooth video output.
- However, we found that the parameter_dict of the vae in ZeroScope_v2 may mismatch with the keys in [diffusers](https://huggingface.co/docs/diffusers), so we recommend you to download the **vae parameters** from [StableDiffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/vae), it will work well for our codebase.

## Requirements
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.12](https://pytorch.org/)
- xformers == 0.0.13
- diffusers == 0.13.1
- accelerate == 0.17.1
- transformers == 4.27.4

A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

## Dataset Preparation
- You need write a DataLoader suitable for your own Dataset, because we just provide a simple example to test the code.

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --multi_gpu train.py
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py --prompt "A white cat is running in the rain."
```

## Acknowledgements
Many thanks to the code bases from [diffusers](https://github.com/huggingface/diffusers) and [SimpleSDM](https://github.com/haoningwu3639/SimpleSDM), and pre-trained model parameters from [ZeroScope_v2](https://huggingface.co/cerspense/zeroscope_v2_576w/tree/main).