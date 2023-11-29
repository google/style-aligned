# Style Aligned Image Generation via Shared Attention


### [Project Page](https://style-aligned-gen.github.io)


## Setup

This code was tested with Python 3.11, [Pytorch 2.1](https://pytorch.org/)  and [Diffusers 0.16](https://github.com/huggingface/diffusers).


## Examples
- For generating style aligned images using [SDXL](https://huggingface.co/docs/diffusers/using-diffusers/sdxl), see the notebook [**style_aligned_sdxl**][style_aligned].

- For generating style aligned images using SDXL with [ControlNet-Depth](https://arxiv.org/abs/2302.05543), see the notebook [**style_aligned_w_controlnet**][controlnet].

- For generating style aligned panoramas using [SD V2](https://huggingface.co/stabilityai/stable-diffusion-2) with [MultiDiffusion](https://multidiffusion.github.io/), see the notebook [**style_aligned_w_multidiffusion**][multidiffusion].


## Disclaimer

This is not an officially supported Google product.

[style_aligned]: style_aligned_sdxl.ipynb
[controlnet]: style_aligned_w_controlnet.ipynb
[multidiffusion]: style_aligned_w_multidiffusion.ipynb