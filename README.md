# Style Aligned Image Generation via Shared Attention


### [Project Page](https://style-aligned-gen.github.io) &ensp; [Paper](https://style-aligned-gen.github.io/data/StyleAligned.pdf)


## Setup

This code was tested with Python 3.11, [Pytorch 2.1](https://pytorch.org/)  and [Diffusers 0.16](https://github.com/huggingface/diffusers).

## Examples
- See [**style_aligned_sdxl**][style_aligned] notebook for generating style aligned images using [SDXL](https://huggingface.co/docs/diffusers/using-diffusers/sdxl).

![alt text](doc/sa_example.jpg)


- See [**style_aligned_w_controlnet**][controlnet] notebook for generating style aligned and depth conditioned images using SDXL with [ControlNet-Depth](https://arxiv.org/abs/2302.05543).

![alt text](doc/cn_example.jpg)


-  [**style_aligned_w_multidiffusion**][multidiffusion] can be used for generating style aligned panoramas using [SD V2](https://huggingface.co/stabilityai/stable-diffusion-2) with [MultiDiffusion](https://multidiffusion.github.io/).

![alt text](doc/md_example.jpg)

## TODOs
- [ ] Adding demo.
- [x] StyleAligned from an input image.
- [ ] Multi-style with MultiDiffusion.
- [ ] StyleAligned with DreamBooth

## Disclaimer
This is not an officially supported Google product.

[style_aligned]: style_aligned_sdxl.ipynb
[controlnet]: style_aligned_w_controlnet.ipynb
[multidiffusion]: style_aligned_w_multidiffusion.ipynb
