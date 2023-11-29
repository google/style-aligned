# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations
import torch
import numpy as np
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image


T = torch.Tensor
TN = T | None


def get_depth_map(image: Image, feature_processor: DPTImageProcessor, depth_estimator: DPTForDepthEstimation) -> Image:
    image = feature_processor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def concat_zero_control(control_reisduel: T) -> T:
    b = control_reisduel.shape[0] // 2
    zerso_reisduel = torch.zeros_like(control_reisduel[0:1])
    return torch.cat((zerso_reisduel, control_reisduel[:b], zerso_reisduel, control_reisduel[b::]))


@torch.no_grad()
def controlnet_call(
    pipeline: StableDiffusionXLControlNetPipeline,
    prompt: str | list[str] = None,
    prompt_2: str | list[str] | None = None,
    image: PipelineImageInput = None,
    height: int | None = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str | list[str] | None = None,
    negative_prompt_2: str | list[str] | None = None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    latents: TN = None,
    prompt_embeds: TN = None,
    negative_prompt_embeds: TN = None,
    pooled_prompt_embeds: TN = None,
    negative_pooled_prompt_embeds: TN = None,
    cross_attention_kwargs: dict[str, Any] | None = None,
    controlnet_conditioning_scale: float | list[float] = 1.0,
    control_guidance_start: float | list[float] = 0.0,
    control_guidance_end: float | list[float] = 1.0,
    original_size: tuple[int, int] = None,
    crops_coords_top_left: tuple[int, int] = (0, 0),
    target_size: tuple[int, int] | None = None,
    negative_original_size: tuple[int, int] | None = None,
    negative_crops_coords_top_left: tuple[int, int] = (0, 0),
    negative_target_size:tuple[int, int] | None = None,
    clip_skip: int | None = None,
) -> list[Image]:
    controlnet = pipeline.controlnet._orig_mod if is_compiled_module(pipeline.controlnet) else pipeline.controlnet

    # align format for control guidance
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
        )

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(
        prompt,
        prompt_2,
        image,
        1,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        controlnet_conditioning_scale,
        control_guidance_start,
        control_guidance_end,
    )

    pipeline._guidance_scale = guidance_scale

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipeline.encode_prompt(
        prompt,
        prompt_2,
        device,
        1,
        True,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
        clip_skip=clip_skip,
    )

    # 4. Prepare image
    if isinstance(controlnet, ControlNetModel):
        image = pipeline.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=1,
            num_images_per_prompt=1,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=True,
            guess_mode=False,
        )
        height, width = image.shape[-2:]
        image = torch.stack([image[0]] * num_images_per_prompt + [image[1]] * num_images_per_prompt)
    else:
        assert False
    # 5. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # 6. Prepare latent variables
    num_channels_latents = pipeline.unet.config.in_channels
    latents = pipeline.prepare_latents(
        1 + num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
        
    # 6.5 Optionally get Guidance Scale Embedding
    timestep_cond = None

    # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    # 7.1 Create tensor stating which controlnets to keep
    controlnet_keep = []
    for i in range(len(timesteps)):
        keeps = [
            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

    # 7.2 Prepare added time ids & embeddings
    if isinstance(image, list):
        original_size = original_size or image[0].shape[-2:]
    else:
        original_size = original_size or image.shape[-2:]
    target_size = target_size or (height, width)

    add_text_embeds = pooled_prompt_embeds
    if pipeline.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipeline.text_encoder_2.config.projection_dim

    add_time_ids = pipeline._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipeline._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    prompt_embeds = torch.stack([prompt_embeds[0]] + [prompt_embeds[1]] * num_images_per_prompt)
    negative_prompt_embeds = torch.stack([negative_prompt_embeds[0]] + [negative_prompt_embeds[1]] * num_images_per_prompt)
    negative_pooled_prompt_embeds = torch.stack([negative_pooled_prompt_embeds[0]] + [negative_pooled_prompt_embeds[1]] * num_images_per_prompt)
    add_text_embeds = torch.stack([add_text_embeds[0]] + [add_text_embeds[1]] * num_images_per_prompt)

    
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
    add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(1 + num_images_per_prompt, 1)
    batch_size = num_images_per_prompt + 1
    # 8. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipeline.scheduler.order
    is_unet_compiled = is_compiled_module(pipeline.unet)
    is_controlnet_compiled = is_compiled_module(pipeline.controlnet)
    is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    # controlnet_prompt_embeds = torch.cat((prompt_embeds[1:batch_size], prompt_embeds[batch_size+1:]))
    controlnet_prompt_embeds = torch.cat((prompt_embeds[1:batch_size], prompt_embeds[1:batch_size]))
    controlnet_added_cond_kwargs = {key: torch.cat((item[1:batch_size,], item[1:batch_size])) for key, item in added_cond_kwargs.items()}
    # controlnet_added_cond_kwargs = {key: torch.cat((item[1:batch_size,], item[batch_size+1:])) for key, item in added_cond_kwargs.items()}
    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # Relevant thread:
            # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
            if (is_unet_compiled and is_controlnet_compiled) and is_torch_higher_equal_2_1:
                torch._inductor.cudagraph_mark_step_begin()
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)           

            # controlnet(s) inference
            control_model_input = torch.cat((latent_model_input[1:batch_size,], latent_model_input[batch_size+1:]))

            if isinstance(controlnet_keep[i], list):
                cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]
            if cond_scale > 0:
                down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=False,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )
    
                mid_block_res_sample = concat_zero_control(mid_block_res_sample)
                down_block_res_samples =  [concat_zero_control(down_block_res_sample) for down_block_res_sample in down_block_res_samples]
            else:
                mid_block_res_sample = down_block_res_samples = None
            # predict the noise residual
            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
               
    # manually for max memory savings
    if pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast:
        pipeline.upcast_vae()
        latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)

    # make sure the VAE is in float32 mode, as it overflows in float16
    needs_upcasting = pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast

    if needs_upcasting:
        pipeline.upcast_vae()
        latents = latents.to(next(iter(pipeline.vae.post_quant_conv.parameters())).dtype)

    image = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]

    # cast back to fp16 if needed
    if needs_upcasting:
        pipeline.vae.to(dtype=torch.float16)
 
    if pipeline.watermark is not None:
        image = pipeline.watermark.apply_watermark(image)

    image = pipeline.image_processor.postprocess(image, output_type='pil')

    # Offload all models
    pipeline.maybe_free_model_hooks()


    return image

        