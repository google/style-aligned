import gradio as gr
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import sa_handler
import pipeline_calls



# Initialize models
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
# Configure pipeline for CPU offloading and VAE slicing
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()

# Initialize style-aligned handler
sa_args = sa_handler.StyleAlignedArgs(share_group_norm=False,
                                      share_layer_norm=False,
                                      share_attention=True,
                                      adain_queries=True,
                                      adain_keys=True,
                                      adain_values=False,
                                     )
handler = sa_handler.Handler(pipeline)
handler.register(sa_args, )


# Function to run ControlNet depth with StyleAligned
def style_aligned_controlnet(ref_style_prompt, depth_map, ref_image, img_generation_prompt, seed):
    try:
        if depth_map == True:
            image = load_image(ref_image)
            depth_image = pipeline_calls.get_depth_map(image, feature_processor, depth_estimator)
        else:
            depth_image = load_image(ref_image).resize((1024, 1024))
        controlnet_conditioning_scale = 0.8
        gen = None if seed is None else torch.manual_seed(int(seed))
        num_images_per_prompt = 3  # adjust according to VRAM size
        latents = torch.randn(1 + num_images_per_prompt, 4, 128, 128, generator=gen).to(pipeline.unet.dtype)

        images = pipeline_calls.controlnet_call(pipeline, [ref_style_prompt, img_generation_prompt],
                                                image=depth_image,
                                                num_inference_steps=50,
                                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                                num_images_per_prompt=num_images_per_prompt,
                                                latents=latents)
        return [images[0], depth_image] + images[1:], gr.Image(value=images[0], visible=True)
    except Exception as e:
        raise gr.Error(f"Error in generating images:{e}")

# Create a Gradio UI
with gr.Blocks() as demo:
    gr.HTML('<h1 style="text-align: center;">ControlNet with StyleAligned</h1>')
    with gr.Row():
      
      with gr.Column(variant='panel'):
        # Textbox for reference style prompt
        ref_style_prompt = gr.Textbox(
          label='Reference style prompt',
          info="Enter a Prompt to generate the reference image", placeholder='a poster in <style name> style'
        )
        with gr.Row(variant='panel'):
        # Checkbox for using controller depth-map
            depth_map = gr.Checkbox(label='Depth-map',)
            seed = gr.Number(value=1234, label="Seed", precision=0, step=1, scale=3,
                                    info="Enter a seed of a previous reference image "
                                         "or leave empty for a random generation.")
        # Image display for the generated reference style image
        ref_style_image = gr.Image(visible=False, label='Reference style image', scale=1)


      with gr.Column(variant='panel'): 
        # Image upload option for uploading a reference image for controlnet
        ref_image = gr.Image(label="Upload the reference image", 
                             type='filepath' )
        # Textbox for ControlNet prompt
        img_generation_prompt = gr.Textbox(
            label='Generation Prompt',
            info="Enter a Prompt to generate images using ControlNet and StyleAligned",
            )

    # Button to trigger image generation
    btn = gr.Button("Generate", size='sm')
    # Gallery to display generated images
    gallery = gr.Gallery(label="Style-Aligned ControlNet - Generated images", 
                           elem_id="gallery",
                           columns=5, 
                           rows=1, 
                           object_fit="contain", 
                           height="auto",
                          )
      
    btn.click(fn=style_aligned_controlnet, 
              inputs=[ref_style_prompt, depth_map, ref_image, img_generation_prompt, seed],
              outputs=[gallery, ref_style_image], 
              api_name="style_aligned_controlnet")


    # Example inputs for the Gradio interface
    gr.Examples(
      examples=[
        ['A couple sitting a wooden bench, in colorful clay animation, claymation style.', True,
         'example_image/train.png', 'A train in colorful clay animation, claymation style.',],
        ['A couple sitting a wooden bench, in colorful clay animation, claymation style.', False,
         'example_image/sun.png', 'Sun in colorful clay animation, claymation style.',],
        ['A poster in a papercut art style.', False,
         'example_image/A.png', 'Letter A in a papercut art style.', None],
        ['A bull in a low-poly, colorful origami style.', True, 'example_image/whale.png',
         'A whale in a low-poly, colorful origami style.', None],
        ['An image in ancient egyptian art style, hieroglyphics style.', True, 'example_image/camel.png',
         'A camel in a painterly, digital illustration style.',],
        ['An image in ancient egyptian art style, hieroglyphics style.', True, 'example_image/whale.png',
         'A whale in ancient egyptian art style, hieroglyphics style.',],
      ],
      inputs=[ref_style_prompt, depth_map, ref_image, img_generation_prompt,],
      outputs=[gallery, ref_style_image], 
      fn=style_aligned_controlnet,
      )

# Launch the Gradio demo   
demo.launch()
