import gradio as gr
import torch
from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler
import sa_handler
import pipeline_calls


# init models
model_ckpt = "stabilityai/stable-diffusion-2-base"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipeline = StableDiffusionPanoramaPipeline.from_pretrained(
     model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
).to("cuda")
# Configure the pipeline for CPU offloading and VAE slicing
pipeline.enable_model_cpu_offload()
pipeline.enable_vae_slicing()
sa_args = sa_handler.StyleAlignedArgs(share_group_norm=True,
                                      share_layer_norm=True,
                                      share_attention=True,
                                      adain_queries=True,
                                      adain_keys=True,
                                      adain_values=False,
                                     )
# Initialize the style-aligned handler
handler = sa_handler.Handler(pipeline)
handler.register(sa_args)


# Define the function to run MultiDiffusion with StyleAligned
def style_aligned_multidiff(ref_style_prompt, img_generation_prompt, seed):
    try:
        view_batch_size = 25  # adjust according to VRAM size
        gen = None if seed is None else torch.manual_seed(int(seed))
        reference_latent = torch.randn(1, 4, 64, 64, generator=gen)
        images = pipeline_calls.panorama_call(pipeline,
                                              [ref_style_prompt, img_generation_prompt],
                                              reference_latent=reference_latent,
                                              view_batch_size=view_batch_size)
    
        return images, gr.Image(value=images[0], visible=True)
    except Exception as e:
        raise gr.Error(f"Error in generating images:{e}")

# Create a Gradio UI
with gr.Blocks() as demo:
    gr.HTML('<h1 style="text-align: center;">MultiDiffusion with StyleAligned </h1>')
    with gr.Row():
      with gr.Column(variant='panel'):
        # Textbox for reference style prompt
        ref_style_prompt = gr.Textbox(
          label='Reference style prompt',
          info='Enter a Prompt to generate the reference image',
          placeholder='A poster in a papercut art style.'
        )
        seed = gr.Number(value=1234, label="Seed", precision=0, step=1,
                         info="Enter a seed of a previous reference image "
                              "or leave empty for a random generation.")
        # Image display for the reference style image
        ref_style_image = gr.Image(visible=False, label='Reference style image')


      with gr.Column(variant='panel'):
        # Textbox for prompt for MultiDiffusion panoramas
        img_generation_prompt = gr.Textbox(
          label='MultiDiffusion Prompt',
          info='Enter a Prompt to generate panoramic images using Style-aligned combined with MultiDiffusion',
          placeholder= 'A village in a papercut art style.'
          )

    # Button to trigger image generation
    btn = gr.Button('Style Aligned MultiDiffusion - Generate', size='sm')
    # Gallery to display generated style image and the panorama
    gallery = gr.Gallery(label='StyleAligned MultiDiffusion - generated images',
                           elem_id='gallery',
                           columns=5,
                           rows=1,
                           object_fit='contain',
                           height='auto',
                           allow_preview=True,
                           preview=True,
                          )
    # Button click event
    btn.click(fn=style_aligned_multidiff,
              inputs=[ref_style_prompt, img_generation_prompt, seed],
              outputs=[gallery, ref_style_image,],
              api_name='style_aligned_multidiffusion')

    # Example inputs for the Gradio demo
    gr.Examples(
      examples=[
        ['A poster in a papercut art style.', 'A village in a papercut art style.'],
        ['A poster in a papercut art style.', 'Futuristic cityscape in a papercut art style.'],
        ['A poster in a papercut art style.', 'A jungle in a papercut art style.'],
        ['A poster in a flat design style.', 'Giraffes in a flat design style.'],
        ['A poster in a flat design style.', 'Houses in a flat design style.'],
        ['A poster in a flat design style.', 'Mountains in a flat design style.'],
      ],
      inputs=[ref_style_prompt, img_generation_prompt],
      outputs=[gallery, ref_style_image],
      fn=style_aligned_multidiff,
      )

# Launch the Gradio demo
demo.launch()