import gradio as gr
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
import sa_handler

# init models
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False)
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    scheduler=scheduler
).to("cuda")
# Configure the pipeline for CPU offloading and VAE slicing#pipeline.enable_sequential_cpu_offload()
pipeline.enable_model_cpu_offload() 
pipeline.enable_vae_slicing()
# Initialize the style-aligned handler
handler = sa_handler.Handler(pipeline)
sa_args = sa_handler.StyleAlignedArgs(share_group_norm=False,
                                      share_layer_norm=False,
                                      share_attention=True,
                                      adain_queries=True,
                                      adain_keys=True,
                                      adain_values=False,
                                     )

handler.register(sa_args, )

# Define the function to generate style-aligned images
def style_aligned_sdxl(initial_prompt1, initial_prompt2, initial_prompt3, initial_prompt4,
                       initial_prompt5, style_prompt, seed):
    try:
        # Combine the style prompt with each initial prompt
        gen = None if seed is None else torch.manual_seed(int(seed))
        sets_of_prompts = [prompt + " in the style of " + style_prompt for prompt in [initial_prompt1, initial_prompt2, initial_prompt3, initial_prompt4, initial_prompt5] if prompt]
        # Generate images using the pipeline
        images = pipeline(sets_of_prompts, generator=gen).images
        return images
    except Exception as e:
        raise gr.Error(f"Error in generating images: {e}")

with gr.Blocks() as demo:
    gr.HTML('<h1 style="text-align: center;">StyleAligned SDXL</h1>')
    with gr.Group():
      with gr.Column():
        with gr.Accordion(label='Enter upto 5 different initial prompts', open=True):
          with gr.Row(variant='panel'):
            # Textboxes for initial prompts
            initial_prompt1 = gr.Textbox(label='Initial prompt 1', value='', show_label=False, container=False, placeholder='a toy train')
            initial_prompt2 = gr.Textbox(label='Initial prompt 2', value='', show_label=False, container=False, placeholder='a toy airplane')
            initial_prompt3 = gr.Textbox(label='Initial prompt 3', value='', show_label=False, container=False, placeholder='a toy bicycle')
            initial_prompt4 = gr.Textbox(label='Initial prompt 4', value='', show_label=False, container=False, placeholder='a toy car')
            initial_prompt5 = gr.Textbox(label='Initial prompt 5', value='', show_label=False, container=False, placeholder='a toy boat')
        with gr.Row():
          # Textbox for the style prompt
          style_prompt = gr.Textbox(label="Enter a style prompt", placeholder='macro photo, 3d game asset', scale=3)
          seed = gr.Number(value=1234, label="Seed", precision=0, step=1, scale=1,
                           info="Enter a seed of a previous run "
                                "or leave empty for a random generation.")
        # Button to generate images
        btn = gr.Button("Generate a set of Style-aligned SDXL images",)
    # Display the generated images
    output = gr.Gallery(label="Style aligned text-to-image on SDXL ", elem_id="gallery",columns=5, rows=1,
                        object_fit="contain", height="auto",)

    # Button click event
    btn.click(fn=style_aligned_sdxl, 
              inputs=[initial_prompt1, initial_prompt2, initial_prompt3, initial_prompt4, initial_prompt5,
                      style_prompt, seed],
              outputs=output, 
              api_name="style_aligned_sdxl")

    # Providing Example inputs for the demo
    gr.Examples(examples=[
                    ["a toy train", "a toy airplane", "a toy bicycle", "a toy car", "a toy boat", "macro photo. 3d game asset."],
                    ["a toy train", "a toy airplane", "a toy bicycle", "a toy car", "a toy boat", "BW logo. high contrast."],
                    ["a cat", "a dog", "a bear", "a man on a bicycle", "a girl working on laptop", "minimal origami."],
                    ["a firewoman", "a Gardner", "a scientist", "a policewoman", "a saxophone player", "made of claymation, stop motion animation."],
                    ["a firewoman", "a Gardner", "a scientist", "a policewoman", "a saxophone player", "sketch, character sheet."],
                    ],
            inputs=[initial_prompt1, initial_prompt2, initial_prompt3, initial_prompt4, initial_prompt5, style_prompt],
            outputs=[output],
            fn=style_aligned_sdxl)

# Launch the Gradio demo
demo.launch()