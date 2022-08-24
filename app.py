import gradio as gr

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
from PIL import Image  
import re

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"
YOUR_TOKEN=""


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=YOUR_TOKEN)
pipe = pipe.to(device)


def infer(prompt, samples, steps, scale, seed):
        
    generator = torch.Generator(device=device).manual_seed(seed)
    with autocast("cuda"):
        images_list = pipe(
            [prompt] * samples,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        )
    images = []
    safe_image = Image.open(r"unsafe.png")
    for i, image in enumerate(images_list["sample"]):
        #if(images_list["nsfw_content_detected"][i]):
        #    images.append(safe_image)
        #else:
            images.append(image)
    return images


css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 1070px;
            margin: auto;
            padding-top: 2rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 24px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
"""

block = gr.Blocks(css=css)

examples = [
    [
        'A high tech solarpunk utopia in the Amazon rainforest',
        3,
        40,
        7.5,
        1024,
    ],
    [
        'A pikachu fine dining with a view to the Eiffel Tower',
        3,
        40,
        7,
        1024,
    ],
    [
        'A mecha robot in a favela in expressionist style',
        3,
        40,
        7,
        1024,
    ],
    [
        'an insect robot preparing a delicious meal',
        3,
        40,
        7,
        1024,
    ],
    [
        "A small cabin on top of a snowy mountain in the style of disney, arstation",
        3,
        40,
        7,
        1024,
    ],
]

with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 736px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Stable Diffusion
                </h1>
              </div>
              <p style="margin-bottom: 20px;">
                Testing gradio with stablediffusion
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=3,
                    placeholder="Enter your prompt",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[3], height="auto")

        advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")

        with gr.Row(elem_id="advanced-options"):
            samples = gr.Slider(label="Images", minimum=1, maximum=3, value=3, step=1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=40, step=1)
            scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
            )
            seed = gr.Slider(
                label="Random seed",
                minimum=0,
                maximum=2147483647,
                step=1,
                randomize=True,
            )

        ex = gr.Examples(examples=examples, fn=infer, inputs=[text, samples, steps, scale, seed], outputs=gallery, cache_examples=True)
        ex.dataset.headers = [""]

        
        text.submit(infer, inputs=[text, samples, steps, scale, seed], outputs=gallery)
        btn.click(infer, inputs=[text, samples, steps, scale, seed], outputs=gallery)
        advanced_button.click(
            None,
            [],
            text,
            _js="""
            () => {
                const options = document.querySelector("body > gradio-app").querySelector("#advanced-options");
                options.style.display = ["none", ""].includes(options.style.display) ? "flex" : "none";
            }""",
        )
        gr.HTML(
            """
                <div class="footer">
 <p>Alex is just testing this stuff, don't expect much 🤦‍️</p>
           """
        )

block.queue(max_size=40).launch(server_name="0.0.0.0", server_port=1234)
