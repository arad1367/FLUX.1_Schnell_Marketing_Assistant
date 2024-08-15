import gradio as gr
import numpy as np
import random
import spaces
import torch
from diffusers import DiffusionPipeline

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=dtype).to(device)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048

@spaces.GPU()
def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, num_inference_steps=4, progress=gr.Progress(track_tqdm=True)):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
            prompt = prompt,
            width = width,
            height = height,
            num_inference_steps = num_inference_steps,
            generator = generator,
            guidance_scale=0.0
    ).images[0]
    return image, seed

examples = [
    "Create a new logo for a tech startup",
    "Design an engaging Instagram post for a fashion brand",
    "Create a new character for a social media campaign",
    "Generate a marketing advertisement for a new product launch",
    "Design a social media banner for a charity event",
    "Create a new branding concept for a luxury hotel",
    "Design a promotional video thumbnail for a movie premiere",
    "Generate a marketing campaign for a sustainable lifestyle brand"
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 800px;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
}

#title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 20px;
}

#prompt {
    margin-bottom: 20px;
}

#result {
    margin-bottom: 20px;
}

#advanced-settings {
    margin-bottom: 20px;
}

#footer {
    text-align: center;
    font-size: 14px;
    color: #888;
}
"""

footer = """
<div id="footer">
    <a href="https://www.linkedin.com/in/pejman-ebrahimi-4a60151a7/" target="_blank">LinkedIn</a> |
    <a href="https://github.com/arad1367" target="_blank">GitHub</a> |
    <a href="https://arad1367.pythonanywhere.com/" target="_blank">Live demo of my PhD defense</a> |
    <a href="https://huggingface.co/black-forest-labs/FLUX.1-schnell" target="_blank">black-forest-labs/FLUX.1-schnell</a>
    <br>
    Made with 💖 by Pejman Ebrahimi
</div>
"""

with gr.Blocks(css=css, theme='gradio/soft') as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown("""
        # FLUX.1 Schnell Marketing Assistant

        This app uses the FLUX.1 Schnell model to generate high-quality images based on your prompt. Use it to create new logos, social media content, marketing advertisements, and more.
        """, elem_id="title")

        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
                elem_id="prompt"
            )

            run_button = gr.Button("Run", scale=0)

        result = gr.Image(label="Result", show_label=False, elem_id="result")

        with gr.Accordion("Advanced Settings", open=False, elem_id="advanced-settings"):
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                )

            with gr.Row():
                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=4,
                )

        gr.Examples(
            examples = examples,
            fn = infer,
            inputs = [prompt],
            outputs = [result, seed],
            cache_examples="lazy"
        )

        gr.HTML(footer)

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt, seed, randomize_seed, width, height, num_inference_steps],
        outputs = [result, seed]
    )

demo.launch()
