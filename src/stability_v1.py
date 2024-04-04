import gradio as gr
from langchain.globals import set_verbose

from core import __set_base_path__
from src.agent.sd_prompt import SDPromptGenerator


def generate_image(
    prompt: str,
    models: str,
    temperature: int,
    width: float,
    height: float,
    art_style: str,
):
    generator = SDPromptGenerator(models, temperature)
    response, token_info, images = generator.generate_sd_prompt(
        prompt=prompt,
        api_version="v1",
        width=width,
        height=height,
        art_style=art_style,
    )

    return (response, images[0], token_info.total_tokens, token_info.total_cost)


prompt_input = gr.Textbox(label="Prompt", placeholder="Here is Prompt")

model_selector = gr.Radio(
    choices=["gpt-3.5-turbo", "gpt-4"],
    label="Models",
    value="gpt-3.5-turbo",
    type="value",
)

tmp_slider = gr.Slider(minimum=0, maximum=1, step=0.05, label="Temperature")
width_slider = gr.Slider(minimum=128, maximum=2048, step=1, label="Width", value=512)
height_slider = gr.Slider(minimum=128, maximum=2048, step=1, label="Height", value=512)
styles_dropdown = gr.Dropdown(
    choices=[
        ("指定なし", ""),
        ("3Dモデル", "3d-model"),
        ("アナログフィルム", "analog-film"),
        ("アニメ", "anime"),
        ("シネマティック", "cinematic"),
        ("コミックブック", "comic-book"),
        ("ファンタジーアート", "fantasy-art"),
        ("ラインアート", "line-art"),
    ],
    value="",
    label="Art Style",
)

output_sd_prompt = gr.TextArea(label="Generated Prompt")
output_image = gr.Image(label="Output Image")
total_tokens = gr.Textbox(label="Total tokens")
total_cost = gr.Textbox(label="Total Cost (chatGPT)")


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        prompt_input,
        model_selector,
        tmp_slider,
        width_slider,
        height_slider,
        styles_dropdown,
    ],
    outputs=[output_sd_prompt, output_image, total_tokens, total_cost],
)

if __name__ == "__main__":
    set_verbose(True)

    demo.queue()
    demo.launch(server_port=7861)
