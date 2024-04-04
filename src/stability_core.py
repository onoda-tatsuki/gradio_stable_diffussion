import gradio as gr
from langchain.globals import set_verbose

from core import __set_base_path__
from src.agent.sd_prompt import SDPromptGenerator


def generate_image(
    prompt: str, models: str, temperature: int, aspect_ratio: str, art_style: str
):
    generator = SDPromptGenerator(models, temperature)
    response, token_info, images = generator.generate_sd_prompt(
        prompt=prompt,
        api_version="core",
        aspect_ratio=aspect_ratio,
        art_style=art_style,
    )

    return (response, images, token_info.total_tokens, token_info.total_cost)


prompt_input = gr.Textbox(label="Prompt", placeholder="Here is Prompt")

model_selector = gr.Radio(
    choices=["gpt-3.5-turbo", "gpt-4"],
    label="Models",
    value="gpt-3.5-turbo",
    type="value",
)

tmp_slider = gr.Slider(minimum=0, maximum=1, step=0.05, label="Temperature")
accept_dropdown = gr.Dropdown(
    choices=["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
    value="1:1",
    label="accept ratio",
)
styles_dropdown = gr.Dropdown(
    choices=[
        ("指定なし", ""),
        ("アニメ", "anime style"),
        ("リアル", "realistic"),
        ("ベクターアート", "vector art"),
        ("水彩", "watercolor"),
        ("キアロスクーロ", "chiaroscuro"),
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
    inputs=[prompt_input, model_selector, tmp_slider, accept_dropdown, styles_dropdown],
    outputs=[output_sd_prompt, output_image, total_tokens, total_cost],
)

if __name__ == "__main__":
    set_verbose(True)

    demo.queue()
    demo.launch(server_port=7861)
