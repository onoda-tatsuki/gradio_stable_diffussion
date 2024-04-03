from typing import Any, Literal

from langchain.callbacks.manager import get_openai_callback
from langchain.globals import set_debug, set_verbose
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.pydantic_v1 import SecretStr
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.agent.stable_diffusion import SDGeneratorFactory
from src.config import get_config

config = get_config()

if config.LANGCHAIN_DEBUG_MODE == "ALL":
    set_debug(True)
elif config.LANGCHAIN_DEBUG_MODE == "VERBOSE":
    set_verbose(True)

SD_DESIGN_TEMPLATE = """
    You are an excellent designer. With the information given by the user, you can describe an illustration that would impress any illustrator or novelist.

    All you have to do is to use your imagination to describe the details of the illustration scene from the information given by the user.
    Specifically, you should describe the person's clothing, hairstyle, facial expression, age, gender, and other external characteristics; the person's facial expression, state of mind, and emotional landscape; the illustration's composition and object placement (what objects are where and their characteristics); the surrounding landscape and geography, weather and sky conditions, light levels, and the atmosphere conveyed to the person viewing the illustration.
    You will describe the scenery and the placement of the objects (what objects are located where and their characteristics), the surrounding landscape and geography, the weather and sky, the light and the atmosphere conveyed to the viewer. You are very good at describing a scene in a way that appeals to the user. Users are looking for illustrations with people in them. Another person will do the actual illustration, so you should concentrate only on describing the details.

    Use your imagination.
"""

SD_PROMPT_TEMPLATE = """
    You are a talented illustrator. From a description of a scene given by a designer, you can use Stable Diffusion (an image generation model) to generate an illustration that will amaze any designer or artist.

    To generate an illustration, a list of words called "prompt" is required. The prompt determine the quality of the illustration. The more variegated words you include, the more information you include, the better the illustration.
    Please output a brief, carefully selected output of about 20 words for the prompt. You do not have to present the words as they are given by the user, and you may supplement them with other words from your imagination if necessary.

    Prompt output must be in English, and output must be comma-separated word strings.
"""


class SDPromptGenerator:
    def __init__(self, model: str, temperature: float, verbose: bool = True) -> None:
        if config.OPENAI_API_KEY is None:
            raise ValueError("Missing OpenAI API Key")

        self.lim = ChatOpenAI(
            model=model,
            api_key=SecretStr(config.OPENAI_API_KEY),
            temperature=temperature,
            verbose=verbose,
        )

        sd_design_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SD_DESIGN_TEMPLATE),
                ("human", "{human_input}"),
            ]
        )

        sd_prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SD_PROMPT_TEMPLATE),
                ("human", "{ai_input}"),
            ]
        )

        output_parser = StrOutputParser()

        self.chain = (
            {
                "ai_input": sd_design_template | self.lim | output_parser,
                "human_input": RunnablePassthrough(),
            }
            | sd_prompt_template
            | self.lim
            | output_parser
        )

    def generate_sd_prompt(
        self,
        prompt: str,
        api_version: Literal["local", "v1", "core"],
        aspect_ratio: str = "1:1",
        art_style: str = "",
        width: float = 512,
        height: float = 512,
    ) -> tuple[str, OpenAICallbackHandler, Any]:

        with get_openai_callback() as callback:
            response = self.chain.invoke({"human_input": prompt})

        generator = SDGeneratorFactory.create(api_version)
        image = generator.generate_image(
            prompt=response,
            **{
                "aspect_ratio": aspect_ratio,
                "art_style": art_style,
                "width": width,
                "height": height,
            }
        )

        return (response, callback, image)
