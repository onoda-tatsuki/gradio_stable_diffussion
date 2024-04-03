import base64
import io
import os
import uuid
from abc import ABCMeta, abstractmethod
from typing import Callable, Literal, Type, TypeVar

import numpy as np
import requests
from PIL import Image

from src.config import get_config

T = TypeVar("T", bound="StableDiffusionGenerator")

config = get_config()

api_type = Literal["local", "v1", "core"]

QUALITY_PROMPT = "best quality, masterpiece, extremely detailed"

NEGATIVE_PROMPT = "low quality, worst quality, out of focus, ugly, error, jpeg artifacts, lowers, blurry, bokeh, \
    bad anatomy, long_neck, long_body, longbody, deformed mutated disfigured, missing arms, extra_arms, mutated hands, \
    extra_legs, bad hands, poorly_drawn_hands, malformed_hands, missing_limb, floating_limbs, disconnected_limbs, extra_fingers, \
    bad fingers, liquid fingers, poorly drawn fingers, missing fingers, extra digit, fewer digits, ugly face, deformed eyes, \
    partial face, partial head, bad face, inaccurate limb, cropped text, signature, watermark, username, artist name, stamp, title, \
    subtitle, date, footer, header"


class StableDiffusionGenerator(metaclass=ABCMeta):
    def __init__(self) -> None:
        if config.STABILITY_API_KEY is None:
            raise ValueError("Missing Stability AI API Key")
        self.api_key = config.STABILITY_API_KEY
        self.quality_prompt = QUALITY_PROMPT
        self.negative_prompt = NEGATIVE_PROMPT

    def save_image(self, decoded_image: bytes, output_dir: str = "./output"):
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(f"./{output_dir}/v1_txt2img_{uuid.uuid4()}.png", "wb") as f:
            f.write(decoded_image)

    def decoded_image(self, encoded_data: bytes | str, *args, **kwargs):
        decoded_image = base64.b64decode(encoded_data)
        io_image = Image.open(io.BytesIO(decoded_image))
        image_np = np.array(io_image)

        return decoded_image, image_np

    @abstractmethod
    def generate_image(self, prompt: str, *args, **kwargs):
        pass


class SDGeneratorFactory:
    _generator_types: dict[api_type, Type[StableDiffusionGenerator]] = {}

    @classmethod
    def register(cls, generator_type: api_type) -> Callable:
        def decorator(cls_: Type[T]) -> Type[T]:
            if not issubclass(cls_, StableDiffusionGenerator):
                raise TypeError("this object is not StableDiffusionGenerator class")
            cls._generator_types[generator_type] = cls_
            return cls_

        return decorator

    @classmethod
    def create(
        cls, generator_type: api_type, *args, **kwargs
    ) -> StableDiffusionGenerator:
        generator_cls = cls._generator_types.get(generator_type)
        if generator_cls is None:
            raise ValueError(f"Generator type {generator_type} is not registered")
        return generator_cls(*args, **kwargs)


@SDGeneratorFactory.register("core")
class SDCoreGenerator(StableDiffusionGenerator):
    def __init__(self):
        super().__init__()

    def generate_image(self, prompt: str, art_style: str, *args, **kwargs):
        art_style = kwargs.get("art_style", "")
        aspect_ratio = kwargs.get("aspect_ratio", "1:1")

        response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/generate/core",
            headers={"authorization": self.api_key, "accept": "application/json"},
            files={"none": ""},
            data={
                "prompt": f"({self.quality_prompt}:1.3), " + f"({art_style}:1.4), "  + f"{prompt}",
                "negative_prompt": self.negative_prompt,
                "output_format": "webp",
                "aspect_ratio": aspect_ratio,
            },
        )

        data = response.json()

        if response.status_code == 200 and data.get("finish_reason") == "SUCCESS":
            decoded_image, image_np = self.decoded_image(data.get("image"))

            decoded_image = base64.b64decode(data.get("image"))
            io_image = Image.open(io.BytesIO(decoded_image))
            image_np = np.array(io_image)

            self.save_image(decoded_image)
            
        else:
            raise Exception(str(response.json()))

        return image_np


@SDGeneratorFactory.register("v1")
class SDV1Generator(StableDiffusionGenerator):
    def __init__(self, engine_id="stable-diffusion-v1-6") -> None:
        super().__init__()
        self.engine_id = engine_id
        self.api_host = config.STABILITY_API_HOST

    def generate_image(self, prompt: str, *args, **kwargs):
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        art_style = kwargs.get("art_style", "")

        response = requests.post(
            f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "text_prompts": [{"text": self.quality_prompt + prompt, "weight": 0.7}],
                "cfg_scale": 7,
                "height": height,
                "width": width,
                "samples": 1,
                "steps": 30,
                "sampler": "DDIM",
                "style_preset": art_style,
            },
        )

        data = response.json()

        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))

        output_images = []

        for image in data.get("artifacts"):
            decoded_image, image_np = self.decoded_image(image.get("base64"))

            self.save_image(decoded_image)
            output_images.append(image_np)

        return output_images


@SDGeneratorFactory.register("local")
class SDLocalGenerator(StableDiffusionGenerator):
    def __init__(self, url: str = "http://127.0.0.1:7860") -> None:
        super().__init__()
        self.url = url

    def generate_image(self, prompt: str, *args, **kwargs):
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)

        payload = {
            "prompt": self.quality_prompt + prompt,
            "negative_prompt": self.negative_prompt,
            "steps": 35,
            "width": width,
            "height": height,
        }
        response = requests.post(url=f"{self.url}/sdapi/v1/txt2img", json=payload)

        data = response.json()

        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))

        output_images = []

        for image in data["images"]:
            decoded_image, image_np = self.decoded_image(image)

            self.save_image(decoded_image)
            output_images.append(image_np)

        return output_images
