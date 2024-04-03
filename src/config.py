import os
from functools import cache

from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"), verbose=True
)


class Config:
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    LANGCHAIN_DEBUG_MODE: str | None = os.getenv("LANGCHAIN_DEBUG_MODE")
    STABILITY_API_KEY: str | None = os.getenv("STABILITY_API_KEY")
    STABILITY_API_HOST: str = "https://api.stability.ai"

    # Noneを許容する変数のリスト
    ALLOW_NONE_VARIABLES: list[str] = [
        "LANGCHAIN_DEBUG_MODE",
    ]


@cache
def get_config() -> Config:
    config = Config()

    # Configクラスの全変数をループしてチェック
    for key, value in vars(config).items():
        if value is None and key not in Config.ALLOW_NONE_VARIABLES:
            raise ValueError(f"{key} is not set in the environment variables.")

    return config
