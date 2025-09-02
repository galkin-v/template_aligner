from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.client import GIGA_SETTINGS


class AppSettings(BaseSettings):
    provider_name: str = GIGA_SETTINGS.llm_provider
    model_name: str = "GigaChat-2-Pro"

    model_config = SettingsConfigDict(
        env_prefix="APP__",
        env_nested_max_split=1,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


APP_SETTINGS = AppSettings()
