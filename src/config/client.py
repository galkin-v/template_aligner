from pydantic_settings import BaseSettings, SettingsConfigDict


class GigaSettings(BaseSettings):
    url: str = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    limit: int = 100
    force_close: bool = True
    timeout: int = 120
    profanity_check: bool = False
    temperature: float = 0.0000001

    auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    scope: str = "GIGACHAT_API_CORP"
    auth_token: str = ""
    verify_ssl_certs: bool = False

    llm_provider: str = "gigachat-provider"

    model_config = SettingsConfigDict(
        env_prefix="GIGA__",
        env_nested_max_split=1,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


GIGA_SETTINGS = GigaSettings()