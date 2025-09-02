import asyncio
import uuid
from typing import Dict, List, Optional

import aiohttp
import litellm
from litellm import APIError, Choices, CustomLLM, ModelResponse, RateLimitError
from litellm.utils import Message as LiteLLMMessage
from litellm.utils import Usage

from src.client.giga._auth import GigaChatOAuthTokenAuthorizationMiddleware
from src.client.giga.dto import GigaChatPayload, GigaChatResponse
from src.config.client import GIGA_SETTINGS


class CustomModelResponse(ModelResponse):
    @property
    def content(self) -> str:
        return self.choices[0].message.content


class GigaChatLLM(CustomLLM):
    def __init__(self):
        super().__init__()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=GIGA_SETTINGS.limit,
                force_close=GIGA_SETTINGS.force_close,
                ssl=GIGA_SETTINGS.verify_ssl_certs,
            )
            auth_middleware = GigaChatOAuthTokenAuthorizationMiddleware(
                url=GIGA_SETTINGS.auth_url,
                access_key=GIGA_SETTINGS.auth_token,
                scope=GIGA_SETTINGS.scope,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                middlewares=[auth_middleware],
                timeout=aiohttp.ClientTimeout(total=GIGA_SETTINGS.timeout),
            )
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure the session is closed gracefully on exit."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def acompletion(self, model: str, messages: List[Dict | LiteLLMMessage], **kwargs) -> CustomModelResponse:
        """The main async completion method called by LiteLLM."""
        session = await self._get_session()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-request-id": str(uuid.uuid4()),
        }

        if isinstance(messages[0], dict):
            messages = [LiteLLMMessage(role=msg["role"], content=str(msg["content"])) for msg in messages]

        # LiteLLM passes the full model name like "provider/model", we need to extract the actual model part.
        actual_model_name = model.split("/")[-1]

        optional_params = kwargs.get("optional_params", {})

        messages_dicts = [m.json() if isinstance(m, LiteLLMMessage) else m for m in messages]

        payload_data = {
            "model": actual_model_name,
            "messages": messages_dicts,
            "temperature": optional_params.get("temperature", 1.0),
            **{k: v for k, v in optional_params.items() if k in GigaChatPayload.model_fields},
        }
        payload = GigaChatPayload(**payload_data, max_tokens=512, profanity_check=False)

        try:
            async with session.post(
                url=GIGA_SETTINGS.url,
                data=payload.model_dump_json(),
                headers=headers,
            ) as response:
                if response.status == 429:  # Too Many Requests
                    raise RateLimitError(
                        message="GigaChat API rate limit exceeded.",
                        llm_provider=GIGA_SETTINGS.llm_provider,
                        model=model,  # Pass the original model alias
                    )
                response.raise_for_status()
                response_json = await response.json()
                giga_response = GigaChatResponse(**response_json)

            return self.convert_response(giga_response, original_model=model)

        except aiohttp.ClientResponseError as e:
            raise APIError(
                message=f"GigaChat API Error: {e.status} {e.message}",
                llm_provider=GIGA_SETTINGS.llm_provider,
                model=model,
                status_code=e.status,
            )
        except Exception as e:
            raise APIError(
                message=str(e),
                llm_provider=GIGA_SETTINGS.llm_provider,
                model=model,
                status_code=500,  # Default to 500 for unexpected errors
            )

    def completion(self, model: str, messages: List[Dict], **kwargs) -> CustomModelResponse:
        """Synchronous wrapper for acompletion."""
        return asyncio.run(self.acompletion(model=model, messages=messages, **kwargs))

    def convert_response(self, giga_response: GigaChatResponse, original_model: str) -> CustomModelResponse:
        """Helper to convert GigaChat's response to LiteLLM's standard CustomModelResponse."""
        model_response = CustomModelResponse(
            id=str(uuid.uuid4()),
            choices=[
                Choices(
                    finish_reason=choice.finish_reason,
                    index=choice.index,
                    message=LiteLLMMessage(content=choice.message.content, role=choice.message.role),
                )
                for choice in giga_response.choices
            ],
            # CRITICAL: Return the model alias that LiteLLM originally called
            model=original_model,
            usage=Usage(
                prompt_tokens=giga_response.usage.prompt_tokens,
                completion_tokens=giga_response.usage.completion_tokens,
                total_tokens=giga_response.usage.total_tokens,
            ),
        )
        model_response._hidden_params["original_response"] = giga_response.model_dump()
        return model_response


# --- 5. INSTANTIATE THE HANDLER ---
# Create a single, reusable instance of your LLM class that LiteLLM will use.
def init_giga_client():
    gigachat_handler = GigaChatLLM()
    litellm.custom_provider_map = [
        {
            "provider": GIGA_SETTINGS.llm_provider,
            "custom_handler": gigachat_handler,
        }
    ]
