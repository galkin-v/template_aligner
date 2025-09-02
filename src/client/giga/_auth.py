import asyncio
import uuid
from http import HTTPStatus

from aiohttp import ClientHandlerType, ClientRequest, ClientResponse, ClientSession, hdrs


class GigaChatOAuthTokenAuthorizationMiddleware:
    def __init__(self, url: str, access_key: str, scope: str):
        self.url = str(url)
        self.scope = scope
        self._access_key = access_key
        self._alock = asyncio.Lock()
        self._access_token: str | None = None

    async def _refresh(self, session: ClientSession):
        async with self._alock:
            headers = {
                hdrs.CONTENT_TYPE: "application/x-www-form-urlencoded",
                hdrs.ACCEPT: "application/json",
                hdrs.AUTHORIZATION: f"Basic {self._access_key}",
                "RqUID": str(uuid.uuid4()),
            }
            body = {"scope": self.scope}
            async with session.post(self.url, headers=headers, data=body, middlewares=()) as response:
                response.raise_for_status()
                data = await response.json()

            if "access_token" not in data:
                raise ValueError("Access token not received")

            self._access_token = data["access_token"]

    async def __call__(self, request: ClientRequest, handler: ClientHandlerType) -> ClientResponse:
        if str(request.url) == self.url:
            return await handler(request)

        if not self._access_token:
            await self._refresh(request.session)

        request.headers[hdrs.AUTHORIZATION] = f"Bearer {self._access_token}"
        response = await handler(request)

        if response.status == HTTPStatus.UNAUTHORIZED:
            await self._refresh(request.session)
            request.headers[hdrs.AUTHORIZATION] = f"Bearer {self._access_token}"
            response = await handler(request)

        return response
