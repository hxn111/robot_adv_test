from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Type

from openai import OpenAI
from pydantic import BaseModel

from .base import BaseLLMResponse, ChatMessage, LLMClient, RetryConfig


class OpenAiClient(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        super().__init__(temperature=temperature, retry_config=retry_config)
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        instructions: str,
        system_prompt: Optional[str] = None,
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instructions})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        return response_schema(response=text).model_dump()

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        openai_messages = [
            {"role": str(m.role), "content": str(m.content)} for m in messages
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        return response_schema(response=text).model_dump()

