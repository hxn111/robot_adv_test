from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Type

from pydantic import BaseModel


class BaseLLMResponse(BaseModel):
    response: str


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 3


class LLMClient:
    def __init__(self, *, temperature: float = 0.0, retry_config: Optional[RetryConfig | dict[str, Any]] = None):
        self.temperature = float(temperature)
        if isinstance(retry_config, dict):
            self.retry_config = RetryConfig(**retry_config)
        else:
            self.retry_config = retry_config

    async def complete(
        self,
        instructions: str,
        system_prompt: Optional[str] = None,
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        raise NotImplementedError

