from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Type

from pydantic import BaseModel

from echo_chamber.llm_clients.base import BaseLLMResponse, ChatMessage, LLMClient, RetryConfig

from llm_agent import LLMAgentForConversation


def _robot_plan_to_text(raw: Dict[str, Any]) -> str:
    turns = raw.get("planned_turns") or []
    parts: list[str] = []
    for turn in turns:
        text = str((turn or {}).get("speech_content") or "").strip()
        if text:
            parts.append(text)
    if parts:
        return "\n".join(parts)
    fallback = raw.get("note") or raw.get("response") or raw.get("text")
    return str(fallback or "").strip()


def _chat_messages_to_prompt(messages: Sequence[ChatMessage]) -> tuple[str, Optional[str]]:
    system_chunks: list[str] = []
    convo_chunks: list[str] = []
    for m in messages or []:
        role = str(getattr(m, "role", "") or "").strip().lower()
        content = str(getattr(m, "content", "") or "").strip()
        if not content:
            continue

        if role == "system":
            system_chunks.append(content)
        elif role in ("assistant", "user"):
            convo_chunks.append(f"{role}: {content}")
        else:
            convo_chunks.append(content)

    system_prompt = "\n\n".join(system_chunks).strip() or None
    instructions = "\n".join(convo_chunks).strip()
    return instructions, system_prompt


class RobotTargetClient(LLMClient):
    """Expose `LLMAgentForConversation` via echo_chamber's `LLMClient` interface."""

    def __init__(
        self,
        *,
        group_size: Optional[int] = None,
        group_source: str = "echo-attack",
        max_turns: int = 4,
        temperature: float = 0.0,
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        super().__init__(temperature=temperature, retry_config=retry_config)
        self.agent = LLMAgentForConversation()
        self.max_turns = max(1, int(max_turns))
        self._session_initialized = False
        self._group_size = group_size
        self._group_source = group_source

    def _ensure_session(self) -> None:
        if self._session_initialized:
            return
        self.agent.reset_convo_hist()
        self.agent.start_new_session(group_size=self._group_size, group_source=self._group_source)
        self._session_initialized = True

    def reset_session(self) -> None:
        """Reset conversation state so the next call starts a fresh session."""
        self._session_initialized = False

    async def complete(
        self,
        instructions: str,
        system_prompt: Optional[str] = None,
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        self._ensure_session()
        prompt = str(instructions or "").strip()
        if system_prompt:
            prompt = f"{str(system_prompt).strip()}\n\n{prompt}"

        raw = self.agent.generate_response(prompt, max_turns=self.max_turns)
        text = _robot_plan_to_text(raw)
        return response_schema(response=text).model_dump()

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        instructions, system_prompt = _chat_messages_to_prompt(messages)
        return await self.complete(
            instructions=instructions,
            system_prompt=system_prompt,
            response_schema=response_schema,
        )

