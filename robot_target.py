from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Type

from dotenv import load_dotenv
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
    """Adapter that exposes `LLMAgentForConversation` via echo_chamber's LLMClient."""

    def __init__(
        self,
        *,
        group_size: Optional[int] = None,
        group_source: str = "echo-attack",
        max_turns: int = 4,
        temperature: float = 0.0,
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        load_dotenv()
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

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Type

from dotenv import load_dotenv
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
    """Adapter that lets echo_chamber call this repo's LLM agent."""

    def __init__(
        self,
        *,
        group_size: Optional[int] = None,
        group_source: str = "echo-attack",
        max_turns: int = 4,
        temperature: float = 0.0,
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        load_dotenv()
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
        self.agent.start_new_session(
            group_size=self._group_size,
            group_source=self._group_source,
        )
        self._session_initialized = True

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

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Type

from dotenv import load_dotenv
from pydantic import BaseModel

from echo_chamber.llm_clients.base import (
    BaseLLMResponse,
    ChatMessage,
    LLMClient,
    RetryConfig,
)

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


def _chat_messages_to_prompt(
    messages: Sequence[ChatMessage],
) -> tuple[str, Optional[str]]:
    system_chunks: list[str] = []
    convo_chunks: list[str] = []
    for m in messages or []:
        role = str(getattr(m, "role", "") or "").strip().lower()
        content = str(getattr(m, "content", "") or "").strip()
        if not content:
            continue

        if role == "system":
            system_chunks.append(content)
            continue

        if role in ("assistant", "user"):
            convo_chunks.append(f"{role}: {content}")
            continue

        convo_chunks.append(content)

    system_prompt = "\n\n".join(system_chunks).strip() or None
    instructions = "\n".join(convo_chunks).strip()
    return instructions, system_prompt


class RobotTargetClient(LLMClient):
    """Adapter that lets `echo_chamber` drive this repo's `LLMAgentForConversation`.

    EchoChamberAttack expects an `LLMClient` with `complete()` / `complete_chat()`.
    This client forwards prompts to `LLMAgentForConversation.generate_response()`
    and converts the agent's multi-turn plan into a single text response.
    """

    def __init__(
        self,
        *,
        group_size: Optional[int] = None,
        group_source: str = "echo-attack",
        max_turns: int = 4,
        temperature: float = 0.0,
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        load_dotenv()
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
        self.agent.start_new_session(
            group_size=self._group_size,
            group_source=self._group_source,
        )
        self._session_initialized = True

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

'''
class RobotTargetClient(LLMClient):
    """Client for interacting with OpenAI's API.

    This class implements the LLMClient interface and provides methods to interact
    with OpenAI's API for both standard and Azure deployments. It supports single
    and batch completions with various configuration options.

    Attributes:
        client: An instance of either OpenAI or AzureOpenAI client for making API calls.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        # api_key: Optional[str] = None,
        # base_url: Optional[str] = None,
        condition_id: int,
        group_soze: Optional[int],
        retry_config: Optional[RetryConfig | dict[str, Any]] = None,
    ):
        """Initialize the OpenAI client.

        Args:
            model (str, optional): The model to use.
            temperature (float, optional): Sampling temperature. Defaults to 0.2.
            api_key (Optional[str], optional): API key to use. Defaults to None.
            base_url (Optional[str], optional): Base URL to use. Defaults to None.
            retry_config (Optional[RetryConfig | dict[str, Any]], optional): Retry configuration. Defaults to None.
        """
        super().__init__(temperature=temperature, retry_config=retry_config)
        self.model = model
        if self.retry_config:
            max_retries = self.retry_config.attempts
            if (
                self.retry_config.initial_delay
                or self.retry_config.max_delay
                or self.retry_config.exp_base
            ):
                LOGGER.warning(
                    "Initial delay, max delay, and exp base are not supported for OpenAI."
                )
        else:
            max_retries = DEFAULT_MAX_RETRIES

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

    async def complete(
        self,
        instructions: str,
        system_prompt: Optional[str] = None,
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        """Evaluate the response of the LLM using the ground truth and system prompt.

        Args:
            instructions (str): The prompt/instructions to send to the LLM.
            system_prompt (Optional[str], optional): System prompt to prepend. Defaults to None.
            response_schema (Type[BaseModel], optional): Response schema to use. Defaults to BaseLLMResponse.

        Returns:
            Dict[str, str]: The LLM's response parsed as a JSON dictionary.
        """
        messages = self._generate_messages(instructions, system_prompt)

        params = {
            "model": self.model,
            "messages": messages,
            "response_format": response_schema,
        }
        if not self._is_gpt_5():
            params["temperature"] = self.temperature

        # response = await self.client.beta.chat.completions.parse(**params)
        raw = self.agent.generate_response(instructions, max_turns = 10)

        content = response.choices[0].message.parsed
        if content is None:
            if (
                hasattr(response.choices[0].message, "refusal")
                and response.choices[0].message.refusal
            ):
                return {"response": "Refused to answer."}
            raise ValueError(
                "Received empty response from OpenAI - this may be due to a malicious or inappropriate prompt"
            )

        return content.model_dump()
'''
    """

    # def _is_gpt_5(self) -> bool:
    #     return "gpt-5" in self.model

    @staticmethod
    # def _generate_messages(
    #     instructions: str, system_prompt: Optional[str] = None
    # ) -> List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]:
    #     """Generate the message list for the LLM API call.

    #     Creates a list of message dictionaries in the format expected by OpenAI's API,
    #     optionally including a system prompt.

    #     Args:
    #         instructions (str): The user instructions/prompt to include.
    #         system_prompt (Optional[str], optional): System prompt to prepend. Defaults to None.

    #     Returns:
    #         List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]: message dictionaries with 'role' and 'content' keys.
    #     """
    #     messages: List[
    #         ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam
    #     ] = []
    #     if system_prompt:
    #         messages = [
    #             ChatCompletionSystemMessageParam(
    #                 role="system", content=system_prompt.strip()
    #             ),
    #             ChatCompletionUserMessageParam(
    #                 role="user", content=instructions.strip()
    #             ),
    #         ]
    #     else:
    #         messages = [
    #             ChatCompletionUserMessageParam(
    #                 role="user", content=instructions.strip()
    #             ),
    #         ]
    #     return messages

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        response_schema: Type[BaseModel] = BaseLLMResponse,
    ) -> Dict[str, Any]:
        """Complete a batch of chat messages using the OpenAI API.

        This method handles more complex chat completions with multiple messages and
        additional configuration options.

        Args:
            messages (Sequence[ChatMessage]): List of chat messages to send to the LLM.
            response_schema (Type[BaseModel], optional): Response schema to use. Defaults to BaseLLMResponse.

        Returns:
            Dict[str, Any]: The LLM's response parsed as a dictionary.
        """
        openai_messages = [
            ChatCompletionSystemMessageParam(
                role="system", content=str(m.content).strip()
            )
            if m.role == "system"
            else ChatCompletionUserMessageParam(
                role="user", content=str(m.content).strip()
            )
            for m in messages
        ]

        params = {
            "model": self.model,
            "messages": openai_messages,
            "response_format": response_schema,
        }
        if not self._is_gpt_5():
            params["temperature"] = self.temperature

        response = await self.client.beta.chat.completions.parse(**params)

        content = response.choices[0].message.parsed
        if content is None:
            if (
                hasattr(response.choices[0].message, "refusal")
                and response.choices[0].message.refusal
            ):
                raise ValueError(
                    f"Request refused due to content policy violation: {response.choices[0].message.refusal}"
                )
            raise ValueError(
                "Received empty response from OpenAI - this may be due to a malicious or inappropriate prompt"
            )

        return content.model_dump()
