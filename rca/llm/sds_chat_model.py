from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import requests
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT_SECONDS = 300.0
CONNECT_TIMEOUT_SECONDS = 10.0



def _content_text(content: Any) -> str:
    """Текст из OpenAI-compatible ``message.content``."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _message_content(content: Any) -> str:
    """Содержимое LangChain-сообщения в строку для REST payload."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _content_text(content)
        return text if text else json.dumps(content, ensure_ascii=False)
    return str(content)


def _to_gateway_messages(prompt: Any) -> list[dict[str, str]]:
    """ChatPromptValue/список сообщений/строка -> сообщения AI Gateway."""
    if hasattr(prompt, "to_messages"):
        source = prompt.to_messages()
    elif isinstance(prompt, list):
        source = prompt
    else:
        return [{"role": "user", "content": str(prompt)}]

    messages = []
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    for item in source:
        if isinstance(item, tuple) and len(item) == 2:
            role, content = item
        elif isinstance(item, dict):
            role, content = item.get("role", "user"), item.get("content", "")
        else:
            role = getattr(item, "type", "user")
            content = getattr(item, "content", item)
        messages.append({
            "role": role_map.get(str(role), str(role)),
            "content": _message_content(content),
        })
    return messages


def _extract_response(payload: Any, *, model_id: str) -> tuple[str, dict]:
    """Ответ AI Gateway -> текст и безопасная транспортная метаинформация."""
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"AI Gateway ({model_id}) вернул ответ без choices[0].message"
        ) from exc

    text = _content_text(message.get("content"))
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    metadata = {
        "model": model_id,
        "request_id": payload.get("id") if isinstance(payload, dict) else None,
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
    if not text.strip():
        reasoning = message.get("reasoning_content")
        raise RuntimeError(
            f"AI Gateway ({model_id}) вернул пустой message.content "
            f"(finish_reason={metadata['finish_reason']!r}, "
            f"reasoning_chars={len(reasoning) if isinstance(reasoning, str) else 0}, "
            f"request_id={metadata['request_id']!r})"
        )
    return text, metadata


def _extract_json_object(text: str) -> Any:
    """Извлекает один JSON-объект после необязательного блока рассуждений."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise RuntimeError(f"в ответе модели нет JSON-объекта: {text[:300]!r}")
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"ответ модели содержит невалидный JSON: {text[start:end + 1][:300]!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"структурированный ответ модели не является объектом: {payload!r}")
    return payload


def _structured_contract(schema: type) -> str:
    """Короткий instance-контракт, не провоцирующий модель повторить JSON Schema."""
    fields = getattr(schema, "model_fields", None)
    if fields is None:
        fields = getattr(schema, "__fields__", {})
    example = {name: None for name in fields}
    return (
        "Верни только один JSON-объект-ЭКЗЕМПЛЯР без markdown и описания "
        "схемы. Сохрани каждый ключ и замени null допустимой оценкой:\n"
        + json.dumps(example, ensure_ascii=False)
    )


def _validate_structured(payload: dict, schema: type):
    """Словарь -> экземпляр динамической Pydantic-схемы ассессора."""
    if "properties" in payload and not set(payload).intersection(
            getattr(schema, "model_fields", {})):
        raise RuntimeError("модель повторила JSON Schema вместо экземпляра ответа")
    try:
        if hasattr(schema, "model_validate"):
            return schema.model_validate(payload)
        return schema.parse_obj(payload)
    except Exception as exc:
        raise RuntimeError(
            f"ответ модели не соответствует схеме {schema.__name__}: {payload!r}"
        ) from exc



class SdsChatModel(Runnable[Any, AIMessage]):
    """LangChain-совместимая chat-модель отдельного inference SDS."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        temperature: float,
        top_p: float,
        timeout: float | str | None,
        verify_ssl_certs: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if not str(base_url or "").strip():
            raise ValueError(
                "Для модели AI Gateway не задан AI_GATEWAY_URL; "
                "маршрут доступен только в контуре SDS"
            )
        if not str(model_id or "").strip():
            raise ValueError("Имя модели AI Gateway не может быть пустым")
        self._url = str(base_url).rstrip("/") + "/chat/completions"
        self.model_id = str(model_id).strip()
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._timeout = float(timeout or DEFAULT_TIMEOUT_SECONDS)
        self._verify_ssl_certs = bool(verify_ssl_certs)
        self._max_tokens = int(max_tokens)

    def _complete(self, messages: list[dict[str, str]]) -> AIMessage:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": 1,
            "n": 1,
            "stream": False,
            "max_tokens": self._max_tokens,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_model_len": 131072,
        }
        started = time.monotonic()
        response = requests.post(
            self._url,
            headers={"Content-Type": "application/json"},
            json=payload,
            verify=self._verify_ssl_certs,
            timeout=(CONNECT_TIMEOUT_SECONDS, self._timeout),
        )
        # HTTPError сохраняет response.status_code: общий retry-контур
        # ассессора распознаёт 429 и ставит на паузу все параллельные вызовы.
        response.raise_for_status()
        try:
            response_payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"AI Gateway ({self.model_id}) вернул ответ не в JSON"
            ) from exc
        text, metadata = _extract_response(response_payload, model_id=self.model_id)
        logger.info(
            "AI Gateway judge: model=%s http=%s elapsed=%.2fs request_id=%r "
            "finish_reason=%r prompt_tokens=%r completion_tokens=%r",
            self.model_id, response.status_code, time.monotonic() - started,
            metadata["request_id"], metadata["finish_reason"],
            metadata["prompt_tokens"], metadata["completion_tokens"],
        )
        return AIMessage(content=text, response_metadata=metadata)

    def invoke(self, input: Any, config=None, **kwargs: Any) -> AIMessage:
        return self._complete(_to_gateway_messages(input))

    async def ainvoke(self, input: Any, config=None, **kwargs: Any) -> AIMessage:
        return await asyncio.to_thread(self.invoke, input, config, **kwargs)

    def with_structured_output(self, schema: type) -> Runnable:
        def invoke_structured(prompt: Any):
            messages = _to_gateway_messages(prompt)
            messages.append({"role": "user", "content": _structured_contract(schema)})
            answer = self._complete(messages)
            return _validate_structured(_extract_json_object(answer.content), schema)

        async def ainvoke_structured(prompt: Any):
            return await asyncio.to_thread(invoke_structured, prompt)

        return RunnableLambda(invoke_structured, afunc=ainvoke_structured)
