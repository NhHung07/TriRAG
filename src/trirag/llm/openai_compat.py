"""Backend tương thích OpenAI (OpenAI, Groq, OpenRouter, Ollama, vLLM...) — dùng cho judge."""

from __future__ import annotations

import os

from .base import LLMBackend, LLMResponse, RetryableError


class OpenAICompatBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "JUDGE_API_KEY",
        base_url_env: str = "JUDGE_BASE_URL",
    ):
        import openai

        self.model = model
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            base_url=base_url or os.environ.get(base_url_env) or None,
        )
        self._openai = openai

    def complete(self, prompt, *, system, temperature, max_tokens) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
        except (
            self._openai.RateLimitError,
            self._openai.APITimeoutError,
            self._openai.APIConnectionError,
            self._openai.InternalServerError,
        ) as e:
            raise RetryableError(str(e)[:200]) from e
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
