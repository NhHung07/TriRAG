"""Backend Gemini (SDK `google-genai`)."""

from __future__ import annotations

import os

from .base import LLMBackend, LLMResponse, RetryableError


class GeminiBackend(LLMBackend):
    def __init__(self, model: str, api_key: str | None = None, thinking_budget: int | None = 0):
        from google import genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Thiếu GEMINI_API_KEY (đặt trong file .env).")
        self.model = model
        self.thinking_budget = thinking_budget
        self._client = genai.Client(api_key=api_key)

    def complete(self, prompt, *, system, temperature, max_tokens) -> LLMResponse:
        from google.genai import errors, types

        cfg = dict(temperature=temperature, max_output_tokens=max_tokens)
        if system:
            cfg["system_instruction"] = system
        if self.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=prompt, config=types.GenerateContentConfig(**cfg)
            )
        except errors.ClientError as e:  # 4xx
            if getattr(e, "code", None) in (408, 429):
                raise RetryableError(str(e)[:200]) from e
            raise
        except errors.ServerError as e:  # 5xx
            raise RetryableError(str(e)[:200]) from e
        except (TimeoutError, ConnectionError) as e:
            raise RetryableError(str(e)) from e

        usage = getattr(resp, "usage_metadata", None)
        text = resp.text or ""
        return LLMResponse(
            text=text,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
