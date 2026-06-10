"""
llm_client.py — Unified LLM client for ai-job-search-enhanced.
Providers: Claude (primary) -> OpenAI (fallback) -> Ollama (offline/privacy).
Supports streaming, exponential backoff retry, and per-call cost tracking.
"""

import json
import logging
import os
import time
from typing import Generator, Optional

logger = logging.getLogger(__name__)

COST_TABLE = {
    "claude-opus-4-8":    {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-6":  {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5":   {"input": 0.00025, "output": 0.00125},
    "gpt-4o":             {"input": 0.005, "output": 0.015},
    "gpt-4o-mini":        {"input": 0.00015, "output": 0.0006},
    "llama3":             {"input": 0.0, "output": 0.0},
    "mistral":            {"input": 0.0, "output": 0.0},
}

PROVIDER_ORDER = ["claude", "openai", "ollama"]


class LLMClient:
    """
    Unified client for Claude / OpenAI / Ollama.
    Falls back down the provider chain on error.
    Logs cost per call to MemoryManager if available.
    """

    def __init__(self):
        self._anthropic: Optional[object] = None
        self._openai: Optional[object] = None
        self._memory: Optional[object] = None
        self._claude_model = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")
        self._openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        self._ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._privacy_mode = os.getenv("PRIVACY_MODE", "false").lower() == "true"

    @property
    def memory(self):
        if self._memory is None:
            try:
                from agent.memory.memory_manager import MemoryManager
                self._memory = MemoryManager()
            except Exception:
                pass
        return self._memory

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        use_case: str = "general",
    ) -> str:
        if self._privacy_mode:
            return self._ollama_complete(system, user, max_tokens)

        for provider in PROVIDER_ORDER:
            try:
                if provider == "claude":
                    result = self._claude_complete(system, user, max_tokens, use_case)
                elif provider == "openai":
                    result = self._openai_complete(system, user, max_tokens, use_case)
                else:
                    result = self._ollama_complete(system, user, max_tokens)
                return result
            except Exception as exc:
                logger.warning("Provider %s failed: %s; trying next", provider, exc)

        return self._emergency_fallback(system, user)

    def stream(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        if self._privacy_mode:
            yield from self._ollama_stream(system, user, max_tokens)
            return
        try:
            yield from self._claude_stream(system, user, max_tokens)
        except Exception:
            try:
                yield from self._openai_stream(system, user, max_tokens)
            except Exception:
                yield from self._ollama_stream(system, user, max_tokens)

    def _claude_complete(
        self, system: str, user: str, max_tokens: int, use_case: str
    ) -> str:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        if self._anthropic is None:
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = self._call_with_retry(
            lambda: self._anthropic.messages.create(
                model=self._claude_model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ),
            provider="claude",
        )

        text = response.content[0].text
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
        self._log_cost("claude", self._claude_model, use_case, in_tok, out_tok)
        return text

    def _claude_stream(self, system: str, user: str, max_tokens: int) -> Generator[str, None, None]:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        if self._anthropic is None:
            import anthropic
            self._anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        with self._anthropic.messages.stream(
            model=self._claude_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _openai_complete(
        self, system: str, user: str, max_tokens: int, use_case: str
    ) -> str:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")

        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = self._call_with_retry(
            lambda: self._openai.chat.completions.create(
                model=self._openai_model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            ),
            provider="openai",
        )

        text = response.choices[0].message.content
        in_tok = response.usage.prompt_tokens
        out_tok = response.usage.completion_tokens
        self._log_cost("openai", self._openai_model, use_case, in_tok, out_tok)
        return text

    def _openai_stream(self, system: str, user: str, max_tokens: int) -> Generator[str, None, None]:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")

        if self._openai is None:
            from openai import OpenAI
            self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        stream = self._openai.chat.completions.create(
            model=self._openai_model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _ollama_complete(self, system: str, user: str, max_tokens: int) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self._ollama_model,
            "prompt": f"System: {system}\n\nUser: {user}",
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self._ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        return data.get("response", "")

    def _ollama_stream(self, system: str, user: str, max_tokens: int) -> Generator[str, None, None]:
        import urllib.request
        payload = json.dumps({
            "model": self._ollama_model,
            "prompt": f"System: {system}\n\nUser: {user}",
            "stream": True,
            "options": {"num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self._ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                if line:
                    chunk = json.loads(line.decode())
                    text = chunk.get("response", "")
                    if text:
                        yield text
                    if chunk.get("done"):
                        break

    def _call_with_retry(self, fn, provider: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as exc:
                err = str(exc).lower()
                if "rate_limit" in err or "429" in err:
                    wait = 2 ** attempt
                    logger.info("%s rate limited; retrying in %ds", provider, wait)
                    time.sleep(wait)
                elif attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    raise

    def _log_cost(
        self, provider: str, model: str, use_case: str, in_tok: int, out_tok: int
    ):
        rates = COST_TABLE.get(model, {"input": 0, "output": 0})
        cost = (in_tok * rates["input"] + out_tok * rates["output"]) / 1000
        if self.memory:
            try:
                self.memory.log_llm_cost(provider, model, use_case, in_tok, out_tok, cost)
            except Exception:
                pass

    def _emergency_fallback(self, system: str, user: str) -> str:
        logger.error("All LLM providers failed; returning empty fallback")
        return '{"error": "All LLM providers unavailable. Check API keys and network connectivity."}'
