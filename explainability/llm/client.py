"""
explainability/llm/client.py
=============================
Unified LLM client that abstracts over OpenAI, Anthropic, and Ollama.

All explanation generation goes through this single interface so the
rest of the pipeline never needs to know which LLM provider is active.

Supported providers
-------------------
    openai      : GPT-4o-mini (default), GPT-4o, GPT-3.5-turbo
    anthropic   : claude-3-5-haiku, claude-3-5-sonnet
    ollama      : Any locally hosted model (llama3, mistral, phi3, etc.)

Usage
-----
    # From config dict
    client = LLMClient.from_config({
        "provider": "openai",
        "model":    "gpt-4o-mini",
        "api_key":  "sk-...",
    })

    # Simple completion
    response = client.complete("Explain CVE-2021-44228 in simple terms.")
    print(response.content)

    # Chat completion
    response = client.chat([
        {"role": "system", "content": "You are a cybersecurity expert."},
        {"role": "user",   "content": "What does Log4Shell exploit?"},
    ])
    print(response.content)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────
_DEFAULT_PROVIDER    = "openai"
_DEFAULT_MODEL       = "gpt-4o-mini"
_DEFAULT_MAX_TOKENS  = 1024
_DEFAULT_TEMPERATURE = 0.2     # Low temperature for factual explanations
_MAX_RETRIES         = 3
_RETRY_DELAY_S       = 2.0


@dataclass
class LLMResponse:
    """
    Structured response from an LLM completion call.

    Attributes
    ----------
    content : str
        The generated text content.
    provider : str
        Which LLM provider generated this response.
    model : str
        Specific model name used.
    prompt_tokens : int
        Number of tokens in the input prompt.
    completion_tokens : int
        Number of tokens in the generated response.
    total_tokens : int
        Sum of prompt + completion tokens.
    latency_s : float
        Wall-clock time for the API call in seconds.
    finish_reason : str
        Why generation stopped: ``"stop"``, ``"length"``, ``"error"``.
    """
    content:           str
    provider:          str
    model:             str
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    latency_s:         float = 0.0
    finish_reason:     str   = "stop"

    @property
    def is_truncated(self) -> bool:
        """True if generation was cut off by token limit."""
        return self.finish_reason == "length"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content":           self.content,
            "provider":          self.provider,
            "model":             self.model,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "latency_s":         round(self.latency_s, 3),
            "finish_reason":     self.finish_reason,
        }


class LLMClient:
    """
    Unified LLM client supporting OpenAI, Anthropic, and Ollama.

    Parameters
    ----------
    provider : str
        LLM provider: ``"openai"``, ``"anthropic"``, or ``"ollama"``.
    model : str
        Model name for the provider.
    api_key : str, optional
        API key.  Falls back to ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``
        environment variables.
    base_url : str, optional
        Base URL override.  Required for Ollama (default: ``http://localhost:11434``).
    max_tokens : int
        Maximum tokens in the generated response.  Default 1024.
    temperature : float
        Sampling temperature.  Default 0.2 (factual, low variance).
    timeout : float
        Request timeout in seconds.  Default 30.
    """

    def __init__(
        self,
        provider:    str  = _DEFAULT_PROVIDER,
        model:       str  = _DEFAULT_MODEL,
        api_key:     Optional[str] = None,
        base_url:    Optional[str] = None,
        max_tokens:  int   = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        timeout:     float = 30.0,
    ) -> None:
        self.provider    = provider.lower()
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout

        self._api_key  = api_key or self._env_api_key(provider)
        self._base_url = base_url or self._default_base_url(provider)

        # Lazy-initialise provider client
        self._client: Optional[Any] = None
        self._init_client()

        # Usage tracking
        self._total_prompt_tokens:     int = 0
        self._total_completion_tokens: int = 0
        self._total_calls:             int = 0

        logger.info(
            "LLMClient ready — provider=%r, model=%r", self.provider, self.model
        )

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LLMClient":
        """
        Build a client from a config dict.

        Expected keys: ``provider``, ``model``, ``api_key``,
        ``base_url``, ``max_tokens``, ``temperature``.
        """
        return cls(
            provider    = config.get("provider",    _DEFAULT_PROVIDER),
            model       = config.get("model",       _DEFAULT_MODEL),
            api_key     = config.get("api_key"),
            base_url    = config.get("base_url"),
            max_tokens  = config.get("max_tokens",  _DEFAULT_MAX_TOKENS),
            temperature = config.get("temperature", _DEFAULT_TEMPERATURE),
            timeout     = config.get("timeout",     30.0),
        )

    # ------------------------------------------------------------------ #
    # Public completion API
    # ------------------------------------------------------------------ #

    def complete(
        self,
        prompt:      str,
        system:      Optional[str] = None,
        max_tokens:  Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Simple single-turn text completion.

        Parameters
        ----------
        prompt : str
            The user prompt.
        system : str, optional
            System message (sets the assistant's persona/instructions).
        max_tokens : int, optional
            Override the default max_tokens for this call.
        temperature : float, optional
            Override the default temperature for this call.

        Returns
        -------
        LLMResponse
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
        )

    def chat(
        self,
        messages:    List[Dict[str, str]],
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """
        Multi-turn chat completion.

        Parameters
        ----------
        messages : list[dict]
            Conversation history.  Each dict has ``"role"`` and ``"content"``.
        max_tokens : int, optional
            Override for this call.
        temperature : float, optional
            Override for this call.

        Returns
        -------
        LLMResponse
        """
        effective_max    = max_tokens  or self.max_tokens
        effective_temp   = temperature or self.temperature

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                start = time.monotonic()

                if self.provider == "openai":
                    resp = self._call_openai(messages, effective_max, effective_temp)
                elif self.provider == "anthropic":
                    resp = self._call_anthropic(messages, effective_max, effective_temp)
                elif self.provider == "ollama":
                    resp = self._call_ollama(messages, effective_max, effective_temp)
                else:
                    resp = self._call_mock(messages)

                resp.latency_s = time.monotonic() - start

                # Track usage
                self._total_prompt_tokens     += resp.prompt_tokens
                self._total_completion_tokens += resp.completion_tokens
                self._total_calls             += 1

                return resp

            except Exception as exc:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt, _MAX_RETRIES, exc, _RETRY_DELAY_S * attempt,
                    )
                    time.sleep(_RETRY_DELAY_S * attempt)
                else:
                    logger.error(
                        "LLM call failed after %d attempts: %s", _MAX_RETRIES, exc
                    )
                    return LLMResponse(
                        content       = f"[LLM unavailable: {exc}]",
                        provider      = self.provider,
                        model         = self.model,
                        finish_reason = "error",
                    )

        # Should never reach here but mypy needs it
        return LLMResponse(
            content="[LLM error]", provider=self.provider, model=self.model
        )

    # ------------------------------------------------------------------ #
    # Provider-specific call implementations
    # ------------------------------------------------------------------ #

    def _call_openai(
        self,
        messages:    List[Dict[str, str]],
        max_tokens:  int,
        temperature: float,
    ) -> LLMResponse:
        """Call the OpenAI Chat Completions API."""
        resp = self._client.chat.completions.create(
            model       = self.model,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            timeout     = self.timeout,
        )
        choice = resp.choices[0]
        usage  = resp.usage or {}

        return LLMResponse(
            content           = choice.message.content or "",
            provider          = "openai",
            model             = self.model,
            prompt_tokens     = getattr(usage, "prompt_tokens",     0),
            completion_tokens = getattr(usage, "completion_tokens", 0),
            total_tokens      = getattr(usage, "total_tokens",      0),
            finish_reason     = choice.finish_reason or "stop",
        )

    def _call_anthropic(
        self,
        messages:    List[Dict[str, str]],
        max_tokens:  int,
        temperature: float,
    ) -> LLMResponse:
        """Call the Anthropic Messages API."""
        # Anthropic separates system from messages
        system_msg = next(
            (m["content"] for m in messages if m["role"] == "system"),
            None,
        )
        user_msgs = [m for m in messages if m["role"] != "system"]

        kwargs: Dict[str, Any] = {
            "model":      self.model,
            "messages":   user_msgs,
            "max_tokens": max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg

        resp = self._client.messages.create(**kwargs)

        content = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content += block.text

        usage = resp.usage or {}
        return LLMResponse(
            content           = content,
            provider          = "anthropic",
            model             = self.model,
            prompt_tokens     = getattr(usage, "input_tokens",  0),
            completion_tokens = getattr(usage, "output_tokens", 0),
            total_tokens      = getattr(usage, "input_tokens",  0) + getattr(usage, "output_tokens", 0),
            finish_reason     = resp.stop_reason or "stop",
        )

    def _call_ollama(
        self,
        messages:    List[Dict[str, str]],
        max_tokens:  int,
        temperature: float,
    ) -> LLMResponse:
        """Call a locally hosted Ollama model via its REST API."""
        import requests

        payload = {
            "model":    self.model,
            "messages": messages,
            "options":  {"temperature": temperature, "num_predict": max_tokens},
            "stream":   False,
        }
        resp = requests.post(
            f"{self._base_url}/api/chat",
            json    = payload,
            timeout = self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        return LLMResponse(
            content       = content,
            provider      = "ollama",
            model         = self.model,
            finish_reason = "stop" if data.get("done") else "length",
        )

    def _call_mock(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        Mock LLM for testing when no provider is configured.

        Returns a structured but clearly fake explanation.
        """
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "unknown query",
        )
        content = (
            f"[MOCK LLM — no provider configured]\n\n"
            f"Query: {last_user[:100]}\n\n"
            f"This is a placeholder explanation generated because no LLM "
            f"provider is configured.  Set OPENAI_API_KEY and provider=openai "
            f"in the config to get real explanations."
        )
        return LLMResponse(
            content       = content,
            provider      = "mock",
            model         = "mock",
            finish_reason = "stop",
        )

    # ------------------------------------------------------------------ #
    # Initialisation helpers
    # ------------------------------------------------------------------ #

    def _init_client(self) -> None:
        """Lazily initialise the provider SDK client."""
        try:
            if self.provider == "openai":
                from openai import OpenAI
                kwargs: Dict[str, Any] = {"api_key": self._api_key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = OpenAI(**kwargs)

            elif self.provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self._api_key)

            elif self.provider == "ollama":
                # Ollama uses plain HTTP — no SDK needed
                self._client = None

            else:
                logger.warning(
                    "Unknown provider %r — using mock LLM.", self.provider
                )
                self._client = None

        except ImportError as exc:
            logger.warning(
                "LLM SDK not installed for provider %r: %s — using mock.",
                self.provider, exc,
            )
            self.provider  = "mock"
            self._client   = None

    # ------------------------------------------------------------------ #
    # Stats & utilities
    # ------------------------------------------------------------------ #

    def get_usage(self) -> Dict[str, Any]:
        """Return cumulative token usage statistics."""
        return {
            "total_calls":             self._total_calls,
            "total_prompt_tokens":     self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_tokens":            self._total_prompt_tokens + self._total_completion_tokens,
        }

    @staticmethod
    def _env_api_key(provider: str) -> Optional[str]:
        key_map = {
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = key_map.get(provider.lower())
        return os.getenv(env_var) if env_var else None

    @staticmethod
    def _default_base_url(provider: str) -> Optional[str]:
        if provider.lower() == "ollama":
            return "http://localhost:11434"
        return None

    def __repr__(self) -> str:
        return (
            f"LLMClient("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"calls={self._total_calls})"
        )