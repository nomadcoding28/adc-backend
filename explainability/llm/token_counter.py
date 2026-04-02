"""
explainability/llm/token_counter.py
=====================================
Token budget management and prompt truncation utilities.

LLMs have context window limits.  This module:
  1. Counts tokens in prompt strings (using tiktoken for OpenAI models)
  2. Truncates prompt components to fit within a budget
  3. Warns when approaching context limits

Usage
-----
    counter = TokenCounter(model="gpt-4o-mini")

    # Count tokens in a string
    n = counter.count("This is a test sentence.")

    # Count tokens in a message list
    n = counter.count_messages([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ])

    # Truncate a string to fit within a budget
    truncated = counter.truncate_to_budget(long_text, max_tokens=500)

    # Check if a prompt fits within the model's context window
    fits = counter.fits_in_context(prompt, reserved_for_completion=512)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try importing tiktoken (OpenAI's tokeniser)
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    logger.debug(
        "tiktoken not installed — using character-based token estimation. "
        "Install with: pip install tiktoken"
    )

# ── Model context window limits ─────────────────────────────────────────────
_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-4o":              128_000,
    "gpt-4o-mini":         128_000,
    "gpt-4-turbo":         128_000,
    "gpt-4":                8_192,
    "gpt-3.5-turbo":       16_385,
    "claude-3-5-sonnet":   200_000,
    "claude-3-5-haiku":    200_000,
    "claude-3-opus":       200_000,
    "llama3":                8_192,
    "mistral":               8_192,
    "phi3":                  4_096,
}

# Approximate characters-per-token for models without tiktoken
_CHARS_PER_TOKEN = 4.0


class TokenCounter:
    """
    Token counting and budget management for LLM prompts.

    Parameters
    ----------
    model : str
        Model name — used to look up the context window size and
        select the correct tiktoken encoding.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._context_window = _CONTEXT_WINDOWS.get(model, 8_192)
        self._encoding: Optional[Any] = None

        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoding = tiktoken.encoding_for_model(model)
                logger.debug("tiktoken encoding loaded for model: %r", model)
            except KeyError:
                try:
                    self._encoding = tiktoken.get_encoding("cl100k_base")
                    logger.debug("tiktoken fallback: cl100k_base encoding.")
                except Exception:
                    self._encoding = None

    # ------------------------------------------------------------------ #
    # Counting
    # ------------------------------------------------------------------ #

    def count(self, text: str) -> int:
        """
        Count the number of tokens in a plain text string.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        int
            Approximate token count.
        """
        if not text:
            return 0

        if self._encoding is not None:
            return len(self._encoding.encode(text))

        # Fallback: character-based estimate
        return max(1, round(len(text) / _CHARS_PER_TOKEN))

    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens for a list of chat messages.

        Includes the per-message overhead (role tokens, separators) that
        OpenAI adds beyond the raw content.

        Parameters
        ----------
        messages : list[dict]
            Chat message list with ``"role"`` and ``"content"`` keys.

        Returns
        -------
        int
            Total token count including overhead.
        """
        total = 0
        for msg in messages:
            # Per-message overhead: ~4 tokens for role + separators
            total += 4
            total += self.count(msg.get("content", ""))
            total += self.count(msg.get("role", ""))
        total += 2   # Trailing priming tokens
        return total

    # ------------------------------------------------------------------ #
    # Budget management
    # ------------------------------------------------------------------ #

    @property
    def context_window(self) -> int:
        """Maximum context window size in tokens for the current model."""
        return self._context_window

    def available_tokens(
        self,
        used_tokens:                int,
        reserved_for_completion:    int = 512,
    ) -> int:
        """
        Compute how many tokens remain for additional prompt content.

        Parameters
        ----------
        used_tokens : int
            Tokens already used by system + fixed parts of the prompt.
        reserved_for_completion : int
            Tokens to reserve for the model's output.  Default 512.

        Returns
        -------
        int
            Remaining tokens available for variable prompt content.
        """
        return max(0, self._context_window - used_tokens - reserved_for_completion)

    def fits_in_context(
        self,
        text:                       str,
        already_used:               int = 0,
        reserved_for_completion:    int = 512,
    ) -> bool:
        """
        Return True if ``text`` fits within the remaining context budget.

        Parameters
        ----------
        text : str
            Text to check.
        already_used : int
            Tokens already committed to the prompt.
        reserved_for_completion : int
            Tokens reserved for the model output.

        Returns
        -------
        bool
        """
        available = self.available_tokens(already_used, reserved_for_completion)
        return self.count(text) <= available

    def truncate_to_budget(
        self,
        text:       str,
        max_tokens: int,
        suffix:     str = "...",
    ) -> str:
        """
        Truncate ``text`` to fit within ``max_tokens``.

        Uses a binary search approach with tiktoken for precision,
        or a character-based estimate as fallback.

        Parameters
        ----------
        text : str
            Text to truncate.
        max_tokens : int
            Maximum number of tokens allowed.
        suffix : str
            String appended to truncated text.  Default ``"..."``.

        Returns
        -------
        str
            Truncated (or original if it already fits) text.
        """
        if self.count(text) <= max_tokens:
            return text

        if self._encoding is not None:
            tokens = self._encoding.encode(text)
            suffix_tokens = len(self._encoding.encode(suffix))
            truncated_tokens = tokens[:max_tokens - suffix_tokens]
            return self._encoding.decode(truncated_tokens) + suffix

        # Fallback: character-based truncation
        max_chars = int(max_tokens * _CHARS_PER_TOKEN) - len(suffix)
        return text[:max_chars] + suffix

    def truncate_messages_to_budget(
        self,
        messages:                List[Dict[str, str]],
        reserved_for_completion: int = 512,
    ) -> List[Dict[str, str]]:
        """
        Truncate a message list so the total fits within the context window.

        Strategy:
          1. System message is never truncated (highest priority)
          2. The most recent user message is never truncated
          3. Earlier messages are truncated from oldest first

        Parameters
        ----------
        messages : list[dict]
            Chat messages.
        reserved_for_completion : int
            Tokens to reserve for output.

        Returns
        -------
        list[dict]
            Possibly truncated message list.
        """
        total_budget = self._context_window - reserved_for_completion
        current = self.count_messages(messages)

        if current <= total_budget:
            return messages

        # Work backwards — protect system and last user message
        result = list(messages)
        i = len(result) - 2   # Start from second-to-last (skip most recent)

        while self.count_messages(result) > total_budget and i > 0:
            msg = result[i]
            if msg.get("role") == "system":
                i -= 1
                continue

            # Halve the content of this message
            content = msg.get("content", "")
            result[i] = {**msg, "content": content[:len(content) // 2] + "..."}
            i -= 1

        return result

    # ------------------------------------------------------------------ #
    # Prompt assembly helpers
    # ------------------------------------------------------------------ #

    def build_prompt_within_budget(
        self,
        system:         str,
        query:          str,
        context_docs:   str,
        max_context_tokens: int = 2000,
        reserved_for_completion: int = 512,
    ) -> List[Dict[str, str]]:
        """
        Assemble a complete prompt within token budget.

        Truncates ``context_docs`` if needed to ensure the full prompt
        fits within the context window.

        Parameters
        ----------
        system : str
            System message content.
        query : str
            User query / task description.
        context_docs : str
            Retrieved RAG documents to include.
        max_context_tokens : int
            Maximum tokens to allocate to context docs.
        reserved_for_completion : int
            Tokens reserved for output.

        Returns
        -------
        list[dict]
            Chat message list ready to pass to LLMClient.chat().
        """
        # Calculate how many tokens the system + query use
        fixed_tokens = (
            self.count(system)
            + self.count(query)
            + 20   # overhead margin
        )
        remaining = self._context_window - fixed_tokens - reserved_for_completion

        # Cap context tokens at the lesser of max_context_tokens and remaining
        context_budget = min(max_context_tokens, max(0, remaining))
        truncated_context = self.truncate_to_budget(context_docs, context_budget)

        user_content = f"{query}\n\n## Context Documents\n{truncated_context}"

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_content},
        ]

    def __repr__(self) -> str:
        return (
            f"TokenCounter("
            f"model={self.model!r}, "
            f"context_window={self._context_window}, "
            f"tiktoken={'yes' if self._encoding else 'no'})"
        )