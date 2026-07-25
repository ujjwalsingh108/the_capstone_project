from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


class EchoLLMClient:
    async def generate(self, prompt: str) -> str:
        return prompt
