from __future__ import annotations

from typing import Iterable


def build_instruction_examples(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for record in records:
        prompt = record.get("prompt", "")
        response = record.get("response", "")
        if prompt and response:
            examples.append({"prompt": prompt, "response": response})
    return examples
