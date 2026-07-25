from __future__ import annotations

from ..training.data_prep import build_instruction_examples


def run_training_pipeline(records: list[dict[str, str]]) -> list[dict[str, str]]:
    return build_instruction_examples(records)
