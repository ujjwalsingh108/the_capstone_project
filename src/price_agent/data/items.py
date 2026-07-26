from __future__ import annotations

from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"


class Item(BaseModel):
    """An Item is a data-point of a product with a price."""

    title: str
    category: str
    price: float
    full: Optional[str] = None
    weight: Optional[float] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    id: Optional[int] = None
    parent_asin: Optional[str] = None

    def make_prompt(self, text: str) -> None:
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{round(self.price)}.00"

    def test_prompt(self) -> str:
        if self.prompt is None:
            raise ValueError("Prompt has not been created. Call make_prompt first.")
        return self.prompt.split(PREFIX)[0] + PREFIX

    def __repr__(self) -> str:
        return f"<{self.title} = ${self.price}>"

    @staticmethod
    def push_to_hub(dataset_name: str, train: list[Item], val: list[Item], test: list[Item]) -> None:
        """Push Item lists to Hugging Face Hub."""
        DatasetDict(
            {
                "train": Dataset.from_list([item.model_dump() for item in train]),
                "validation": Dataset.from_list([item.model_dump() for item in val]),
                "test": Dataset.from_list([item.model_dump() for item in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Item], list[Item], list[Item]]:
        """Load from Hugging Face Hub and reconstruct Item objects."""
        ds: Any = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )