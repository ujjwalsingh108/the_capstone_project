from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..models.schemas import SupportingSource


@dataclass(slots=True)
class RetrievedDocument:
    source_id: str
    title: str
    content: str
    score: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_supporting_source(self) -> SupportingSource:
        return SupportingSource(
            source_id=self.source_id,
            title=self.title,
            score=self.score,
            metadata=self.metadata,
        )


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Fetch the most relevant documents for a query."""


class InMemoryRetriever:
    def __init__(self, documents: list[RetrievedDocument] | None = None) -> None:
        self._documents = documents or []

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        if not self._documents:
            return []
        ranked = sorted(
            self._documents,
            key=lambda document: (document.score or 0.0, document.source_id),
            reverse=True,
        )
        return ranked[:top_k]
