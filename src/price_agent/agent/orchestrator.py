from __future__ import annotations

from ..core.config import Settings, get_settings
from ..llm.client import LLMClient
from ..llm.prompts import PREDICTION_PROMPT
from ..models.schemas import PredictionRequest, PredictionResponse
from ..rag.retrieval import InMemoryRetriever, RetrievedDocument, Retriever


class PricePredictionAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or InMemoryRetriever()
        self.llm_client = llm_client

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        query = request.context or request.asset_id
        documents = await self.retriever.retrieve(query=query, top_k=self.settings.rag_top_k)
        sources = [document.to_supporting_source() for document in documents]
        predicted_price = self._baseline_prediction(request)
        rationale = self._build_rationale(request, documents)

        if self.llm_client is not None:
            prompt = self._build_prompt(request, documents)
            rationale = await self.llm_client.generate(prompt)

        confidence = min(0.95, 0.4 + 0.1 * len(sources)) if sources else 0.35

        return PredictionResponse(
            asset_id=request.asset_id,
            predicted_price=predicted_price,
            confidence=confidence,
            rationale=rationale,
            sources=sources,
        )

    def _baseline_prediction(self, request: PredictionRequest) -> float:
        feature_signal = sum(request.features.values()) if request.features else 0.0
        horizon_factor = max(1.0, request.horizon_days / 7)
        return round(max(0.01, (100.0 + feature_signal) * horizon_factor), 2)

    def _build_rationale(
        self,
        request: PredictionRequest,
        documents: list[RetrievedDocument],
    ) -> str:
        support_count = len(documents)
        return (
            f"Baseline forecast for {request.asset_id} over {request.horizon_days} days. "
            f"Retrieved {support_count} support documents. Replace this scaffold with a trained "
            f"forecasting model and LLM-generated explanation."
        )

    def _build_prompt(
        self,
        request: PredictionRequest,
        documents: list[RetrievedDocument],
    ) -> str:
        sources = "\n".join(f"- {document.title}: {document.content}" for document in documents) or "- None"
        features = ", ".join(f"{name}={value}" for name, value in request.features.items()) or "none"
        notes = "\n".join(f"- {note}" for note in request.supporting_notes) or "- None"
        return (
            f"{PREDICTION_PROMPT}\n\n"
            f"Asset: {request.asset_id}\n"
            f"Horizon: {request.horizon_days} days\n"
            f"Features: {features}\n"
            f"Context: {request.context or 'None'}\n"
            f"Supporting notes:\n{notes}\n\n"
            f"Retrieved sources:\n{sources}"
        )
