from __future__ import annotations

from ..agent.orchestrator import PricePredictionAgent
from ..models.schemas import PredictionRequest, PredictionResponse


async def run_inference_pipeline(
    agent: PricePredictionAgent,
    request: PredictionRequest,
) -> PredictionResponse:
    return await agent.predict(request)
