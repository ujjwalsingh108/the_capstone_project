from __future__ import annotations

from fastapi import APIRouter, Request

from ...models.schemas import PredictionRequest, PredictionResponse

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    agent = request.app.state.agent
    return await agent.predict(payload)
