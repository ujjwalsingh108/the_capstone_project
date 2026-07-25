from __future__ import annotations

from fastapi import FastAPI

from ..agent.orchestrator import PricePredictionAgent
from ..core.config import get_settings
from ..core.logging import configure_logging
from .routes.health import router as health_router
from .routes.predict import router as predict_router


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Agentic price prediction platform with RAG and fine-tuning support.",
    )
    app.state.settings = settings
    app.state.agent = PricePredictionAgent(settings=settings)

    app.include_router(health_router)
    app.include_router(predict_router, prefix="/v1")
    return app
