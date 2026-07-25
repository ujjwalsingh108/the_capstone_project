# Architecture

This project is organized around a few stable layers:

- `api/`: HTTP interface and request handling
- `agent/`: orchestration across retrieval, prompting, and response generation
- `rag/`: chunking, indexing, and document retrieval helpers
- `training/`: dataset preparation and fine-tuning jobs
- `evaluation/`: forecasting metrics and benchmark logic
- `data/`: ingestion and feature engineering utilities
- `infra/`: deployment and cloud infrastructure assets

The current codebase is a scaffold. Replace the placeholder agent logic with your trained forecasting model, vector store integration, experiment tracking, and deployment pipeline as the project matures.
