# Intelligent Price Prediction Agent

A production-oriented scaffold for an agentic price prediction platform built around LLMs, retrieval-augmented generation, and fine-tuning workflows.

## What is included

- FastAPI application skeleton for prediction and health endpoints
- Agent orchestration layer for retrieval, reasoning, and response generation
- RAG, LLM, training, evaluation, and data pipeline modules
- Cloud-ready Docker and deployment placeholders
- CI, testing, and documentation structure

## Project layout

- `config/`: local and production application settings
- `data/`: staged raw, preprocessed, feature, and prediction outputs
- `entrypoints/`: training and inference launch scripts
- `notebooks/`: exploratory analysis and experiment notebooks
- `src/price_agent/`: application code, agents, pipelines, and services
- `tests/`: unit and integration tests
- `docs/`: architecture and workflow notes
- `infra/`: deployment and infrastructure placeholders
- `.github/`: Copilot instructions and CI workflows

## Local development

1. Create a Python 3.11 environment.
2. Install the project in editable mode with dev dependencies: `pip install -e .[dev,training,llm,cloud]`.
3. Run the app with `python -m price_agent.main` or use `make run`.

## Next implementation steps

- Wire a real model provider and retrieval backend
- Connect data ingestion and feature pipelines
- Add evaluation, experiment tracking, and release automation
- Replace placeholder logic with trained forecasting and reasoning components
