.PHONY: install lint test run

install:
	python -m pip install -U pip
	python -m pip install -e .[dev,training,llm,cloud]

lint:
	ruff check .

test:
	pytest

run:
	python -m price_agent.main
