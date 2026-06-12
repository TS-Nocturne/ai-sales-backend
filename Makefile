.PHONY: install test lint serve docker-build docker-up

install:
	poetry install

test:
	poetry run pytest -q

lint:
	poetry run black --check ai_sales tests
	poetry run isort --check-only ai_sales tests

serve:
	poetry run python -m ai_sales serve --host 0.0.0.0 --port 8000

docker-build:
	docker build -t ai-sales-brain .

docker-up:
	docker compose up --build
