.PHONY: setup run flow test ui mlflow clean
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
run:
	python -m src.pipeline
flow:
	python dags/recommendation_flow.py
test:
	python -m pytest -q
mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
ui:
	prefect server start
clean:
	rm -rf data/raw/* data/interim/* data/processed/* data/features/* data/errors/* models/*.joblib models/*.json reports/* mlflow.db mlartifacts logs/*.log
