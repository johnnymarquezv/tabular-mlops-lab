PYTHON ?= python3
VENV ?= $(HOME)/.venvs/tabular-mlops-lab
IMAGE ?= tabular-mlops-lab:latest
CONTAINER ?= tabular-mlops-lab-smoke
DOCKER_PORT ?= 18000
K8S_PORT ?= 18080

.PHONY: setup data validate train evaluate promote repro test api mlflow docker-build docker-smoke k8s-deploy k8s-rollout k8s-port-forward k8s-smoke k8s-clean clean

setup:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && python3 -m pip install --upgrade pip
	. $(VENV)/bin/activate && python3 -m pip install -e ".[dev]"

data:
	$(PYTHON) -m mlops_tabular.data

validate:
	$(PYTHON) -m mlops_tabular.validate

train:
	$(PYTHON) -m mlops_tabular.train

evaluate:
	$(PYTHON) -m mlops_tabular.evaluate

promote:
	$(PYTHON) -m mlops_tabular.promote_model

repro:
	$(PYTHON) -m dvc repro

test:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy src tests
	$(PYTHON) -m pytest

api:
	$(PYTHON) -m uvicorn mlops_tabular.api:app --host 0.0.0.0 --port 8000 --reload

mlflow:
	PYTHONPATH=. MLFLOW_ALLOW_FILE_STORE=true $(PYTHON) -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000

docker-build:
	docker build -t $(IMAGE) .

docker-smoke:
	docker rm -f $(CONTAINER) >/dev/null 2>&1 || true; \
	docker run -d --name $(CONTAINER) -p $(DOCKER_PORT):8000 $(IMAGE); \
	trap 'docker rm -f $(CONTAINER) >/dev/null 2>&1 || true' EXIT; \
	sleep 5; \
	curl -fsS http://127.0.0.1:$(DOCKER_PORT)/health

k8s-deploy:
	kubectl config use-context orbstack
	kubectl apply -k k8s/base

k8s-rollout:
	kubectl rollout status deployment/tabular-mlops-api --timeout=120s

k8s-port-forward:
	kubectl port-forward service/tabular-mlops-api 8000:80

k8s-smoke:
	kubectl rollout status deployment/tabular-mlops-api --timeout=120s
	kubectl port-forward service/tabular-mlops-api $(K8S_PORT):80 >/tmp/tabular-mlops-lab-port-forward.log 2>&1 & \
	pf_pid=$$!; \
	trap 'kill $$pf_pid >/dev/null 2>&1 || true' EXIT; \
	sleep 3; \
	curl -fsS http://127.0.0.1:$(K8S_PORT)/health

k8s-clean:
	kubectl delete -k k8s/base --ignore-not-found

clean:
	rm -rf data models reports mlruns
