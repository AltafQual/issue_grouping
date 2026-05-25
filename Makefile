# =============================================================================
# Makefile — one-stop shop for all docker workflows for this project
# =============================================================================
#
# Quick start:
#   make help              # list every target with a one-line description
#   make build             # build all images
#   make up                # start prod stack in background
#   make dev               # start dev stack with hot-reload (foreground)
#   make logs              # tail logs from all services
#   make down              # stop and remove the stack
#
# Layout:
#   - App stack (FastAPI + Streamlit + nginx):  docker-compose.yml
#   - Dev overlay (bind-mount + --reload):       docker-compose.dev.yml
#   - SPLADE on GPU host:                        docker-compose.splade.yml
#
# Requires: docker (>= 20.10), docker compose plugin (v2+).
# =============================================================================

# Use bash so `set -e` etc work consistently
SHELL := /bin/bash

# Compose file shortcuts
COMPOSE      := docker compose
COMPOSE_DEV  := docker compose -f docker-compose.yml -f docker-compose.dev.yml
COMPOSE_GPU  := docker compose -f docker-compose.splade.yml

# All recipes are phony — they invoke docker, never produce a file
.PHONY: help build up down restart logs ps shell-fastapi shell-streamlit \
        dev dev-build dev-down redeploy-fastapi redeploy-streamlit \
        splade-build splade-up splade-down splade-logs splade-shell \
        clean clean-volumes prune health smoke

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------

help: ## show this help
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# Production stack (FastAPI + Streamlit + nginx)
# -----------------------------------------------------------------------------

build: ## build all prod images (uses .dockerignore to skip .venv, models/, etc)
	$(COMPOSE) build

up: ## start prod stack in background (nginx waits for healthchecks)
	$(COMPOSE) up -d

down: ## stop and remove prod stack (volumes preserved)
	$(COMPOSE) down

restart: ## restart prod stack (no rebuild)
	$(COMPOSE) restart

logs: ## tail logs from all services (Ctrl+C to exit)
	$(COMPOSE) logs -f --tail=100

ps: ## show status of all containers
	$(COMPOSE) ps

# Re-deploy ONE service after `git pull` — the others stay up.
# Use this for low-downtime releases: only fastapi (or streamlit) restarts.
redeploy-fastapi: ## rebuild & restart only fastapi (others keep serving)
	$(COMPOSE) up -d --no-deps --build fastapi

redeploy-streamlit: ## rebuild & restart only streamlit
	$(COMPOSE) up -d --no-deps --build streamlit

# -----------------------------------------------------------------------------
# Development stack (bind-mount source + uvicorn --reload)
# -----------------------------------------------------------------------------
# Edits to .py files on the host are picked up in <1s by the in-container
# uvicorn / streamlit reloaders. New deps still need `make dev-build`.

dev: ## start dev stack with hot-reload (foreground; Ctrl+C stops it)
	$(COMPOSE_DEV) up

dev-build: ## rebuild dev stack (run after pyproject.toml / Dockerfile change)
	$(COMPOSE_DEV) up -d --build

dev-down: ## stop dev stack
	$(COMPOSE_DEV) down

# -----------------------------------------------------------------------------
# SPLADE encoder (GPU host)
# -----------------------------------------------------------------------------
# Run these on the GPU machine. Prereq (one-time):
#   sudo apt install -y nvidia-container-toolkit
#   sudo nvidia-ctk runtime configure --runtime=docker
#   sudo systemctl restart docker

splade-build: ## build SPLADE image
	$(COMPOSE_GPU) build

splade-up: ## start SPLADE on the GPU host
	$(COMPOSE_GPU) up -d

splade-down: ## stop SPLADE
	$(COMPOSE_GPU) down

splade-logs: ## tail SPLADE logs
	$(COMPOSE_GPU) logs -f --tail=100

splade-shell: ## open a shell inside the SPLADE container
	$(COMPOSE_GPU) exec splade /bin/bash

# -----------------------------------------------------------------------------
# Debugging / inspection
# -----------------------------------------------------------------------------

shell-fastapi: ## open a shell inside the running fastapi container
	$(COMPOSE) exec fastapi /bin/bash

shell-streamlit: ## open a shell inside the running streamlit container
	$(COMPOSE) exec streamlit /bin/bash

health: ## hit healthchecks via nginx (8080) and direct (8010, 8501)
	@echo "--- via nginx (port 8080) ---"
	@curl -fsS http://localhost:8080/api -o /dev/null -w "FastAPI swagger: %{http_code}\n" || true
	@curl -fsS http://localhost:8080/_stcore/health -w "Streamlit health: %{http_code}\n" || true
	@echo "--- direct ---"
	@curl -fsS http://localhost:8010/api -o /dev/null -w "FastAPI direct:  %{http_code}\n" || true
	@curl -fsS http://localhost:8501/_stcore/health -w "Streamlit direct: %{http_code}\n" || true

smoke: ## quick end-to-end smoke test — assumes stack is up
	@echo "Checking nginx → fastapi /api ..."
	@curl -fsS http://localhost:8080/api | head -c 200 ; echo
	@echo "Checking nginx → streamlit / ..."
	@curl -fsS http://localhost:8080/ | head -c 200 ; echo

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

clean: ## stop stack & remove containers (volumes kept = vector DB safe)
	$(COMPOSE) down --remove-orphans
	$(COMPOSE_DEV) down --remove-orphans

clean-volumes: ## stop stack & DELETE volumes — DESTROYS DB cache. Are you sure?
	@echo "About to delete docker volumes. Ctrl+C in 5s to abort..."
	@sleep 5
	$(COMPOSE) down -v --remove-orphans
	$(COMPOSE_GPU) down -v --remove-orphans

prune: ## remove dangling images, build cache, stopped containers (system-wide)
	docker system prune -f
	docker builder prune -f
