RERANKER_MODEL_DIR := $(HOME)/.cache/repo-search/models
RERANKER_MODEL := jina-reranker-v2-base-multilingual-Q8_0.gguf
RERANKER_PORT := 8082
APP_PORT := 9000
PID_DIR := .pids

# ── Service management ──────────────────────────────────

# Start all services in the background.
dev:
	@mkdir -p $(PID_DIR)
	@if [ -f $(PID_DIR)/app.pid ] && kill -0 $$(cat $(PID_DIR)/app.pid) 2>/dev/null; then \
		echo "Services already running. Use 'make restart' or 'make stop' first."; \
		exit 1; \
	fi
	@# Start reranker if model is available
	@if [ -f "$(RERANKER_MODEL_DIR)/$(RERANKER_MODEL)" ]; then \
		echo "Starting reranker on port $(RERANKER_PORT)..."; \
		llama-server \
			-m "$(RERANKER_MODEL_DIR)/$(RERANKER_MODEL)" \
			--port $(RERANKER_PORT) \
			--rerank \
			--no-webui \
			-ngl 99 \
			>> logs/reranker.log 2>&1 & \
		echo $$! > $(PID_DIR)/reranker.pid; \
	else \
		echo "⚠  Reranker model not found — run 'make setup-reranker' to enable cross-encoder reranking."; \
	fi
	@# Start app server
	@mkdir -p logs
	@echo "Starting app server on port $(APP_PORT)..."
	@if [ -f "$(RERANKER_MODEL_DIR)/$(RERANKER_MODEL)" ]; then \
		RERANKER_BASE_URL=http://127.0.0.1:$(RERANKER_PORT) \
		RERANKER_MODEL=jina-reranker-v2-base-multilingual \
			cargo watch -x run >> logs/app.log 2>&1 & \
		echo $$! > $(PID_DIR)/app.pid; \
	else \
		cargo watch -x run >> logs/app.log 2>&1 & \
		echo $$! > $(PID_DIR)/app.pid; \
	fi
	@echo ""
	@echo "Services started:"
	@echo "  app        → http://127.0.0.1:$(APP_PORT)  (pid $$(cat $(PID_DIR)/app.pid))"
	@if [ -f $(PID_DIR)/reranker.pid ]; then \
		echo "  reranker   → http://127.0.0.1:$(RERANKER_PORT)  (pid $$(cat $(PID_DIR)/reranker.pid))"; \
	fi
	@echo ""
	@echo "Logs:  make logs"
	@echo "Stop:  make stop"

# Stop all services.
stop:
	@echo "Stopping services..."
	@if [ -f $(PID_DIR)/app.pid ]; then \
		PID=$$(cat $(PID_DIR)/app.pid); \
		kill $$PID 2>/dev/null && echo "  app        stopped (pid $$PID)" || echo "  app        not running"; \
		rm -f $(PID_DIR)/app.pid; \
	else \
		echo "  app        not running"; \
	fi
	@if [ -f $(PID_DIR)/reranker.pid ]; then \
		PID=$$(cat $(PID_DIR)/reranker.pid); \
		kill $$PID 2>/dev/null && echo "  reranker   stopped (pid $$PID)" || echo "  reranker   not running"; \
		rm -f $(PID_DIR)/reranker.pid; \
	else \
		echo "  reranker   not running"; \
	fi

# Restart all services.
restart: stop
	@sleep 1
	@$(MAKE) dev

# Show status of all services.
status:
	@echo "Services:"
	@if [ -f $(PID_DIR)/app.pid ] && kill -0 $$(cat $(PID_DIR)/app.pid) 2>/dev/null; then \
		echo "  app        ● running  (pid $$(cat $(PID_DIR)/app.pid))  http://127.0.0.1:$(APP_PORT)"; \
	else \
		echo "  app        ○ stopped"; \
		rm -f $(PID_DIR)/app.pid; \
	fi
	@if [ -f $(PID_DIR)/reranker.pid ] && kill -0 $$(cat $(PID_DIR)/reranker.pid) 2>/dev/null; then \
		echo "  reranker   ● running  (pid $$(cat $(PID_DIR)/reranker.pid))  http://127.0.0.1:$(RERANKER_PORT)"; \
	else \
		echo "  reranker   ○ stopped"; \
		rm -f $(PID_DIR)/reranker.pid; \
	fi

# Tail logs from all services.
logs:
	@mkdir -p logs
	@tail -f logs/app.log logs/reranker.log 2>/dev/null || echo "No logs yet. Run 'make dev' first."

# ── Setup ────────────────────────────────────────────────

# Download the reranker model and install llama.cpp if needed.
setup-reranker:
	@command -v llama-server >/dev/null 2>&1 || { echo "Installing llama.cpp via Homebrew..."; brew install llama.cpp; }
	@mkdir -p "$(RERANKER_MODEL_DIR)"
	@if [ ! -f "$(RERANKER_MODEL_DIR)/$(RERANKER_MODEL)" ]; then \
		echo "Downloading reranker model (~300 MB)..."; \
		curl -L -o "$(RERANKER_MODEL_DIR)/$(RERANKER_MODEL)" \
			"https://huggingface.co/gpustack/jina-reranker-v2-base-multilingual-GGUF/resolve/main/$(RERANKER_MODEL)"; \
	else \
		echo "Reranker model already downloaded."; \
	fi
	@echo "Done. Run 'make dev' to start with cross-encoder reranking."

# ── Build ────────────────────────────────────────────────

build:
	cargo build --release

clean:
	@$(MAKE) stop
	@rm -rf logs $(PID_DIR)
	@echo "Cleaned logs and pid files."

.PHONY: dev stop restart status logs setup-reranker build clean
