RERANKER_MODEL_DIR := $(HOME)/.cache/repo-search/models
RERANKER_MODEL := jina-reranker-v2-base-multilingual-Q8_0.gguf
RERANKER_PORT := 8082

# Run the app + reranker sidecar together. Ctrl-C stops both.
dev:
	@if [ ! -f "$(RERANKER_MODEL_DIR)/$(RERANKER_MODEL)" ]; then \
		echo "⚠  Reranker model not found. Run 'make setup-reranker' first, or dev will start without reranking."; \
		echo "  Starting without cross-encoder reranker (LLM fallback will be used)...\n"; \
		cargo watch -x run; \
	else \
		echo "Starting reranker sidecar on port $(RERANKER_PORT)..."; \
		llama-server \
			-m "$(RERANKER_MODEL_DIR)/$(RERANKER_MODEL)" \
			--port $(RERANKER_PORT) \
			--rerank \
			--no-webui \
			-ngl 99 \
			2>&1 | sed 's/^/[reranker] /' & \
		RERANKER_PID=$$!; \
		sleep 1; \
		echo "Starting app server..."; \
		RERANKER_BASE_URL=http://127.0.0.1:$(RERANKER_PORT) \
		RERANKER_MODEL=jina-reranker-v2-base-multilingual \
			cargo watch -x run; \
		kill $$RERANKER_PID 2>/dev/null; \
	fi

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

# Run without the reranker sidecar.
dev-no-reranker:
	cargo watch -x run

build:
	cargo build --release
