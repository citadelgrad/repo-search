# repo-search

A Rust web app for cloning git repositories and searching them with a hybrid pipeline combining **BM25 full-text search**, **vector semantic search**, and **LLM re-ranking**.

## Search Pipeline

```
                         User Query
                             |
               +-------------+-------------+
               v                           v
        Query Expansion              Original Query
        (LLM: 2 variants)             (x2 weight)
               |                           |
               +-------------+-------------+
                             |
          +------------------+------------------+
          v                  v                  v
     Original Q         Expanded Q1        Expanded Q2
     BM25 + Vector      BM25 + Vector      BM25 + Vector
          |                  |                  |
          +------------------+------------------+
                             |
                    RRF Fusion + Bonus
                    (original x2, top-rank +0.05)
                    Keep top 30
                             |
                    LLM Re-ranking
                    (yes/no + confidence per doc)
                             |
                    Position-Aware Blend
                    Top 1-3:  75% RRF / 25% rerank
                    Top 4-10: 60% RRF / 40% rerank
                    Top 11+:  40% RRF / 60% rerank
```

## Features

- **Clone any git repo** via URL - walks all source files and indexes them
- **BM25 full-text search** via [tantivy](https://github.com/quickwit-oss/tantivy)
- **Vector semantic search** with embeddings from Ollama or OpenAI-compatible APIs
- **LLM re-ranking** with yes/no relevance judgments and position-aware score blending
- **Query expansion** using the LLM to generate alternative search phrasings
- **Multi-query RRF fusion** combining results across original + expanded queries
- Dark-themed web UI with real-time pipeline status

## Quick Start

### Prerequisites

- Rust toolchain (1.75+)
- For vector search & re-ranking: [Ollama](https://ollama.ai) running locally

### Setup Ollama (optional, for full pipeline)

```bash
# Install ollama, then pull models:
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.2           # chat/reranking
```

### Build & Run

```bash
cargo build --release
./target/release/repo-search
```

The server starts on `http://localhost:3000`.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REPO_SEARCH_DATA_DIR` | `./data` | Where repos and indexes are stored |
| `REPO_SEARCH_BIND_ADDR` | `0.0.0.0:3000` | Server bind address |
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API base URL |
| `LLM_CHAT_MODEL` | `llama3.2` | Model for re-ranking & query expansion |
| `LLM_EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings |
| `LLM_EMBEDDING_DIM` | `768` | Embedding vector dimension |
| `LLM_API_KEY` | - | API key (cloud providers only) |

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/repos` | List all repositories |
| `POST` | `/api/repos` | Add a repo `{"url": "..."}` |
| `DELETE` | `/api/repos/{id}` | Delete a repo and its index |
| `POST` | `/api/search` | Search `{"query": "...", "use_bm25": true, "use_vector": true, "use_rerank": false}` |
| `GET` | `/api/config` | Get LLM configuration |
| `PUT` | `/api/config` | Update LLM configuration |

## Architecture

- **Axum** web framework with async handlers
- **git2** for cloning repositories
- **tantivy** for BM25 full-text indexing and search
- **In-memory vector store** with cosine similarity (persisted to JSON)
- **Ollama / OpenAI** compatible API for embeddings and chat
- File chunking: ~100 lines per chunk with 20-line overlap
- Supports 60+ programming languages and file types
