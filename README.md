# repo-search

A Rust web app for cloning git repositories and searching them with a hybrid pipeline combining **BM25 full-text search**, **vector semantic search**, and **LLM re-ranking**.

## Search Pipeline DAG

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Hybrid Search Pipeline (DAG)                              │
└─────────────────────────────────────────────────────────────────────────────────┘

                                ┌─────────────┐
                                │  User Query  │
                                └──────┬───────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                 ┌────────────────┐       ┌─────────────────┐
                 │ Query Expansion│       │  Original Query  │
                 │  (LLM: 2 alt)  │       │   (×2 weight)   │
                 └───────┬────────┘       └────────┬────────┘
                         │ 2 alternatives          │
                         └────────────┬────────────┘
                                      │ 3 queries total
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
     │  Original Query  │    │ Expanded Query 1│    │ Expanded Query 2│
     └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
              │                      │                      │
       ┌──────┴──────┐       ┌──────┴──────┐       ┌──────┴──────┐
       ▼             ▼       ▼             ▼       ▼             ▼
   ┌───────┐    ┌───────┐ ┌───────┐  ┌───────┐ ┌───────┐  ┌───────┐
   │ BM25  │    │Vector │ │ BM25  │  │Vector │ │ BM25  │  │Vector │
   │tantivy│    │cosine │ │tantivy│  │cosine │ │tantivy│  │cosine │
   └───┬───┘    └───┬───┘ └───┬───┘  └───┬───┘ └───┬───┘  └───┬───┘
       │            │         │           │         │           │
       └─────┬──────┘         └─────┬─────┘         └─────┬─────┘
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │ 6 ranked lists
                                    ▼
                       ┌───────────────────────┐
                       │   RRF Fusion + Bonus  │
                       │  Original query: ×2   │
                       │  Top-rank bonus: +0.05│
                       │     Top 30 Kept       │
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │    LLM Re-ranking     │
                       │  yes/no + confidence   │
                       │  per-document scoring  │
                       │  4 concurrent workers  │
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │  Position-Aware Blend │
                       │  Top 1-3:  75% RRF    │
                       │  Top 4-10: 60% RRF    │
                       │  Top 11+:  40% RRF    │
                       └───────────┬───────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │    Final Results      │
                       │  (user-specified limit)│
                       └───────────────────────┘
```

### Indexing Pipeline DAG

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Indexing Pipeline (DAG)                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │  Git URL      │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  git2 clone  │─────── clone to data/repos/<uuid>/
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Walk Files  │─────── skip hidden dirs, node_modules, target, etc.
     │  60+ langs   │        skip binary files, >1MB files
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  Chunk Files │─────── 100 lines/chunk, 20 line overlap
     └──────┬───────┘
            │
            ├──────────────────────────┐
            ▼                          ▼
     ┌──────────────┐          ┌──────────────────┐
     │ Tantivy BM25 │          │ LLM Embeddings   │
     │ Index        │          │ (Ollama/OpenAI)   │
     │ (sync, disk) │          │ batch, 32 at a    │
     └──────────────┘          │ time              │
                               └────────┬─────────┘
                                        │
                                        ▼
                               ┌──────────────────┐
                               │ Vector Store      │
                               │ (memory + JSON)   │
                               └──────────────────┘
```

## Module Architecture

```
src/
├── lib.rs              # Crate root with pipeline DAG documentation
├── main.rs             # Axum server entrypoint
├── config.rs           # Env-based config (data dirs, LLM settings, bind address)
├── models.rs           # Shared types: Repo, FileChunk, SearchHit, SearchRequest
├── state.rs            # AppState: shared BM25 + vector + config + HTTP client
│
├── git/
│   ├── mod.rs
│   └── clone.rs        # git2 clone, file walking, language detection (60+ langs)
│
├── search/
│   ├── mod.rs
│   ├── bm25.rs         # Tantivy BM25 index (create/search/delete)
│   ├── vector.rs       # In-memory cosine similarity store (add/search/delete/persist)
│   └── hybrid.rs       # Multi-query RRF fusion with weighted scoring + top-rank bonus
│
├── llm/
│   ├── mod.rs
│   ├── embeddings.rs   # Batch embedding (Ollama /api/embed, OpenAI /v1/embeddings)
│   ├── query_expand.rs # Query expansion: 2 alternative phrasings via LLM chat
│   └── rerank.rs       # Yes/no per-doc re-ranking + position-aware score blending
│
└── api/
    ├── mod.rs
    ├── repos.rs        # POST/GET/DELETE repos, clone+index pipeline, chunk_file()
    └── search.rs       # Full search orchestration: expand → search × 3 → RRF → rerank

tests/
└── integration_test.rs # End-to-end tests: indexing, search, multi-repo, fusion, delete
```

## Features

- **Clone any git repo** via URL - walks all source files and indexes them
- **BM25 full-text search** via [tantivy](https://github.com/quickwit-oss/tantivy) - no LLM required
- **Vector semantic search** with embeddings from Ollama or OpenAI-compatible APIs
- **Query expansion** - LLM generates 2 alternative search phrasings
- **Multi-query RRF fusion** - combines 6 ranked lists (3 queries x 2 retrieval methods) with weighted scoring
- **LLM re-ranking** - per-document yes/no + confidence judgments
- **Position-aware blending** - top results preserve RRF ordering, lower results defer to LLM
- **Graceful degradation** - BM25 works standalone; vector/LLM features skip if unavailable
- Dark-themed web UI with pipeline status visualization

## Quick Start

### Prerequisites

- Rust toolchain (1.75+)
- For vector search & re-ranking: [Ollama](https://ollama.ai) running locally

### Setup Ollama (optional, for full pipeline)

```bash
# Install ollama, then pull models:
ollama pull nomic-embed-text   # embeddings (768-dim)
ollama pull llama3.2           # chat/reranking/query expansion
```

### Build & Run

```bash
cargo build --release
./target/release/repo-search
```

The server starts on `http://localhost:9000`.

### Run Tests

```bash
cargo test
```

95 tests: 89 unit tests + 6 integration tests covering:
- BM25 indexing, search, deletion, persistence, repo filtering
- Cosine similarity edge cases, vector store CRUD, persistence
- Multi-query RRF fusion, weighting, deduplication, top-rank bonus
- File walking, language detection, binary filtering
- LLM response parsing (JSON, markdown-wrapped, keyword fallback)
- Query expansion parsing (clean JSON, embedded, Unicode)
- File chunking (overlap, boundaries, content integrity)
- End-to-end index → search → fusion pipelines

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REPO_SEARCH_DATA_DIR` | `./data` | Where repos and indexes are stored |
| `REPO_SEARCH_BIND_ADDR` | `127.0.0.1:9000` | Server bind address (localhost only by default) |
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API base URL |
| `LLM_CHAT_MODEL` | `llama3.2` | Model for re-ranking & query expansion |
| `LLM_EMBEDDING_MODEL` | `nomic-embed-text` | Model for embeddings |
| `LLM_EMBEDDING_DIM` | `768` | Embedding vector dimension |
| `LLM_API_KEY` | - | API key (cloud providers only) |
| `REPO_SEARCH_GIT_TOKEN` | - | Git PAT for cloning private repos |
| `REPO_SEARCH_MAX_REPOS` | `50` | Maximum number of repos allowed |
| `REPO_SEARCH_MAX_REPO_SIZE_MB` | `500` | Maximum repo size after clone (MB) |
| `REPO_SEARCH_CLONE_TIMEOUT_SECS` | `300` | Clone timeout in seconds |
| `REPO_SEARCH_MAX_CONCURRENT_CLONES` | `2` | Parallel clone limit |
| `REPO_SEARCH_MAX_VECTOR_ENTRIES` | `500000` | Max in-memory vector entries |

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/repos` | List all repositories |
| `POST` | `/api/repos` | Add a repo `{"url": "https://..."}` |
| `DELETE` | `/api/repos/{id}` | Delete a repo and its index |
| `POST` | `/api/search` | Hybrid search (see below) |
| `GET` | `/api/config` | Get LLM configuration |
| `PUT` | `/api/config` | Update LLM configuration (`base_url` is read-only) |

### Search Request

```json
{
  "query": "database connection pool",
  "limit": 20,
  "use_bm25": true,
  "use_vector": true,
  "use_rerank": false,
  "repo_ids": null
}
```

- `use_bm25` / `use_vector`: toggle retrieval methods independently
- `use_rerank`: enables query expansion + LLM re-ranking (requires running LLM)
- `repo_ids`: optional array of UUIDs to filter results to specific repos

### Search Response

```json
{
  "query": "database connection pool",
  "results": [
    {
      "repo_id": "...",
      "repo_name": "my-service",
      "file_path": "src/db.rs",
      "chunk_index": 0,
      "content": "pub async fn connect(url: &str) ...",
      "language": "rust",
      "start_line": 1,
      "end_line": 100,
      "bm25_score": 5.23,
      "vector_score": 0.87,
      "combined_score": 0.0328,
      "rerank_score": null
    }
  ],
  "total_bm25_hits": 15,
  "total_vector_hits": 12
}
```

## Key Design Decisions

- **Reciprocal Rank Fusion (RRF)** over score normalization: RRF doesn't require calibrating scores across different retrieval methods
- **×2 weight for original query**: prevents query expansion from dominating when the original already matches well
- **Top-rank bonus (+0.05)**: rewards documents that rank #1 in any retrieval list
- **Position-aware blending**: top RRF results are likely already good, so they keep more of their RRF score; lower-ranked results benefit more from LLM re-ranking
- **100-line chunks with 20-line overlap**: balances context window efficiency with avoiding split-function boundaries
- **Best-effort vector indexing**: if Ollama isn't running, BM25 indexing still completes successfully
