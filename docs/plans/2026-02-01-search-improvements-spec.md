# Search Pipeline Improvements: Technical Specification

**Date:** 2026-02-01
**Status:** Draft
**Companion:** [PRD](./2026-02-01-search-improvements-prd.md)

---

## Implementation Phases

```
Phase 1 ─── Items 1+2+6 (single re-index) ─── Quality foundation
Phase 2 ─── Items 4+3 (no re-index) ──────── Query hardening
Phase 3 ─── Item 7 (no re-index) ──────────── Caching
Phase 4 ─── Item 5 (no re-index) ──────────── Reranker upgrade
Phase 5 ─── Item 8 (no re-index) ──────────── Vector scalability
```

---

## SPEC-1: Embedding Task Prefixes

### Files Modified

- `src/llm/embeddings.rs` -- Add `EmbedTask` enum and prefix application
- `src/api/repos.rs` -- Pass `EmbedTask::SearchDocument` at embed call sites (lines 295, 513, 666)
- `src/api/search.rs` -- Pass `EmbedTask::SearchQuery` at embed call site (line 85)
- `src/config.rs` -- Add prefix lookup method to `LlmConfig`

### Design

```rust
// src/llm/embeddings.rs

#[derive(Debug, Clone, Copy)]
pub enum EmbedTask {
    SearchDocument,
    SearchQuery,
}

impl EmbedTask {
    /// Returns the prefix for the given model family.
    /// Returns empty string for models that don't use prefixes.
    pub fn prefix_for_model(&self, model_name: &str) -> &'static str {
        if model_name.contains("nomic-embed") {
            match self {
                EmbedTask::SearchDocument => "search_document: ",
                EmbedTask::SearchQuery => "search_query: ",
            }
        } else if model_name.contains("e5-") {
            match self {
                EmbedTask::SearchDocument => "passage: ",
                EmbedTask::SearchQuery => "query: ",
            }
        } else {
            ""
        }
    }
}

pub async fn embed_batch(
    client: &reqwest::Client,
    config: &LlmConfig,
    texts: &[String],
    task: EmbedTask,
) -> Result<Vec<Vec<f32>>> {
    let prefix = task.prefix_for_model(&config.embedding_model);
    let prefixed: Vec<String> = texts.iter()
        .map(|t| format!("{prefix}{t}"))
        .collect();
    let truncated: Vec<String> = prefixed.iter()
        .map(|t| truncate_for_embedding(t).to_string())
        .collect();
    // ... existing provider dispatch
}
```

### Migration

- Adding prefixes changes the embedding vector for all existing content.
- All documents MUST be re-embedded after this change.
- Combined with Phase 1 re-index (items 2 and 6).

---

## SPEC-2: AST-Aware Code Chunking

### New Files

- `src/chunking.rs` -- Chunking module with strategy dispatch
- `src/chunking/ast.rs` -- Tree-sitter AST chunker (cAST split-then-merge algorithm)
- `src/chunking/fallback.rs` -- Blank-line-aware text chunker

### Files Modified

- `src/api/repos.rs` -- Replace `chunk_file` calls (lines 265, 478, 628) with new chunker
- `src/models.rs` -- Extend `FileChunk` with `scope`, `entity_type`, `contextualized`
- `Cargo.toml` -- Add tree-sitter + grammar crates

### Dependencies

```toml
# Cargo.toml additions
tree-sitter = "0.24"
tree-sitter-rust = "0.23"
tree-sitter-python = "0.23"
tree-sitter-javascript = "0.23"
tree-sitter-typescript = "0.23"
tree-sitter-go = "0.23"
tree-sitter-java = "0.23"
tree-sitter-c = "0.23"
tree-sitter-cpp = "0.23"
tree-sitter-ruby = "0.23"
tree-sitter-bash = "0.23"
tree-sitter-json = "0.24"
tree-sitter-html = "0.23"
tree-sitter-css = "0.23"
tree-sitter-md = "0.4"
```

### Algorithm: cAST Split-Then-Merge

```
ChunkCode(code, max_nws_chars=1500):
    tree = tree_sitter::parse(code, language)
    if tree has >30% error nodes:
        return fallback_chunk(code)
    return chunk_nodes(tree.root.children, max_nws_chars)

chunk_nodes(nodes, budget):
    chunks = []
    current_nodes = []
    current_size = 0
    for node in nodes:
        node_size = count_non_whitespace(node.text)
        if node_size > budget:
            flush(current_nodes -> chunks)
            chunks.extend(chunk_nodes(node.children, budget))
        elif current_size + node_size <= budget:
            current_nodes.push(node)
            current_size += node_size
        else:
            flush(current_nodes -> chunks)
            current_nodes = [node]
            current_size = node_size
    flush(current_nodes -> chunks)
    return chunks
```

### Language-to-Grammar Mapping

```rust
fn get_language(lang: &str) -> Option<tree_sitter::Language> {
    match lang {
        "rust"       => Some(tree_sitter_rust::LANGUAGE.into()),
        "python"     => Some(tree_sitter_python::LANGUAGE.into()),
        "javascript" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" => Some(tree_sitter_typescript::language_typescript()),
        "tsx"        => Some(tree_sitter_typescript::language_tsx()),
        "go"         => Some(tree_sitter_go::LANGUAGE.into()),
        "java"       => Some(tree_sitter_java::LANGUAGE.into()),
        "c"          => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp"        => Some(tree_sitter_cpp::LANGUAGE.into()),
        "ruby"       => Some(tree_sitter_ruby::LANGUAGE.into()),
        "bash" | "shell" => Some(tree_sitter_bash::LANGUAGE.into()),
        "json"       => Some(tree_sitter_json::LANGUAGE.into()),
        "html"       => Some(tree_sitter_html::LANGUAGE.into()),
        "css"        => Some(tree_sitter_css::LANGUAGE.into()),
        "markdown"   => Some(tree_sitter_md::LANGUAGE.into()),
        _            => None, // fallback to line-based
    }
}
```

### Fallback: Smart Line-Based Chunker

When no grammar is available or AST parsing fails:

```
1. Split at blank lines (double newline)
2. Merge adjacent blocks until reaching chunk budget
3. If a single block exceeds budget, split at single newlines
4. If a single line exceeds budget, split at character boundary
```

### Metadata Enrichment

For each AST chunk, collect:

```rust
pub struct ChunkMetadata {
    pub scope: Option<String>,       // "AuthService > validate_token"
    pub entity_type: Option<String>, // "function", "class", "impl", "module"
    pub signature: Option<String>,   // "fn validate_token(&self, token: &str) -> Result<Claims>"
}
```

Embedding text becomes:

```
search_document: # src/auth/service.rs
# Scope: AuthService > validate_token
# Defines: fn validate_token(&self, token: &str) -> Result<Claims>

pub fn validate_token(&self, token: &str) -> Result<Claims> {
    ...
}
```

---

## SPEC-3: Strong Signal Detection

### Files Modified

- `src/api/search.rs` -- Restructure pipeline to run BM25 first, then conditionally expand
- `src/search/bm25.rs` -- Add BM25 score normalization

### BM25 Score Normalization

Tantivy BM25 scores are unbounded. Normalize to 0-1 for portable thresholds:

```rust
/// Normalize a Tantivy BM25 score to 0-1 range.
/// Uses the formula: score / (score + k) where k is a tunable constant.
fn normalize_bm25(raw_score: f32) -> f32 {
    let k = 10.0; // tunable; higher k = more conservative normalization
    raw_score / (raw_score + k)
}
```

### Pipeline Restructure

```rust
pub async fn run_search(state, query, repo_ids, use_bm25, use_vector, use_rerank, limit) {
    // Stage 1: BM25-only search with original query
    let bm25_results = bm25_search(query, limit * 2, repo_ids);

    // Stage 1.5: Strong signal detection
    if use_rerank && detect_strong_signal(&bm25_results) {
        tracing::info!("Strong signal detected, skipping expansion+reranking");
        return format_results(bm25_results, limit);
    }

    // Stage 2: Query expansion (only if not skipped)
    let queries = expand_queries(query, use_rerank);

    // Stage 3: Multi-query BM25 + vector search
    // ... existing pipeline continues
}

fn detect_strong_signal(results: &[Bm25Hit]) -> bool {
    if results.len() < 2 { return false; }
    let top1 = normalize_bm25(results[0].score);
    let top2 = normalize_bm25(results[1].score);
    let gap = top1 - top2;
    top1 >= 0.85 && gap >= 0.15
}
```

### Post-Fusion Rerank Gating (Stage 2)

After RRF fusion, before LLM reranking:

```rust
fn should_skip_rerank(fused: &[SearchHit]) -> bool {
    if fused.len() < 2 { return true; }
    // Both BM25 and vector agree on top result
    let agreement = fused[0].bm25_score > 0.0 && fused[0].vector_score > 0.0;
    let gap = fused[0].combined_score - fused[1].combined_score;
    agreement && gap >= 0.005 // RRF scores are small; 0.005 is significant
}
```

---

## SPEC-4: Query Expansion Filtering

### Files Modified

- `src/api/search.rs` -- Add filtering after `expand_query()` returns (lines 39-45)
- `src/llm/query_expand.rs` -- (optional) Add structured output with Ollama JSON schema

### Term Overlap Filter

```rust
const STOPWORDS: &[&str] = &[
    "what", "is", "how", "to", "the", "a", "an", "in", "on", "for",
    "of", "and", "or", "with", "my", "your", "do", "does", "can",
    "i", "me", "we", "who", "where", "when", "why", "which",
    "find", "get", "show", "tell", "this", "that",
];

fn filter_expansions(original: &str, expansions: Vec<String>) -> Vec<String> {
    let query_terms: Vec<String> = original
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| !STOPWORDS.contains(&w.as_str()))
        .collect();

    if query_terms.is_empty() {
        return expansions; // all stopwords, can't filter
    }

    expansions.into_iter().filter(|exp| {
        let exp_lower = exp.to_lowercase();

        // Reject if no original terms present
        let has_overlap = query_terms.iter().any(|t| exp_lower.contains(t.as_str()));
        if !has_overlap {
            tracing::info!("Filtered expansion (no term overlap): {exp}");
            return false;
        }

        // Reject echo (trivial rewording)
        if is_echo(original, exp) {
            tracing::info!("Filtered expansion (echo): {exp}");
            return false;
        }

        // Reject generic filler
        if has_generic_prefix(exp) {
            tracing::info!("Filtered expansion (generic): {exp}");
            return false;
        }

        true
    }).collect()
}

fn is_echo(original: &str, expansion: &str) -> bool {
    let o = original.to_lowercase();
    let e = expansion.to_lowercase();
    e == o
        || (e.contains(&o) && e.len() < o.len() + 10)
        || (o.contains(&e) && o.len() < e.len() + 10)
}

const GENERIC_PREFIXES: &[&str] = &[
    "find information about",
    "search for",
    "look up",
    "tell me about",
    "learn about",
];

fn has_generic_prefix(expansion: &str) -> bool {
    let lower = expansion.to_lowercase();
    GENERIC_PREFIXES.iter().any(|p| lower.starts_with(p))
}
```

---

## SPEC-5: Dedicated Cross-Encoder Reranker

### New Files

- `src/llm/reranker.rs` -- Reranker trait and cross-encoder implementation

### Files Modified

- `src/config.rs` -- Add `RerankerConfig` struct
- `src/state.rs` -- Add reranker client to `AppState`
- `src/llm/rerank.rs` -- Refactor to use `Reranker` trait, keep as LLM fallback
- `src/api/search.rs` -- Use tiered reranking

### Configuration

```rust
// src/config.rs
#[derive(Clone, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Base URL of the reranker endpoint (e.g., "http://localhost:8081")
    pub base_url: Option<String>,
    /// Model name (e.g., "Qwen3-Reranker-0.6B")
    pub model: Option<String>,
    /// Timeout for reranker requests
    #[serde(default = "default_reranker_timeout")]
    pub timeout_secs: u64,
}
```

### Reranker Trait

```rust
// src/llm/reranker.rs

#[derive(Debug)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_n: usize,
    ) -> Result<Vec<RerankResult>>;
}

/// Dedicated cross-encoder (calls /v1/rerank endpoint)
pub struct CrossEncoderReranker {
    client: reqwest::Client,
    base_url: String,
    model: String,
    timeout: Duration,
}

#[async_trait]
impl Reranker for CrossEncoderReranker {
    async fn rerank(&self, query: &str, documents: &[&str], top_n: usize)
        -> Result<Vec<RerankResult>>
    {
        let body = serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        });
        let resp: RerankResponse = self.client
            .post(format!("{}/v1/rerank", self.base_url))
            .json(&body)
            .timeout(self.timeout)
            .send().await?
            .error_for_status()?
            .json().await?;
        Ok(resp.results.into_iter().map(|r| RerankResult {
            index: r.index,
            score: sigmoid(r.relevance_score),
        }).collect())
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
```

### Tiered Fallback

```rust
pub async fn rerank_with_fallback(
    state: &AppState,
    query: &str,
    hits: &mut [SearchHit],
) -> RerankTier {
    // Tier 1: Cross-encoder
    if let Some(ref reranker) = state.reranker {
        if !state.reranker_circuit.is_open() {
            match reranker.rerank(query, &docs, 10).await {
                Ok(scores) => { apply_scores(hits, scores); return RerankTier::CrossEncoder; }
                Err(e) => { state.reranker_circuit.record_failure(); }
            }
        }
    }
    // Tier 2: LLM-based (existing)
    match llm_rerank(state, query, hits).await {
        Ok(()) => RerankTier::Llm,
        Err(_) => RerankTier::RrfOnly,
    }
}
```

### Recommended Model Setup

```bash
# Run llama-server as a reranker sidecar
llama-server \
  --model Qwen3-Reranker-0.6B-Q5_K_M.gguf \
  --port 8081 \
  --embedding \
  --pooling rank \
  --ctx-size 512
```

---

## SPEC-6: Content-Hash Deduplication

### New Files

- `src/hashing.rs` -- BLAKE3 content hashing utilities

### Files Modified

- `src/search/vector.rs` -- Refactor storage to separate content from locations
- `src/api/repos.rs` -- Add dedup logic before embedding calls
- `src/models.rs` -- Add `content_hash` to `FileChunk`
- `Cargo.toml` -- Add `blake3 = "1.8"`

### Core Hashing

```rust
// src/hashing.rs
pub fn content_hash(text: &str) -> blake3::Hash {
    blake3::hash(text.as_bytes())
}

pub fn content_hash_hex(text: &str) -> String {
    content_hash(text).to_hex().to_string()
}

pub fn file_hash(content: &str) -> String {
    blake3::hash(content.as_bytes()).to_hex().to_string()
}
```

### Dedup-Aware Embedding

```rust
// In src/api/repos.rs, replace the direct embed_batch call:

let mut texts_to_embed: Vec<String> = Vec::new();
let mut embed_indices: Vec<usize> = Vec::new();
let mut cached_embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(all_chunks.len());

let embedding_store = state.vector_store.read();
for (i, chunk) in all_chunks.iter().enumerate() {
    let hash = content_hash_hex(&chunk.content);
    if let Some(existing) = embedding_store.get_by_hash(&hash) {
        cached_embeddings.push(Some(existing));
    } else {
        texts_to_embed.push(format!("File: {}\n{}", chunk.file_path, chunk.content));
        embed_indices.push(i);
        cached_embeddings.push(None);
    }
}

// Only embed chunks that aren't already cached
let new_embeddings = if texts_to_embed.is_empty() {
    vec![]
} else {
    embed_batch(&state.http_client, &config, &texts_to_embed, EmbedTask::SearchDocument).await?
};

// Merge cached + new embeddings
let mut all_embeddings = Vec::with_capacity(all_chunks.len());
let mut new_idx = 0;
for cached in cached_embeddings {
    match cached {
        Some(emb) => all_embeddings.push(emb),
        None => {
            all_embeddings.push(new_embeddings[new_idx].clone());
            new_idx += 1;
        }
    }
}
```

### Incremental Re-index

```rust
// File-level change detection
pub struct IndexedFileInfo {
    pub file_hash: String,
    pub chunk_hashes: Vec<String>,
}

pub enum FileAction {
    Skip,                          // file unchanged
    Reindex(Vec<FileChunk>),       // file changed, re-chunk
    New(Vec<FileChunk>),           // new file
}

fn plan_file_reindex(
    path: &str,
    content: &str,
    previous: Option<&IndexedFileInfo>,
) -> FileAction {
    let new_hash = file_hash(content);
    match previous {
        Some(prev) if prev.file_hash == new_hash => FileAction::Skip,
        Some(_) => FileAction::Reindex(chunk_file(path, content)),
        None => FileAction::New(chunk_file(path, content)),
    }
}
```

### Embedding Config Validation

```rust
#[derive(Serialize, Deserialize)]
pub struct VectorStoreMetadata {
    pub schema_version: u32,
    pub embedding_model: String,
    pub embedding_dim: usize,
    pub chunking_version: u32,
    pub prefix_convention: String, // e.g., "nomic" or "none"
}

// On startup: if metadata doesn't match current config, warn + treat store as empty
```

---

## SPEC-7: Pipeline Caching (moka)

### New Files

- `src/cache.rs` -- `PipelineCaches` struct and cache configuration

### Files Modified

- `src/state.rs` -- Add `PipelineCaches` and `QueryHistory` to `AppState`
- `src/api/search.rs` -- Wire cache lookups/inserts around expansion, embedding, reranking
- `src/llm/rerank.rs` -- Check rerank cache before LLM calls
- `src/api/repos.rs` -- Call `invalidate_on_reindex()` in sync endpoint
- `Cargo.toml` -- Add `moka = { version = "0.12", features = ["future"] }`

### Cache Configuration

```rust
// src/cache.rs
use moka::future::Cache;
use std::time::Duration;

#[derive(Clone)]
pub struct PipelineCaches {
    pub expansion: Cache<u64, Vec<String>>,
    pub embeddings: Cache<u64, Vec<f32>>,
    pub rerank: Cache<u64, f32>,
}

impl PipelineCaches {
    pub fn new() -> Self {
        Self {
            expansion: Cache::builder()
                .max_capacity(1_000)
                .time_to_live(Duration::from_secs(3600))
                .time_to_idle(Duration::from_secs(900))
                .build(),
            embeddings: Cache::builder()
                .max_capacity(32 * 1024 * 1024) // 32MB weight budget
                .weigher(|_: &u64, v: &Vec<f32>| (v.len() as u32) * 4 + 24)
                .time_to_live(Duration::from_secs(14400))
                .time_to_idle(Duration::from_secs(1800))
                .build(),
            rerank: Cache::builder()
                .max_capacity(50_000)
                .time_to_live(Duration::from_secs(7200))
                .time_to_idle(Duration::from_secs(1800))
                .build(),
        }
    }

    pub fn invalidate_on_reindex(&self) {
        self.rerank.invalidate_all();
        // expansion + embeddings are query-only, not content-dependent
    }

    pub fn invalidate_all(&self) {
        self.expansion.invalidate_all();
        self.embeddings.invalidate_all();
        self.rerank.invalidate_all();
    }
}
```

### Cache Key Function

```rust
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn cache_key(components: &[&str]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for c in components { c.hash(&mut hasher); }
    hasher.finish()
}
```

### Integration Points

```rust
// In search.rs -- expansion cache
let key = cache_key(&[&config.chat_model, &query]);
let expansions = match state.caches.expansion.get(&key).await {
    Some(cached) => cached,
    None => {
        let result = expand_query(...).await?;
        state.caches.expansion.insert(key, result.clone()).await;
        result
    }
};

// In search.rs -- embedding cache
let key = cache_key(&[&config.embedding_model, q]);
let embedding = match state.caches.embeddings.get(&key).await {
    Some(cached) => cached,
    None => {
        let emb = embed_single(..., EmbedTask::SearchQuery).await?;
        state.caches.embeddings.insert(key, emb.clone()).await;
        emb
    }
};

// In rerank.rs -- per-chunk score cache
let chunk_hash = content_hash_hex(&content);
let key = cache_key(&[&config.chat_model, query, &chunk_hash]);
let score = match caches.rerank.get(&key).await {
    Some(cached) => cached,
    None => {
        let s = score_single(...).await?;
        caches.rerank.insert(key, s).await;
        s
    }
};
```

### Query History for Cache Warming

```rust
pub struct QueryHistory {
    queries: parking_lot::Mutex<std::collections::VecDeque<String>>,
    capacity: usize,
}

impl QueryHistory {
    pub fn new(capacity: usize) -> Self { ... }
    pub fn record(&self, query: String) { ... }
    pub fn recent(&self) -> Vec<String> { ... }
}
```

---

## SPEC-8: HNSW Vector Index (usearch)

### Files Modified

- `src/search/vector.rs` -- Replace internals with usearch
- `Cargo.toml` -- Add `usearch = "2"`

### Design: Dynamic Index

```rust
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub struct VectorStore {
    // Metadata lookup: usearch key -> entry metadata
    metadata: RwLock<Vec<VectorMetadata>>,
    // HNSW index (None when entry count < threshold)
    hnsw: RwLock<Option<Index>>,
    // Flat store for small datasets and legacy compat
    flat: RwLock<Vec<(Vec<f32>, usize)>>, // (embedding, metadata_index)
    // Threshold for switching from flat to HNSW
    hnsw_threshold: usize, // default: 10_000
    persist_dir: PathBuf,
}

pub struct VectorMetadata {
    pub repo_id: Uuid,
    pub repo_name: String,
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub content_hash: String,
    pub language: String,
    pub start_line: usize,
    pub end_line: usize,
}
```

### Search Implementation

```rust
impl VectorStore {
    pub fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        repo_ids: Option<&[Uuid]>,
    ) -> Vec<VectorHit> {
        let meta = self.metadata.read();
        let count = meta.len();

        if count == 0 { return vec![]; }

        if count < self.hnsw_threshold {
            // Brute-force (faster for small datasets)
            self.search_flat(query_embedding, limit, repo_ids)
        } else {
            // HNSW approximate search
            self.search_hnsw(query_embedding, limit, repo_ids)
        }
    }

    fn search_hnsw(
        &self,
        query: &[f32],
        limit: usize,
        repo_ids: Option<&[Uuid]>,
    ) -> Vec<VectorHit> {
        let index = self.hnsw.read();
        let index = index.as_ref().expect("HNSW index should exist above threshold");

        // Over-fetch to account for repo_ids filtering
        let fetch_limit = if repo_ids.is_some() { limit * 3 } else { limit };
        let results = index.search(query, fetch_limit).unwrap();

        let meta = self.metadata.read();
        results.keys.iter().zip(results.distances.iter())
            .filter_map(|(&key, &distance)| {
                let m = &meta[key as usize];
                if let Some(ids) = repo_ids {
                    if !ids.contains(&m.repo_id) { return None; }
                }
                Some(VectorHit {
                    repo_id: m.repo_id,
                    repo_name: m.repo_name.clone(),
                    file_path: m.file_path.clone(),
                    chunk_index: m.chunk_index,
                    content: m.content.clone(),
                    language: m.language.clone(),
                    start_line: m.start_line,
                    end_line: m.end_line,
                    score: 1.0 - distance, // usearch cosine returns distance
                })
            })
            .take(limit)
            .collect()
    }
}
```

### HNSW Configuration

```rust
fn create_hnsw_index(dim: usize) -> Index {
    let options = IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::BF16, // 2x memory reduction
        connectivity: 16,       // M parameter
        expansion_add: 200,     // ef_construction
        expansion_search: 100,  // ef parameter
    };
    let index = usearch::new_index(&options).unwrap();
    index
}
```

### Persistence

```rust
// Save: usearch native binary format
fn save(&self) -> Result<()> {
    if let Some(ref index) = *self.hnsw.read() {
        index.save(&self.persist_dir.join("vectors.usearch"))?;
    }
    // Metadata saved separately as bincode (much faster than JSON)
    let meta = self.metadata.read();
    let encoded = bincode::serialize(&*meta)?;
    std::fs::write(self.persist_dir.join("vectors.meta.bin"), encoded)?;
    Ok(())
}

// Load: check for usearch file first, fall back to legacy JSON
fn load(&self) -> Result<()> {
    let usearch_path = self.persist_dir.join("vectors.usearch");
    let legacy_path = self.persist_dir.join("vectors.json");

    if usearch_path.exists() {
        // Load native format
        let index = create_hnsw_index(self.dim);
        index.load(&usearch_path)?;
        *self.hnsw.write() = Some(index);
    } else if legacy_path.exists() {
        // Backfill from legacy JSON
        tracing::info!("Migrating from vectors.json to HNSW index...");
        self.backfill_from_json(&legacy_path)?;
    }
    Ok(())
}
```

### Migration Path

1. On first startup after upgrade, detect `vectors.json` without `vectors.usearch`.
2. Load legacy entries, insert into HNSW index, save in native format.
3. Rename `vectors.json` to `vectors.json.bak` (do not delete).
4. All subsequent operations use native format.

---

## Observability Additions

### Structured Tracing Spans

Add to `src/api/search.rs`:

```rust
#[tracing::instrument(skip(state), fields(
    pipeline.expansion_ms,
    pipeline.embedding_ms,
    pipeline.bm25_ms,
    pipeline.vector_ms,
    pipeline.rerank_ms,
    pipeline.total_ms,
    pipeline.strong_signal,
    pipeline.cache_hits,
    pipeline.rerank_tier,
))]
pub async fn run_search(...) { ... }
```

### Expected Latency After All Phases

| Scenario | Current | After All Phases |
|----------|---------|-----------------|
| Cold query, full pipeline | 1.5-5s | 0.4-1.6s |
| Warm cache (repeat query) | 1.5-5s | <10ms |
| Strong signal (exact match) | 1.5-5s | <50ms |
| Chat (search + generation) | 5-20s | 3.5-12s |

The LLM generation phase (3-10s) becomes the dominant cost once search is optimized.

---

## Dependency Summary

| Phase | New Crates | Binary Size Impact |
|-------|-----------|-------------------|
| 1 | `blake3`, `tree-sitter`, 14 grammar crates | +5-10MB |
| 2 | (none) | 0 |
| 3 | `moka` | +500KB |
| 4 | (none, external llama-server process) | 0 |
| 5 | `usearch`, `bincode` | +2-3MB |

---

## Review Findings & Revisions

*Added 2026-02-01 after multi-agent review (architecture, performance, security, data integrity, simplicity).*

### Critical Issues (Must Fix Before Implementation)

#### CF-1: BM25 normalization threshold will almost never trigger (SPEC-3)

**Problem:** With `k=10`, a raw Tantivy BM25 score of ~57 is needed to reach the 0.85 threshold. Typical code search scores cluster in the 5-30 range. The PRD claims 20-40% of queries bypass expansion, but with these thresholds it would be closer to 2-5%.

**Fix:** Lower `k` to 3-5 (with k=5, a raw score of 28.3 normalizes to 0.85, which is realistic for exact function name matches). Alternatively, collect a histogram of actual BM25 scores from live queries before finalizing thresholds. Add a minimum result count requirement (at least 5 results) to prevent single-result false positives.

#### CF-2: HNSW over-fetch with repo filtering is insufficient (SPEC-8)

**Problem:** The 3x multiplier assumes the target repo holds ≥33% of all vectors. With 10 repos of equal size, searching 1 repo means only ~10% of vectors match -- you'd get ~3 results instead of 10. With 50 repos, search effectively returns nothing.

**Fix:** Either (a) compute the multiplier dynamically from `repo_counts()` as `limit * (total_vectors / repo_vectors)` capped at `limit * 100`, or (b) use usearch's filtered search API with a callback predicate. Option (b) is correct but requires verifying usearch's Rust bindings support it.

#### CF-3: HNSW does not support deletion; `delete_repo` will break (SPEC-8)

**Problem:** The current `VectorStore::delete_repo` uses `entries.retain()`. usearch HNSW indexes have no native deletion. The spec does not address this, and repo deletion is a core user operation.

**Fix options:** (a) Maintain a tombstone set and rebuild HNSW periodically, (b) use `hnsw_rs` (pure Rust) which supports deletion, (c) rebuild the index on every `delete_repo` (acceptable if deletions are rare -- typically 0-1/day).

#### CF-4: HNSW metadata vec indexing assumes contiguous keys (SPEC-8)

**Problem:** `meta[key as usize]` assumes usearch keys are contiguous 0..N. After deletions and re-insertions, keys may not be contiguous. This causes index-out-of-bounds panics or wrong-metadata lookups.

**Fix:** Use `HashMap<u64, VectorMetadata>` instead of `Vec<VectorMetadata>`, or implement a free-list for key reuse.

#### CF-5: No atomic writes for HNSW persistence (SPEC-8)

**Problem:** `save()` writes directly to `vectors.usearch` and `vectors.meta.bin`. A crash mid-write produces a corrupt file that the next startup tries to load. The existing `persist_repos` in `state.rs` correctly uses write-to-temp + rename, but the HNSW spec does not.

**Fix:** Write to `vectors.usearch.tmp` then rename to `vectors.usearch`. Same for `vectors.meta.bin`. Validate consistency on load: `index.size() == metadata.len()`.

#### CF-6: Phase 1 re-index has no crash recovery (SPEC-1/2/6)

**Problem:** `reindex_embeddings` deletes old vectors before generating new ones. A crash between deletion and embedding completion leaves a repo with zero vectors. With Phase 1 re-indexing all repos, this window is wide.

**Fix:** Add per-repo `needs_reindex` flag in `VectorStoreMetadata`. On startup, resume any incomplete re-indexes. Better: generate new embeddings into a staging area, then atomically swap old for new.

### High Priority Issues

#### HF-1: Rerank cache key uses wrong model after Phase 4 (SPEC-7)

**Problem:** The rerank cache key uses `config.chat_model` (line 703), but after Phase 4 introduces the cross-encoder, reranking may come from either the cross-encoder or the LLM fallback. Wrong model in the key would serve cached scores from the wrong reranker.

**Fix:** Key rerank cache by the actual model used: `reranker_model` when using cross-encoder, `chat_model` when falling back to LLM.

#### HF-2: Cache key uses deterministic DefaultHasher (SPEC-7)

**Problem:** `DefaultHasher::new()` uses a fixed seed (not `RandomState`). The hash function is unkeyed and predictable. While 64-bit collision is unlikely by accident, a motivated attacker who can submit queries could precompute collisions to poison the cache with wrong results.

**Fix:** Initialize a `RandomState` once at startup, store in `PipelineCaches`, and build hashers from it. Alternatively, use a 128-bit hash (e.g., `blake3::hash` truncated to u128) to push the birthday bound to 2^64.

#### HF-3: Tree-sitter parsing needs timeout and size gate (SPEC-2)

**Problem:** Tree-sitter is a C library called via FFI. Known issues include infinite loops in error recovery and OOM on pathological inputs. Any public GitHub repo can be added, so this is attacker-controlled input.

**Fix:** (a) Skip tree-sitter for files >500KB (fall back to line-based). (b) Wrap `Parser::parse()` in a `tokio::time::timeout` of 5 seconds. (c) Pin exact tree-sitter version: `tree-sitter = "=0.24.x"`.

#### HF-4: Signal detection thresholds need recalibration after Phase 1 (SPEC-3)

**Problem:** AST chunking (Phase 1) changes chunk sizes dramatically. BM25 scoring distribution will shift because document lengths change. The hardcoded `k=10.0` and thresholds need re-tuning against post-Phase-1 data.

**Fix:** Add an explicit calibration step between Phase 1 and Phase 2. Run test queries against the AST-chunked corpus and collect a BM25 score histogram before finalizing normalization constants.

#### HF-5: Cache invalidation not wired to config update endpoint (SPEC-7)

**Problem:** R7.7 requires flushing all caches when LLM config changes. The `update_config` handler in `repos.rs` does not call `caches.invalidate_all()`. The spec does not list this as a modified file.

**Fix:** Add `state.caches.invalidate_all()` to the `update_config` handler when `chat_model` or `embedding_model` changes.

#### HF-6: Incremental re-index skips unchanged files after algorithm change (SPEC-6)

**Problem:** `file_hash` is computed from raw file content. If the file didn't change but the chunking algorithm changed (100-line → AST), `FileAction::Skip` fires. Stored chunks are from the old algorithm.

**Fix:** Include `chunking_version` in the file hash key, or clear all `IndexedFileInfo` records when `VectorStoreMetadata` detects a config mismatch. The BM25 index must also be rebuilt (not just vectors).

### Simplification Recommendations

Based on the simplicity review, these changes reduce complexity without sacrificing the core improvements:

#### SR-1: Remove QueryHistory ring buffer (SPEC-7)

Dead weight. No background process pre-warms caches. The moka caches already provide "warm on second query" behavior naturally. Remove the `QueryHistory` struct entirely (~30 lines).

#### SR-2: Replace circuit breaker with simple fallback (SPEC-5)

For a localhost single-user tool, the circuit breaker pattern is over-engineered. Replace with a simple `match reranker.rerank(...).await { Ok => ..., Err => fall through }`. The reqwest timeout already prevents hangs (~30 lines saved).

#### SR-3: Drop the Reranker trait (SPEC-5)

One implementation (`CrossEncoderReranker`). A trait with one impl is premature abstraction. Use a concrete struct. Extract a trait if/when a second implementation materializes.

#### SR-4: Simplify to two-tier reranking (SPEC-5)

Drop the LLM fallback tier entirely: cross-encoder → RRF-only. The LLM reranker is the thing being replaced (30 chat calls, 1-3s, fragile JSON parsing). RRF-only at <50ms is better UX than waiting 1-3s for the flaky fallback. This eventually allows deleting ~200 lines of `rerank.rs`.

#### SR-5: Trim grammar crates to 6 (SPEC-2)

Start with: Rust, Python, JavaScript, TypeScript/TSX, Go (6 crates). Fallback line-based chunker handles everything else adequately. JSON/HTML/CSS/Markdown have flat structure where AST adds no value. Bash scripts are typically short. YAML/TOML have no well-maintained tree-sitter grammars anyway. Add more grammars later when needed. Saves 8 dependencies, ~3-5MB binary size.

#### SR-6: Replace VectorStoreMetadata with a single config hash (SPEC-6)

The 5-field struct answers one question: "are stored embeddings compatible with current config?" A single hash string achieves the same thing:

```rust
fn embedding_config_hash(config: &LlmConfig) -> String {
    let mut h = DefaultHasher::new();
    config.embedding_model.hash(&mut h);
    config.embedding_dim.hash(&mut h);
    CHUNKING_VERSION.hash(&mut h);
    format!("{:016x}", h.finish())
}
```

Store one file (`vectors.config_hash`). On startup, compare. Mismatch = warn + re-index.

#### SR-7: Defer HNSW to when it's actually needed (SPEC-8)

At typical scale (1-5 repos, 5K-50K vectors), brute-force cosine takes ~5-25ms -- imperceptible. Replace JSON persistence with bincode for faster load/save (the actual startup bottleneck). This removes the `usearch` dependency, the dual-mode switching logic, the migration code, and the deletion problem (CF-3/CF-4). If the corpus grows past 100K vectors, upgrade to always-HNSW (no flat path) in a single step.

#### SR-8: Drop ChunkMetadata.signature field (SPEC-2)

The signature is the first line of the chunk's code. Extracting it as a separate field requires per-language tree-sitter queries. The embedding text already contains the raw code. Use `content.lines().next()` in the embedding prefix if needed. Keep `scope` and `entity_type`.

### Missing Specifications (PRD → SPEC gaps)

| PRD Requirement | Gap |
|---|---|
| R2.6: Include relevant imports in chunk metadata | Not in SPEC-2. `ChunkMetadata` has scope/entity_type but no imports field. |
| R3.4: Never skip for multi-part queries | Not in SPEC-3. No heuristic for detecting multi-part queries. |
| R5.7: Circuit breaker with recovery period | Referenced but never defined. No struct, thresholds, or implementation. (See SR-2: drop it.) |
| R5.8: Include `rerank_tier` in search response | `RerankTier` enum exists but not added to `SearchResponse` in models.rs. |
| R7.9: Expose cache hit rates | No metrics collection in SPEC-7. |
| R8.7: Maintain `delete_repo` API | Not addressed. usearch has no deletion support. (See CF-3.) |
| YAML/TOML grammars (R2.2) | Listed in PRD but no tree-sitter crates exist. (Handled by fallback chunker.) |
| `embed_single` call site | Not listed in SPEC-1 files-modified. `embed_single` at `embeddings.rs:55` delegates to `embed_batch` and needs the `EmbedTask` parameter. |
| `SearchResponse` changes | `rerank_tier` field not added to response struct or API JSON. |

### Latency Attribution Clarification

The SPEC summary table (line 913) conflates quality improvements with latency improvements. For clarity:

| Phase | Primary Effect | Cold-Path Latency Impact |
|---|---|---|
| 1 (Prefixes + AST + Dedup) | Retrieval quality | ~0 (slightly slower indexing, same search speed) |
| 2 (Expansion filter + Signal) | Quality + conditional bypass | 20-40% of queries skip to <50ms |
| 3 (Caching) | Repeat query speed | Cache hits: <10ms. Cold: unchanged |
| 4 (Cross-encoder) | **Main latency win** | Reranking: 1-3s → 50-100ms |
| 5 (HNSW) | Vector search at scale | <5ms at 500K vs ~750ms brute-force |

The cold-path improvement from 1.5-5s to 0.4-1.6s is **almost entirely attributable to Phase 4** (cross-encoder). Phases 1-3 are quality and conditional optimizations.

### Concurrency Notes

- **Lock ordering for SPEC-8:** If keeping HNSW, document canonical lock order: acquire metadata lock first, then hnsw/flat lock. All methods must follow this order.
- **Reranker semaphore:** The current 4-concurrent semaphore in `rerank.rs` should be skipped for the cross-encoder path (single batch HTTP call).
- **Moka caches:** Lock-free for reads (internal sharding). No deadlock risk.

### Additional Fix: Reranker Security (SPEC-5)

- Lock down `reranker_base_url` against runtime modification (same SSRF protection as main LLM `base_url`).
- Validate `base_url` starts with `http://127.0.0.1` or `http://localhost` unless explicitly overridden.
- Cap `timeout_secs` at 30 seconds maximum.
- Construct the reranker's `reqwest::Client` with both `connect_timeout(5s)` and `timeout(30s)`.
