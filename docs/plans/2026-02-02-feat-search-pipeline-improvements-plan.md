# Search Pipeline Quality & Performance Improvements — Implementation Plan

**Date:** 2026-02-02
**Type:** feat (epic)
**Status:** Ready for implementation
**Companion docs:** [PRD](./2026-02-01-search-improvements-prd.md) · [SPEC](./2026-02-01-search-improvements-spec.md)

---

## Overview

This epic transforms the search pipeline from a working prototype into a production-quality retrieval system. The 8 improvements span retrieval quality (AST chunking, embedding prefixes, content dedup), query intelligence (expansion filtering, strong signal detection), latency (pipeline caching, cross-encoder reranker), and scalability (bincode persistence).

The improvements are sequenced into 6 phases to minimize re-index passes, respect data dependencies, and allow calibration between phases. Each phase is independently deployable and testable.

**Why this matters:** The chatbot's answer quality is bounded by retrieval quality. When the search pipeline returns the wrong code chunks, even the best LLM produces wrong answers. These improvements attack the root cause: better chunks, better ranking, and faster iteration.

---

## Problem Statement

### Current Limitations

1. **Chunking is naive.** Fixed 100-line chunks with 20-line overlap split functions mid-body, producing semantically incomplete fragments that degrade both embedding quality and LLM reasoning.

2. **Embeddings ignore model requirements.** nomic-embed-text requires asymmetric task prefixes (`search_query:` / `search_document:`). We embed without any prefix, degrading the embedding space geometry.

3. **Every query pays full latency.** No caching, no short-circuiting. Even repeated queries or obvious exact-matches run the full 4-stage pipeline (expand → embed → search → rerank), taking 1.5-5 seconds.

4. **Reranking is the bottleneck.** 30 individual LLM chat calls for pointwise relevance scoring takes 1-3 seconds. A dedicated cross-encoder does this in 50-100ms with better calibrated scores.

5. **Query expansion drifts.** LLM-generated expansions are accepted without validation. Off-topic expansions dilute RRF results and can push correct answers off the first page.

6. **Identical content is embedded repeatedly.** LICENSE files, .gitignore, vendored code — the same text across repos gets separate embedding API calls and separate storage.

7. **JSON persistence is slow at scale.** The vector store serializes to JSON. At 100K+ vectors, startup parsing takes 10+ seconds. At 500K, it's 30+ seconds.

### What Success Looks Like

| Metric | Current | Target |
|--------|---------|--------|
| Cold query latency | 1.5-5s | 0.4-1.6s |
| Warm cache latency | 1.5-5s | <10ms |
| Strong signal latency | 1.5-5s | <50ms |
| Functions split mid-body | Many | Zero (for supported languages) |
| Embedding API calls on re-index | N (all chunks) | 0.1-0.2N (dedup + incremental) |
| Reranking latency | 1-3s | 50-100ms |

---

## Revised Phase Plan

*Incorporates all findings from the multi-agent review (architecture, performance, security, data integrity, simplicity).*

| Phase | Theme | PRD Items | Re-index? | Key Dependencies |
|-------|-------|-----------|-----------|-----------------|
| **0** | API cleanup | VectorStore refactor | No | None |
| **1** | Quality foundation | #1 (prefixes) + #2 (AST) + #6 (dedup) | Yes (once) | Phase 0 |
| **2** | Query hardening | #4 (expansion filter) then #3 (signal detection) | No | Phase 1 (calibration data) |
| **3** | Pipeline caching | #7 (moka caches) | No | Phase 2 (pipeline stable) |
| **4** | Reranker upgrade | #5 (cross-encoder) | No | Phase 3 (cache keys need reranker model) |
| **5** | Persistence performance | #8 (bincode, defer HNSW) | No | When needed |

### Why This Order

- **Phase 0 before Phase 1:** The `add_chunks` 6-tuple is error-prone and will get worse with new metadata fields. Fix the API surface once before adding content hashes, scope chains, entity types.
- **Phase 1 is one big re-index pass.** Prefixes, AST chunking, and content hashing all invalidate existing embeddings. Doing them together means one re-index, not three.
- **Phase 2 must follow Phase 1.** AST chunking changes BM25 score distributions. Signal detection thresholds (Phase 2) must be calibrated against post-AST data, not pre-AST data.
- **Phase 2 internal ordering:** Expansion filtering (PRD-4) first, then signal detection (PRD-3). Signal detection evaluates BM25 results that may have been influenced by expansions. Clean expansions first.
- **Phase 3 before Phase 4.** Cache key design needs to account for the reranker model (cross-encoder vs LLM). Implementing caching after the pipeline is stable prevents key invalidation churn.
- **Phase 4 last of the core work.** The cross-encoder is the main latency win (reranking 1-3s → 50-100ms) but is an external sidecar dependency. The quality improvements (Phases 1-3) are more foundational.
- **Phase 5 deferred.** At typical scale (5K-50K vectors), brute-force cosine is <25ms. Replace JSON with bincode for faster startup. HNSW only when corpus exceeds 100K vectors.

---

## Phase 0: VectorStore API Refactor

### Motivation

`VectorStore::add_chunks` currently accepts a `Vec<(String, usize, String, String, usize, usize)>` — a 6-element tuple where position determines meaning. This is fragile, unreadable, and will get worse when Phase 1 adds `content_hash`, `scope`, and `entity_type`.

### Tasks

#### P0.1: Define AddChunkEntry struct

**Files:** `src/search/vector.rs`

Replace the tuple with a named struct:

```rust
pub struct AddChunkEntry {
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub language: String,
    pub start_line: usize,
    pub end_line: usize,
}
```

Update `add_chunks` signature: `fn add_chunks(&self, repo_id: Uuid, repo_name: &str, entries: &[AddChunkEntry], embeddings: Vec<Vec<f32>>) -> Result<()>`

#### P0.2: Update all call sites

**Files:** `src/api/repos.rs` (3 call sites: lines ~362-373, ~580-591, ~732-743)

Replace the tuple construction with struct construction at each site.

#### P0.3: Update tests

**Files:** `src/search/vector.rs` (unit tests), `tests/integration_test.rs`

Update any test helpers that build the tuple format.

**Acceptance criteria:**
- [ ] No tuple in `add_chunks` signature
- [ ] All 3 call sites use struct
- [ ] `cargo test` passes
- [ ] No public API change to `search()` or `delete_repo()`

---

## Phase 1: Quality Foundation (Single Re-index)

### Motivation

This phase attacks the three biggest quality gaps simultaneously: wrong embedding geometry (no prefixes), semantically broken chunks (100-line splits), and wasted embedding effort (no dedup). Combining them into one phase means a single re-index pass for all repos.

### Task P1.1: Embedding Task Prefixes (SPEC-1)

#### Background

nomic-embed-text was trained with asymmetric prefixes. The model card explicitly states: "The text prompt must include a task instruction prefix." Without prefixes, query and document embeddings occupy the same region of the embedding space, reducing the model's ability to distinguish "what I'm looking for" from "what this document contains."

#### P1.1a: Add EmbedTask enum

**File:** `src/llm/embeddings.rs`

Add `EmbedTask` enum with `SearchDocument` and `SearchQuery` variants. Implement `prefix_for_model(&self, model_name: &str) -> &'static str` that returns the appropriate prefix (nomic: `search_document: ` / `search_query: `; E5: `passage: ` / `query: `; unknown: empty string).

**Critical detail:** The space after the colon is required. `search_document:text` silently degrades quality vs `search_document: text`.

#### P1.1b: Update embed_batch and embed_single signatures

**File:** `src/llm/embeddings.rs`

Add `task: EmbedTask` parameter. Apply prefix before truncation (prefix must never be truncated away). Update the internal `truncate_for_embedding` to account for prefix length.

**Note:** `embed_single` (line ~55) delegates to `embed_batch`. Both need the new parameter. The SPEC originally missed `embed_single` in the files-modified list.

#### P1.1c: Update all embed call sites

**Files:** `src/api/repos.rs` (3 sites: index, reindex, sync — pass `EmbedTask::SearchDocument`), `src/api/search.rs` (1 site: query embedding — pass `EmbedTask::SearchQuery`)

#### P1.1d: Add embedding config hash validation

**File:** `src/search/vector.rs`

On startup, compute a hash of `(embedding_model, embedding_dim, chunking_version)` and compare against stored value. If mismatched, log a warning and treat the vector store as needing re-index.

Per simplification recommendation SR-6: use a single hash string rather than a 5-field struct. Store in `vectors.config_hash` file.

**Acceptance criteria:**
- [ ] nomic-embed-text queries prefixed with `search_query: `
- [ ] nomic-embed-text documents prefixed with `search_document: `
- [ ] E5 models use `query: ` / `passage: ` prefixes
- [ ] Unknown models work without prefix (no regression)
- [ ] Config hash mismatch detected and logged on startup
- [ ] Tests for prefix selection by model name

---

### Task P1.2: AST-Aware Code Chunking (SPEC-2)

#### Background

The cAST paper (EMNLP 2025) demonstrates that AST-aware chunking improves retrieval precision by +1.2 to +4.3 points over naive line-based splitting. The key insight: code has syntactic structure (functions, classes, modules). Splitting at these boundaries produces semantically complete chunks that embed better and are more useful in RAG context.

#### Design decisions after review

- **6 grammars, not 14** (SR-5): Rust, Python, JavaScript, TypeScript/TSX, Go. These cover the primary use case. The fallback line-based chunker handles everything else adequately. JSON/HTML/CSS/Markdown have flat structure where AST adds no value.
- **Drop signature field** (SR-8): The signature is just the first line of code. Use `content.lines().next()` if needed. Keep `scope` and `entity_type`.
- **File size gate** (HF-3): Skip tree-sitter for files >500KB. Fall back to line-based.
- **Parse timeout** (HF-3): Wrap `Parser::parse()` in 5-second timeout. Tree-sitter is C FFI with known infinite-loop bugs.
- **Pin tree-sitter version** (HF-3): Use exact version pins to prevent regressions from auto-updates.

#### P1.2a: Add tree-sitter dependencies

**File:** `Cargo.toml`

Add `tree-sitter = "=0.24.x"` and 6 grammar crates (rust, python, javascript, typescript, go, java). Use exact version pins for tree-sitter core.

**Consideration:** Java was included in the original 6 because many enterprise repos use it and it has complex class/interface/method nesting that benefits most from AST chunking.

#### P1.2b: Implement cAST split-then-merge algorithm

**New file:** `src/chunking/ast.rs`

Implement the algorithm from SPEC-2: parse with tree-sitter, chunk at syntactic boundaries. If a single node exceeds the 1500 non-whitespace character budget, recurse into its children. Merge adjacent small nodes up to the budget.

**Error threshold:** If >30% of AST nodes are error nodes, fall back to line-based chunker.

#### P1.2c: Implement smart line-based fallback

**New file:** `src/chunking/fallback.rs`

Four-tier fallback: split at blank lines → merge to budget → split at single newlines → split at character boundary. This replaces the current naive 100-line splitter for unsupported languages.

#### P1.2d: Add metadata extraction

**Files:** `src/chunking/ast.rs`, `src/models.rs`

For each AST chunk, extract:
- `scope: Option<String>` — parent scope chain (e.g., `AuthService > validate_token`)
- `entity_type: Option<String>` — function, class, impl, module, etc.

Extend `FileChunk` in models.rs with these fields. Prepend metadata to embedding text.

**Security note (from review):** Cap identifier lengths at 128 chars in scope chain to limit prompt injection via adversarial function names. Use `// scope:` prefix format (less likely to be interpreted as instructions).

#### P1.2e: Add safety guards

**File:** `src/chunking/ast.rs`

- File size gate: files >500KB → fallback chunker
- Parse timeout: wrap `Parser::parse()` call in 5-second timeout via `tokio::time::timeout` around `spawn_blocking`
- Pin tree-sitter version in Cargo.toml

#### P1.2f: Integrate into repos.rs

**File:** `src/api/repos.rs`

Replace all `chunk_file()` calls (3 sites) with the new chunking dispatcher. The dispatcher checks language support and routes to AST or fallback.

#### P1.2g: Remove 20-line overlap

**File:** `src/api/repos.rs`

Delete the overlap logic. AST boundaries + metadata enrichment make overlap unnecessary. This simplifies the chunk model and slightly reduces total chunk count.

**Acceptance criteria:**
- [ ] Zero functions split mid-body for Rust, Python, JS, TS, Go, Java
- [ ] Fallback chunker handles all other languages without error
- [ ] Files >500KB use fallback (no tree-sitter parsing)
- [ ] Parse timeout prevents hangs on pathological files
- [ ] Scope chain present on chunks from supported languages
- [ ] Indexing throughput within 2x of current speed
- [ ] Tests: AST chunking for each supported language, fallback behavior, error threshold, timeout

---

### Task P1.3: Content-Hash Deduplication (SPEC-6)

#### Background

Across typical repo collections, 5-15% of chunks are duplicates (LICENSE, .gitignore, boilerplate, vendored code). On incremental re-index after a small git pull, 80-95% of files are unchanged. BLAKE3 hashing at 256-bit output provides 128-bit collision resistance (computationally infeasible to exploit).

#### P1.3a: Add BLAKE3 hashing utilities

**New file:** `src/hashing.rs`
**File:** `Cargo.toml` — add `blake3 = "1.8"`

Functions: `content_hash(text) -> blake3::Hash`, `content_hash_hex(text) -> String`, `file_hash(content) -> String`.

#### P1.3b: Refactor VectorStore for content-hash keying

**File:** `src/search/vector.rs`

Add `content_hash` field to `VectorEntry`. Support `get_by_hash()` lookup for dedup-aware embedding.

#### P1.3c: Implement dedup-aware embedding

**File:** `src/api/repos.rs`

Before calling `embed_batch`, check each chunk's content hash against the vector store. Only embed chunks with unknown hashes. Merge cached + new embeddings.

**Correctness note (from security review):** The content hash should cover the same text that gets embedded (including metadata prefix). If two chunks have identical code but different file paths, and the embedding text includes the file path, they should have different hashes.

#### P1.3d: Add file-level hash for incremental re-index

**Files:** `src/search/vector.rs`, `src/api/repos.rs`

Store per-file hashes alongside repo metadata. On sync (git pull), compute file hashes first. Skip unchanged files entirely. For changed files, re-chunk and diff at chunk level.

#### P1.3e: Include chunking_version in file hash key (HF-6 fix)

**File:** `src/hashing.rs`

If the file content hasn't changed but the chunking algorithm version has changed, we must re-chunk. Include `CHUNKING_VERSION` constant in the file hash computation so algorithm changes force re-processing. Clear all `IndexedFileInfo` when `VectorStoreMetadata` detects a config mismatch. The BM25 index must also be rebuilt.

**Acceptance criteria:**
- [ ] Duplicate chunks across repos share one embedding
- [ ] Incremental re-index skips unchanged files
- [ ] Chunking version change forces full re-chunk
- [ ] BLAKE3 hashes use full 256-bit output (no truncation)
- [ ] Tests: dedup detection, incremental skip, version change detection

---

### Task P1.4: Re-index Trigger and Crash Recovery

#### Background

Phase 1 invalidates all existing chunks and embeddings. On first startup after upgrade, detect the mismatch via embedding config hash and trigger a full re-index. Per CF-6, re-indexing must be crash-safe: if the server dies mid-re-index, it should resume on next startup.

#### P1.4a: Startup config mismatch detection

**File:** `src/state.rs`

On startup, compare stored `embedding_config_hash` against current config. If mismatched or missing, set all repos to `needs_reindex` status.

#### P1.4b: Background re-index with crash recovery

**File:** `src/api/repos.rs`

On startup, scan repos for `needs_reindex` flag. Spawn background re-index tasks. Generate new embeddings into a staging area, then atomically swap old for new. If interrupted, the flag persists and re-index resumes on next startup.

**Acceptance criteria:**
- [ ] Config change triggers automatic re-index on startup
- [ ] Crash mid-re-index does not lose data
- [ ] Re-index resumes after restart
- [ ] Progress visible in repo status (status: "re-indexing")

---

## Phase 2: Query Pipeline Hardening

### Motivation

Phase 1 improves what we index. Phase 2 improves how we query. Expansion filtering removes noise from LLM-generated query variants. Strong signal detection skips the expensive pipeline entirely when BM25 alone has a definitive answer.

**Critical ordering:** Implement expansion filtering FIRST (P2.1), validate it works, THEN implement signal detection (P2.3) on top. Signal detection evaluates BM25 results quality. If expansions are noisy, the BM25 results being evaluated are polluted.

### Task P2.1: Query Expansion Filtering (SPEC-4)

#### P2.1a: Add stopword list and term overlap filter

**File:** `src/api/search.rs` (or new `src/llm/expansion_filter.rs`)

Implement `filter_expansions(original, expansions) -> Vec<String>`. Reject any expansion sharing zero non-stopword terms with the original query. If all expansions filtered, continue with original query only.

#### P2.1b: Add echo detection

Reject expansions that are identical to or trivially extend the original query (within 10 chars difference).

#### P2.1c: Add generic prefix rejection

Reject expansions starting with filler phrases: "find information about", "search for", "look up", "tell me about", "learn about".

#### P2.1d: Wire into search pipeline

**File:** `src/api/search.rs`

Insert `filter_expansions()` call after `expand_query()` returns, before building the multi-query search. Log filtered expansions with reasons.

**Acceptance criteria:**
- [ ] Off-topic expansions filtered (manual inspection of logs)
- [ ] Echo expansions caught
- [ ] Generic filler rejected
- [ ] All-filtered case continues with original query
- [ ] Zero latency cost (string operations only)
- [ ] Tests: each filter type, edge cases, all-filtered fallback

---

### Task P2.2: BM25 Score Normalization and Calibration (SPEC-3 prereq)

#### Background

Tantivy BM25 scores are unbounded TF-IDF values. For portable thresholds, normalize to 0-1 using `score / (score + k)`. The k constant determines the score at which normalization reaches 0.5.

**Critical finding (CF-1):** With k=10, a raw score of ~57 is needed to reach 0.85 normalized. Typical code search scores cluster at 5-30. The bypass rate would be 2-5%, not the projected 20-40%.

#### P2.2a: Add normalize_bm25 function

**File:** `src/search/bm25.rs`

Implement `normalize_bm25(raw_score: f32, k: f32) -> f32`. Make k configurable (default 5.0, to be calibrated).

#### P2.2b: Add BM25 score histogram logging

**File:** `src/api/search.rs`

After Phase 1 re-index, log raw BM25 scores for all queries in a structured format. Collect data over representative usage to calibrate thresholds.

#### P2.2c: Tune k constant and thresholds

Based on collected histogram data, set k so that exact function name matches normalize to 0.85+ while typical keyword queries normalize to 0.3-0.6. This is a data-driven step — do not hardcode until calibrated.

**Acceptance criteria:**
- [ ] Normalization function implemented and tested
- [ ] Score histogram logging available
- [ ] k constant tuned against real data (not theoretical)
- [ ] Tests: normalization edge cases (0, very large, typical range)

---

### Task P2.3: Strong Signal Detection (SPEC-3)

#### P2.3a: Stage 1 — Pre-expansion BM25-only gating

**File:** `src/api/search.rs`

Restructure `run_search` to run BM25 with the original query FIRST. If `detect_strong_signal()` triggers, return BM25 results directly — skip expansion, vector search, and reranking entirely.

Thresholds: top1 >= threshold_high AND gap to top2 >= threshold_gap. Both configurable, calibrated from P2.2.

**Minimum result count (from review):** Require at least 5 BM25 results before triggering strong signal. A single result with a high score and large gap means thin corpus, not definitive answer.

#### P2.3b: Stage 2 — Post-fusion rerank gating

After RRF fusion, before LLM reranking: if BM25 and vector agree on the top result AND the RRF score gap is significant, skip reranking.

#### P2.3c: Multi-part query heuristic (R3.4)

Never trigger strong signal detection for queries containing "and" between question clauses. Simple heuristic: split on " and ", check if both parts contain non-stopword content.

#### P2.3d: Observability

Log when signal detection triggers, which stage, and the score distribution that triggered it. This is essential for monitoring false positive rates.

**Acceptance criteria:**
- [ ] 20-40% of exact-match queries bypass expansion+reranking
- [ ] Bypassed queries return in <50ms
- [ ] No recall degradation on test query set
- [ ] Multi-part queries never bypassed
- [ ] All triggers logged with scores
- [ ] Tests: trigger scenarios, non-trigger scenarios, multi-part detection

---

## Phase 3: Pipeline Caching

### Motivation

Users refine questions iteratively in the chatbot. Without caching, each refinement re-runs the entire pipeline (expansion, embedding, reranking) even when the underlying search hasn't changed. With caching, repeated or similar queries return in <10ms.

### Task P3.1: Add moka caches infrastructure

**New file:** `src/cache.rs`
**File:** `Cargo.toml` — add `moka = { version = "0.12", features = ["future"] }`
**File:** `src/state.rs` — add `PipelineCaches` to `AppState`

Create `PipelineCaches` with three caches: expansion (1000 entries, 1hr TTL), embeddings (32MB weight budget, 4hr TTL), rerank scores (50K entries, 2hr TTL).

**Cache key security (HF-2):** Use `RandomState`-seeded hasher initialized once at process start, stored in `PipelineCaches`. This prevents precomputed collision attacks. Alternatively use `blake3::hash` truncated to u128 for the birthday bound at 2^64.

**Drop QueryHistory (SR-1):** No background pre-warming process exists. Moka provides "warm on second query" naturally. Remove the ring buffer from the spec.

### Task P3.2: Implement expansion cache

**File:** `src/api/search.rs`

Cache key: `hash(chat_model, query)`. Before calling `expand_query()`, check cache. On miss, expand and insert.

### Task P3.3: Implement embedding cache

**File:** `src/api/search.rs`

Cache key: `hash(embedding_model, text)`. Before calling `embed_single()` for query embeddings, check cache. On miss, embed and insert. Use moka's weigher for the embedding cache (each entry ~3KB for 768-dim f32).

### Task P3.4: Implement rerank score cache

**File:** `src/llm/rerank.rs`

Cache key: `hash(actual_model_used, query, chunk_content_hash)`. Per-chunk score caching. **Use the actual reranker model, not `chat_model`** (HF-1 fix). When cross-encoder is active, key by `reranker_model`; when falling back to LLM, key by `chat_model`.

### Task P3.5: Wire cache invalidation

**Files:** `src/api/repos.rs` (sync/reindex endpoints), config update handler

- On repo sync/re-index: flush rerank cache (content changed). Keep expansion and embedding caches (query-only).
- On LLM config change (`chat_model` or `embedding_model`): flush all caches.
- Wire `state.caches.invalidate_all()` into the `update_config` handler (HF-5 fix).

**Invariant to document:** Expansion cache keys must depend only on query text and model name, never on indexed content. If expansion prompts ever include content context, this assumption breaks and the invalidation logic must update.

### Task P3.6: Cache hit rate observability

Add counters for cache hits/misses per cache type. Log periodically or expose via a `/api/stats` endpoint.

**Acceptance criteria:**
- [ ] 30-60% cache hit rate for repeated/similar queries
- [ ] Cache-hit searches return in <10ms
- [ ] Memory stays within ~50MB budget
- [ ] Cache invalidation works on reindex and config change
- [ ] No QueryHistory ring buffer (removed per SR-1)
- [ ] Cache keys use RandomState or blake3 (not DefaultHasher)
- [ ] Tests: cache hit/miss, invalidation, key correctness, memory bounds

---

## Phase 4: Cross-Encoder Reranker

### Motivation

Reranking is the dominant latency bottleneck. The current approach makes 30 individual LLM chat calls (each a full generative inference pass), taking 1-3 seconds with fragile JSON parsing. A dedicated 0.6B cross-encoder processes all 30 pairs in a single forward pass (~50-100ms) with better calibrated scores.

**This is the main latency win.** The cold-path improvement from 1.5-5s to 0.4-1.6s is almost entirely attributable to this phase.

### Design decisions after review

- **No Reranker trait** (SR-3): One implementation. Concrete struct. Extract a trait if/when a second implementation materializes.
- **Two-tier fallback only** (SR-4): Cross-encoder → RRF-only. Drop the LLM fallback entirely. The LLM reranker is the bottleneck being replaced. RRF-only at <50ms is better UX than waiting 1-3s for the flaky fallback.
- **No circuit breaker** (SR-2): For a localhost single-user tool, `match reranker.rerank().await { Ok => ..., Err => fall through }` is sufficient. The reqwest timeout already prevents hangs.

### Task P4.1: Add reranker configuration

**File:** `src/config.rs`

Add `RerankerConfig` with `base_url: Option<String>`, `model: Option<String>`, `timeout_secs: u64` (capped at 30). Environment variables: `RERANKER_BASE_URL`, `RERANKER_MODEL`.

**Security (from review):**
- Lock down `reranker_base_url` against runtime modification via `PUT /api/config` (same SSRF protection as main LLM `base_url`).
- Validate URL starts with `http://127.0.0.1` or `http://localhost` unless `RERANKER_ALLOW_REMOTE=true`.

### Task P4.2: Implement CrossEncoderReranker

**New file:** `src/llm/reranker.rs`

Concrete struct with `rerank(query, documents, top_n) -> Result<Vec<RerankResult>>`. Sends a single batch request to `/v1/rerank` endpoint. Normalizes raw logits via sigmoid to 0-1 range.

Construct reqwest client with `connect_timeout(5s)` and `timeout(30s)` at the client level, regardless of configurable `timeout_secs`.

### Task P4.3: Implement two-tier fallback

**File:** `src/api/search.rs`

```
if reranker configured {
    match reranker.rerank(...) {
        Ok(scores) => apply and return CrossEncoder tier
        Err(e) => log warning, fall through to RRF-only
    }
} else {
    RRF-only
}
```

Skip the existing 4-concurrent semaphore for the cross-encoder path (single batch HTTP call).

### Task P4.4: Add sigmoid normalization

**File:** `src/llm/reranker.rs`

`sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }`. Apply to raw cross-encoder logits before blending with RRF scores. Maintain existing position-aware blending (75/25 at top-3, 60/40 at 4-10, 40/60 at 11+).

### Task P4.5: Wire into search pipeline

**File:** `src/api/search.rs`

Replace the `rerank()` call path with the new tiered approach. Ensure cache keys in Phase 3 use the correct model name (cross-encoder model vs chat model).

### Task P4.6: Add rerank_tier to SearchResponse

**Files:** `src/models.rs`, `src/api/search.rs`

Add `rerank_tier: String` field ("cross-encoder", "rrf-only") to `SearchResponse`. The frontend can display this for debugging.

### Task P4.7: Eventually delete old LLM reranker (after validation)

**File:** `src/llm/rerank.rs`

Once the cross-encoder is validated in production, the old pointwise LLM reranker (~200 lines) can be deleted. This also removes the JSON parsing fallbacks and the 4-concurrent semaphore.

**Keep this as a separate task — don't delete during the initial phase.**

**Acceptance criteria:**
- [ ] Reranking latency 50-100ms (vs 1-3s)
- [ ] Search P95 under 1 second
- [ ] Graceful fallback to RRF-only when reranker unavailable
- [ ] SSRF protection on reranker URL
- [ ] Timeout capped at 30s
- [ ] rerank_tier visible in search response
- [ ] Tests: rerank scoring, sigmoid normalization, fallback behavior, timeout

---

## Phase 5: Persistence Performance

### Motivation

Per SR-7, defer HNSW until corpus exceeds 100K vectors. At typical scale (5K-50K vectors), brute-force cosine is <25ms. The actual bottleneck is JSON persistence: parsing a 50K-entry JSON file takes several seconds on startup. Bincode is 5-10x faster.

### Task P5.1: Replace JSON persistence with bincode

**File:** `src/search/vector.rs`
**File:** `Cargo.toml` — add `bincode = "1"`

Replace `serde_json::to_string_pretty` / `from_str` with `bincode::serialize` / `deserialize` for vector store persistence.

**Atomic writes:** Write to `vectors.bin.tmp`, then rename to `vectors.bin`. Validate on load.

**Migration:** On startup, check for `vectors.json`. If found and no `vectors.bin` exists, load from JSON, save as bincode, rename JSON to `.bak`.

### Task P5.2: (Future) HNSW when >100K vectors

Deferred. When the vector store grows past 100K entries, implement HNSW with the fixes from CF-2 through CF-5:
- Dynamic over-fetch multiplier for repo filtering
- Deletion via tombstone set + periodic rebuild
- HashMap for metadata lookup (not Vec)
- Atomic persistence writes

**This task is a placeholder. Do not implement until the threshold is reached.**

**Acceptance criteria:**
- [ ] Startup time with 50K vectors: <1 second (vs several seconds with JSON)
- [ ] Atomic writes prevent corruption on crash
- [ ] Migration from JSON to bincode works
- [ ] Search results identical before and after (data integrity)
- [ ] Tests: save/load round-trip, migration, corruption recovery

---

## Cross-Cutting Concerns

### Observability

Add structured tracing spans to `run_search` with fields for per-stage latency, cache hits, strong signal triggers, and rerank tier. This enables performance regression detection across phases.

### Security Summary (from review)

| Issue | Phase | Fix |
|-------|-------|-----|
| Cache key collision via DefaultHasher | 3 | Use RandomState or blake3 |
| Tree-sitter FFI (infinite loop, OOM) | 1 | Timeout + file size gate |
| Prompt injection via AST metadata | 1 | Cap identifier lengths at 128 chars |
| Reranker SSRF | 4 | Lock down base_url, validate localhost |
| Reranker timeout unbounded | 4 | Cap at 30s, set client-level timeout |

### Testing Strategy

Each phase should include:
1. **Unit tests** in the modified modules (colocated `#[cfg(test)]` blocks)
2. **Integration tests** in `tests/integration_test.rs` for pipeline-level behavior
3. **Manual validation** with real queries against indexed repos

---

## Dependency Graph

```
P0.1 ─► P0.2 ─► P0.3
                  │
                  ▼
         ┌── P1.1a ─► P1.1b ─► P1.1c ─► P1.1d
         │
P0.3 ──►─┼── P1.2a ─► P1.2b ─► P1.2c ─► P1.2d ─► P1.2e ─► P1.2f ─► P1.2g
         │
         └── P1.3a ─► P1.3b ─► P1.3c ─► P1.3d ─► P1.3e
                                                      │
                  ┌───────────────────────────────────┘
                  ▼
               P1.4a ─► P1.4b
                           │
                  ┌────────┘
                  ▼
P2.1a ─► P2.1b ─► P2.1c ─► P2.1d
                              │
                  ┌───────────┘
                  ▼
         P2.2a ─► P2.2b ─► P2.2c
                              │
                  ┌───────────┘
                  ▼
P2.3a ─► P2.3b ─► P2.3c ─► P2.3d
                              │
                  ┌───────────┘
                  ▼
P3.1 ─► P3.2 ─► P3.3 ─► P3.4 ─► P3.5 ─► P3.6
                                            │
                  ┌─────────────────────────┘
                  ▼
P4.1 ─► P4.2 ─► P4.3 ─► P4.4 ─► P4.5 ─► P4.6 ─► P4.7
                                                     │
                                          P5.1 ──────┘
                                            │
                                          P5.2 (deferred)
```

---

## References

### Internal
- [Search Pipeline PRD](./2026-02-01-search-improvements-prd.md)
- [Search Pipeline SPEC](./2026-02-01-search-improvements-spec.md)
- [RAG Chat Architecture](../chatbot-architecture.md)
- `src/api/search.rs` — search pipeline orchestration
- `src/search/vector.rs` — vector store (primary refactoring target)
- `src/llm/embeddings.rs` — embedding generation
- `src/api/repos.rs` — indexing pipeline

### External
- [cAST: AST-Aware Code Chunking (EMNLP 2025)](https://arxiv.org/html/2506.15655v1)
- [nomic-embed-text Model Card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [moka — Rust concurrent cache](https://github.com/moka-rs/moka)
- [BLAKE3 Rust crate](https://github.com/BLAKE3-team/BLAKE3/)
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
