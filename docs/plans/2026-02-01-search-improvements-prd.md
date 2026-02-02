# Search Pipeline Improvements: Product Requirements Document

**Date:** 2026-02-01
**Status:** Draft
**Scope:** 8 improvements to search quality, latency, and scalability

---

## Executive Summary

Analysis of [tobi/qmd](https://github.com/tobi/qmd) against our implementation revealed 8 concrete improvements. The architecture review recommends grouping these into 5 phases to minimize re-index passes and manage operational complexity. The combined effect reduces typical search latency from 1.5-5s to 0.4-1.6s (cold) and <10ms (warm cache or strong signal), while substantially improving retrieval quality.

### Phased Delivery

| Phase | Items | Theme | Re-index? |
|-------|-------|-------|-----------|
| 1 | #1 + #2 + #6 | Quality + Re-index (single pass) | Yes (once) |
| 2 | #4 + #3 | Query pipeline hardening | No |
| 3 | #7 | Pipeline caching | No |
| 4 | #5 | Dedicated reranker | No |
| 5 | #8 | Vector index scalability | No |

---

## PRD-1: Embedding Task Prefixes

### Problem

nomic-embed-text was trained with asymmetric task prefixes (`search_query:` for queries, `search_document:` for documents). Our system embeds both without any prefix, degrading the embedding space geometry and reducing retrieval precision. The model card states: "The text prompt must include a task instruction prefix."

### Requirements

- **R1.1:** Prepend `search_document: ` to all document texts before embedding at index time.
- **R1.2:** Prepend `search_query: ` to all query texts before embedding at search time.
- **R1.3:** Make prefix selection model-aware. Support nomic (`search_query:`/`search_document:`), E5 (`query:`/`passage:`), and no-prefix models. Default behavior should be derived from the `embedding_model` config value.
- **R1.4:** Apply prefix before truncation so the prefix is never truncated away.
- **R1.5:** Adding prefixes invalidates all existing embeddings. Trigger a full re-embed of all indexed repos on upgrade.
- **R1.6:** Store an `embedding_config_hash` (model name + prefix convention + chunking version) in vector store metadata. On startup, detect mismatches and warn/trigger re-index.

### Success Metrics

- Measurable improvement in top-5 precision on manual test queries.
- Zero retrieval quality regression when switching to a non-prefix model.

### Risks

- Forgetting the space after the colon (`search_document:text` vs `search_document: text`) silently degrades quality.
- If a user switches embedding models without re-indexing, old embeddings become incompatible. The config hash (R1.6) mitigates this.

### Sources

- [Nomic Embed Paper (arXiv 2402.01613)](https://arxiv.org/html/2402.01613v2)
- [nomic-embed-text HuggingFace Model Card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [E5 Paper](https://arxiv.org/html/2212.03533v2)

---

## PRD-2: AST-Aware Code Chunking

### Problem

Fixed 100-line chunks with 20-line overlap split functions, classes, and structs mid-body. This produces chunks that are semantically incomplete, degrading both embedding quality and the LLM's ability to reason about code in the RAG pipeline. Academic research (cAST, EMNLP 2025) shows AST-aware chunking improves retrieval precision by +1.2 to +4.3 points and code generation quality by +5.5 points.

### Requirements

- **R2.1:** Parse source files with tree-sitter and chunk at syntactic boundaries (functions, classes, modules, top-level declarations).
- **R2.2:** Support at least: Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, Ruby, Bash, JSON, YAML, TOML, HTML, CSS, Markdown.
- **R2.3:** When a single AST node exceeds the chunk size budget, recurse into its children (split-then-merge algorithm from cAST paper).
- **R2.4:** Chunk size budget: 1000-2000 non-whitespace characters (approximately matching current 100-line chunks in token count).
- **R2.5:** Fallback to line-based chunking (with blank-line-aware boundaries) when no tree-sitter grammar is available or parsing fails.
- **R2.6:** Enrich each chunk with metadata: file path, parent scope chain (e.g., `AuthService > validate_token`), entity type (function, class, module), and relevant imports.
- **R2.7:** Use the enriched metadata in the embedding text (prepend scope and entity info to raw code before embedding).
- **R2.8:** Remove the 20-line overlap. AST boundaries + metadata enrichment make overlap unnecessary.
- **R2.9:** This change invalidates all existing chunks and embeddings. Combine with Phase 1 re-index.

### Success Metrics

- Zero functions split mid-body in tree-sitter-supported languages.
- Measurable improvement in retrieval recall for queries targeting specific functions.
- Indexing throughput within 2x of current speed (tree-sitter parsing adds <10ms per file).

### Risks

- Tree-sitter adds native C dependencies and per-language grammar crates. Increases compile time and binary size.
- Grammar version mismatches between tree-sitter core and grammar crates can cause build failures.
- Very large auto-generated files may produce massive AST nodes that exceed the chunk budget even after recursion.

### Sources

- [cAST: Enhancing Code RAG with Structural Chunking (EMNLP 2025)](https://arxiv.org/html/2506.15655v1)
- [Supermemory: Building code-chunk](https://supermemory.ai/blog/building-code-chunk-ast-aware-code-chunking/)
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) -- adding context to chunks reduced retrieval failures by 49%
- [code-splitter Rust crate](https://lib.rs/crates/code-splitter)
- [text-splitter Rust crate](https://github.com/benbrandt/text-splitter)

---

## PRD-3: Strong Signal Detection

### Problem

Every search query runs the full 4-stage pipeline (expand, embed, search, rerank) even when BM25 alone produces a high-confidence result. For exact-match queries (function names, error messages, file paths), the expansion and reranking stages add 1.5-4 seconds of latency without improving results. QMD implements this optimization and skips expansion when the top BM25 score >= 0.85 with a >= 0.15 gap to #2.

### Requirements

- **R3.1:** After the initial BM25 search (original query only), evaluate the score distribution of top results.
- **R3.2:** If strong signal is detected (configurable thresholds), skip query expansion, additional embedding calls, and LLM reranking. Return BM25 results directly.
- **R3.3:** Two-stage gating:
  - Stage 1 (pre-expansion): BM25-only check. If triggered, skip everything.
  - Stage 2 (post-fusion, pre-rerank): If BM25 and vector agree on the top result AND the RRF score gap is large, skip reranking only.
- **R3.4:** Never skip for multi-part queries (heuristic: contains "and" between question clauses).
- **R3.5:** Log when signal detection triggers for observability.
- **R3.6:** Default thresholds should be conservative (minimize false positives). Expose as config if needed.

### Success Metrics

- 20-40% of queries bypass expansion+reranking.
- Bypassed queries return in <50ms (vs 1.5-5s currently).
- No measurable recall degradation on test query set.

### Dependencies

- Requires Item #4 first. If expanded queries can drift off-topic, signal detection might incorrectly trust results from drifted expansions.
- Requires BM25 score normalization to 0-1 range for portable thresholds (Tantivy raw scores are unbounded TF-IDF).

### Sources

- QMD `querySearch()` implementation
- [TARG: Training-Free Adaptive Retrieval Gating (arXiv)](https://arxiv.org/html/2511.09803v1)
- [Vespa: Redefining Hybrid Search](https://blog.vespa.ai/redefining-hybrid-search-possibilities-with-vespa/)

---

## PRD-4: Query Expansion Filtering

### Problem

LLM-generated query expansions can drift off-topic, introducing noise that dilutes good RRF results. Currently, expansions are accepted without validation. QMD implements term-overlap filtering and penalizes echoes, generic phrases, and low-diversity expansions.

### Requirements

- **R4.1:** After `expand_query()` returns, filter out any expansion that shares zero non-stopword terms with the original query.
- **R4.2:** Reject echo expansions (expansions that are identical to or trivially extend the original query).
- **R4.3:** Reject expansions containing generic filler phrases ("find information about", "search for", "how to").
- **R4.4:** Log filtered expansions with reasons for observability.
- **R4.5:** If all expansions are filtered, continue with original query only (never fail the search).

### Success Metrics

- Eliminates off-topic expansions (measured by manual inspection of expansion logs).
- Zero net latency cost (filtering is string comparison).
- No reduction in beneficial expansion rate for semantic/conceptual queries.

### Sources

- QMD `reward.py` expansion scoring
- [LLM-based QE Failure Analysis (SIGIR 2025)](https://arxiv.org/html/2505.12694v1) -- expansion hurts for unfamiliar queries

---

## PRD-5: Dedicated Cross-Encoder Reranker

### Problem

Reranking is the dominant latency bottleneck: 30 individual LLM chat calls (each a full generative inference pass) take 1-3 seconds. A dedicated 0.6B cross-encoder model processes all 30 pairs in a single forward pass (~50-100ms), with better calibrated relevance scores. Qwen3-Reranker-0.6B scores 73.42 on MTEB-Code vs our chat-model-based approach which produces inconsistent JSON requiring 3 fallback parsing paths.

### Requirements

- **R5.1:** Support a dedicated reranker model endpoint (e.g., llama-server with `--pooling rank`, vLLM, or HuggingFace TEI).
- **R5.2:** Add `reranker_base_url` and `reranker_model` to configuration (optional; when absent, fall back to existing LLM-based reranking).
- **R5.3:** Call the reranker with a single batch request containing all candidate documents.
- **R5.4:** Normalize cross-encoder raw scores (logits) to 0-1 range via sigmoid before blending with RRF scores.
- **R5.5:** Maintain the existing position-aware blending strategy (75/25 at top-3, 60/40 at 4-10, 40/60 at 11+).
- **R5.6:** Tiered fallback: cross-encoder -> LLM-based reranking -> RRF-only (no reranking).
- **R5.7:** Circuit breaker: after N consecutive failures, stop attempting the reranker for a recovery period.
- **R5.8:** Include `rerank_tier` ("cross-encoder", "llm", "rrf-only") in the search response for frontend/observability.

### Success Metrics

- Reranking latency drops from 1-3s to 50-100ms (10-30x speedup).
- Search P95 latency under 1 second (currently 3-5s).
- Better code retrieval quality (Qwen3-Reranker-0.6B: 73.42 MTEB-Code vs BGE-reranker-v2-m3: 41.38).

### Risks

- Adds a second LLM process (llama-server) alongside Ollama. No process supervision or health checking exists today.
- Memory: 0.6B model requires ~500MB-1.2GB RAM/VRAM.
- Ollama does not have a native `/api/rerank` endpoint (PR #7219 was closed). Must use llama-server, TEI, or vLLM.

### Sources

- [Qwen3-Reranker-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Cross-Encoders vs LLMs for Reranking (arXiv:2403.10407)](https://arxiv.org/abs/2403.10407)
- [ZeroEntropy: Should You Use LLMs for Reranking?](https://www.zeroentropy.dev/articles/should-you-use-llms-for-reranking-a-deep-dive-into-pointwise-listwise-and-cross-encoders)

---

## PRD-6: Content-Hash Deduplication for Embeddings

### Problem

Identical code chunks across repositories (LICENSE, .gitignore, vendored code, Cargo.lock) are embedded independently, wasting API calls and storage. QMD uses content-addressable storage where identical content shares a single embedding regardless of how many repos contain it.

### Requirements

- **R6.1:** Compute a BLAKE3 hash of each chunk's content before embedding.
- **R6.2:** Before calling the embedding API, check if the hash already has a cached embedding. Skip the API call if found.
- **R6.3:** Store embeddings keyed by content hash, with a separate location table mapping (repo_id, file_path, chunk_index) to content hashes.
- **R6.4:** On incremental re-index (after git pull), compute file-level hashes first. Skip unchanged files entirely. For changed files, re-chunk and diff at chunk level -- only re-embed new chunks.
- **R6.5:** Store file-level hashes alongside repo metadata for fast change detection.
- **R6.6:** Store `embedding_config_hash` (model + prefix + chunking version) so stale embeddings are detected on startup.

### Success Metrics

- 5-15% reduction in embedding API calls during re-indexing (cross-repo dedup).
- 80-95% reduction in embedding API calls during incremental re-index after small git pulls.
- Startup validation catches model/config changes and prevents stale embedding use.

### Sources

- [BLAKE3 Rust crate](https://github.com/BLAKE3-team/BLAKE3/)
- QMD `content` table schema
- [Azure RAG Architecture: Enrichment Phase](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-enrichment-phase)

---

## PRD-7: Pipeline Caching

### Problem

Every search query makes live LLM calls for expansion, embedding, and reranking -- even for repeated or similar queries. There is no caching at any stage. For a chatbot where users refine questions iteratively, this means 5-20 seconds per message even when the underlying search results would be identical.

### Requirements

- **R7.1:** Cache query expansion results: `hash(model + query) -> Vec<String>`. TTL: 1 hour.
- **R7.2:** Cache query embeddings: `hash(model + text) -> Vec<f32>`. TTL: 4 hours.
- **R7.3:** Cache reranker scores: `hash(model + query + chunk_hash) -> f32`. TTL: 2 hours.
- **R7.4:** Use moka async cache (TinyLFU eviction, built-in TTL).
- **R7.5:** Include model name in all cache keys so model changes auto-invalidate.
- **R7.6:** Flush rerank cache on repo sync/re-index (content changed). Keep expansion and embedding caches (query-only, not content-dependent).
- **R7.7:** Flush all caches when LLM config changes (model swap).
- **R7.8:** Memory budget: ~50MB total across all caches. Use moka's weigher for the embedding cache.
- **R7.9:** Record and expose cache hit rates for observability.

### Success Metrics

- 30-60% cache hit rate for repeated/similar queries in a session.
- Cache-hit searches return in <10ms (vs 1.5-5s).
- Memory usage stays within budget under sustained load.

### Sources

- [moka -- Rust concurrent cache](https://github.com/moka-rs/moka)
- [LLM Caching Best Practices](https://wangyeux.medium.com/llm-caching-best-practices-from-exact-keys-to-semantic-conversation-matching-8b06b177a947)

---

## PRD-8: HNSW Vector Index

### Problem

Vector search uses brute-force cosine similarity over all entries. At 100K vectors (768-dim), this takes ~50ms per query variant (150ms for 3 variants). At the configured 500K cap, it would take ~750ms per variant (2.25s total). QMD uses sqlite-vec; the industry standard for this scale is HNSW.

### Requirements

- **R8.1:** Replace brute-force cosine scan with an HNSW approximate nearest neighbor index.
- **R8.2:** Use the `usearch` crate (HNSW with SimSIMD acceleration, built-in quantization, single-file persistence).
- **R8.3:** Dynamic mode: use brute-force for <10K vectors (where it's actually faster due to zero overhead), automatically switch to HNSW above that threshold.
- **R8.4:** Support BFloat16 quantization (2x memory reduction, negligible recall loss).
- **R8.5:** Replace JSON persistence with usearch's native binary format.
- **R8.6:** On first startup after upgrade, backfill existing vectors into the HNSW index from the legacy JSON store.
- **R8.7:** Maintain the existing `VectorStore` public API (`search`, `add_chunks`, `delete_repo`) so search.rs and repos.rs do not need to change.

### Success Metrics

- Vector search latency: <5ms at 500K vectors (vs ~750ms brute-force).
- Recall >= 95% (HNSW is approximate; verify against exact search).
- Startup time with 500K vectors: <2 seconds (vs potentially 30+ seconds parsing a 1.5GB JSON file).
- Memory reduction of 50% with BFloat16 quantization.

### Scaling Thresholds

| Vector Count | Brute-Force | HNSW (projected) |
|-------------|-------------|-------------------|
| 10K | ~5ms | ~1ms |
| 100K | ~50ms | ~2ms |
| 500K | ~750ms | ~5ms |

### Sources

- [USearch Rust crate](https://lib.rs/crates/usearch)
- [hnsw_rs pure Rust HNSW](https://docs.rs/hnsw_rs/latest/hnsw_rs/)
- [sqlite-vec benchmarks](https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html)
- [Qdrant HNSW benchmarks](https://qdrant.tech/benchmarks/)

---

## Review Notes

*See [SPEC Review Findings](./2026-02-01-search-improvements-spec.md#review-findings--revisions) for the full multi-agent review (architecture, performance, security, data integrity, simplicity).*

### Key Decisions After Review

1. **Defer HNSW (PRD-8) until corpus exceeds 100K vectors.** At typical scale (5K-50K), brute-force is <25ms. Replace JSON persistence with bincode for faster startup. This removes the usearch dependency and its deletion/migration complexities.

2. **Recalibrate BM25 normalization (PRD-3).** The k=10 constant makes the 0.85 threshold unreachable for typical code search scores. Must collect empirical score data after Phase 1 before finalizing.

3. **Simplify reranker (PRD-5) to two tiers.** Cross-encoder → RRF-only. Drop the LLM fallback -- it's the bottleneck being replaced. Remove circuit breaker (localhost single-user tool).

4. **Trim tree-sitter grammars (PRD-2) to 6.** Rust, Python, JS, TS/TSX, Go. Fallback line-based chunker handles the rest. Add more grammars on demand.

5. **Phase 2 internal ordering is mandatory.** Implement expansion filtering (PRD-4) first, validate, then implement signal detection (PRD-3) on top. Signal detection relies on expansion quality.

### Revised Phase Plan

| Phase | Items | Theme | Dependencies |
|-------|-------|-------|-------------|
| 0 | Refactor `VectorStore::add_chunks` to accept a struct | API cleanup | None |
| 1 | #1 + #2 + #6 | Quality + Re-index | Phase 0 |
| 2 | #4 then #3 | Query hardening | Phase 1 (calibration data) |
| 3 | #7 | Caching | Phase 2 (pipeline stable) |
| 4 | #5 | Cross-encoder reranker | Phase 3 (cache keys need reranker model) |
| 5 | #8 | HNSW (deferred until >100K vectors) | When needed |
