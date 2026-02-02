# RAG Chat Pipeline Architecture

## Overview

The chat endpoint (`POST /api/chat`) implements a Retrieval-Augmented Generation pipeline that searches the user's indexed repositories for relevant code, then streams an LLM answer grounded in that code via Server-Sent Events.

```
User types question
        │
   POST /api/chat { message, history, repo_ids }
        │
   ┌────▼─────────────────────────────┐
   │ Step 1: Sanitize input           │
   │   - Truncate to 2000 chars       │
   │   - Strip ChatML injection tokens│
   │   - Validate history (last 10)   │
   └────┬─────────────────────────────┘
        │
   ┌────▼─────────────────────────────┐
   │ Step 2: Acquire semaphore        │
   │   - Max 3 concurrent chats       │
   └────┬─────────────────────────────┘
        │
   ┌────▼─────────────────────────────┐
   │ Step 3: Augment search query     │
   │   - Prepend repo names           │
   │   "xf what llm apis supported"   │
   └────┬─────────────────────────────┘
        │
   ┌────▼─────────────────────────────┐
   │ Step 4: Hybrid search pipeline   │
   │   4a. LLM query expansion        │
   │   4b. Multi-query BM25 + vector  │
   │   4c. RRF fusion (top 30)        │
   │   4d. LLM re-ranking → top 10    │
   └────┬─────────────────────────────┘
        │
   ┌────▼─────────────────────────────┐
   │ Step 5: Build prompt             │
   │   - Short system prompt (rules)  │
   │   - Code context in user message │
   │   - History + question           │
   └────┬─────────────────────────────┘
        │
   ┌────▼─────────────────────────────┐
   │ Steps 6-8: Stream SSE response   │
   │   event: context (sources)       │
   │   event: delta (tokens)          │
   │   event: done                    │
   └──────────────────────────────────┘
```

## Step-by-Step Detail

### Step 1: Input Sanitization

**File:** `src/api/chat.rs:27-34`

The user's message arrives as a `ChatRequest` with `message`, optional `history`, and optional `repo_ids` (which repos to search). The message is:

- Trimmed of whitespace
- Truncated to 2000 chars at a safe unicode boundary (`truncate_to_char_boundary`)
- Stripped of ChatML injection tokens (`<|im_start|>`, `<|im_end|>`) via `sanitize_for_prompt`

Conversation history goes through the same treatment: system-role messages are dropped (prevents prompt injection via history), only `user` and `assistant` roles are kept, then it retains only the last 10 turns (most recent).

### Step 2: Concurrency Control

**File:** `src/api/chat.rs:36-47`

A tokio semaphore (`chat_semaphore`, max 3 permits, configured in `src/state.rs:55`) prevents the LLM from being overwhelmed by concurrent requests. If all 3 slots are in use, the request gets a `503 Service Unavailable`. The permit is held for the entire SSE stream lifetime (see Step 8).

### Step 3: Query Augmentation

**File:** `src/api/chat.rs:49-66`

Before searching, the user's message is prepended with the names of repos being searched. If the user selected specific repos via `repo_ids`, only those names are used; otherwise all indexed repo names are included.

```
User typed:    "what llm apis are supported"
Search query:  "xf llm-tldr gastown what llm apis are supported"
```

This improves both BM25 (keyword matching on repo name) and vector search (the embedding captures the project context). The original unmodified message is preserved separately for the LLM conversation — only the search query is augmented.

**Why not brackets?** An earlier version used `[xf]` as the prefix, but Tantivy's `QueryParser` treats `[` and `]` as reserved syntax, causing parse errors. Plain space-separated names avoid this.

### Step 4: Hybrid Search Pipeline

**File:** `src/api/search.rs:13-136`

The `run_search` function is a reusable 4-stage retrieval pipeline shared by both the search endpoint and the chat endpoint. For chat, it runs with `use_bm25=true`, `use_vector=true`, `use_rerank=true`, and `limit=10`.

#### 4a. Query Expansion (`search.rs:30-51`)

The LLM generates 2-3 alternate phrasings of the search query via `expand_query`. For example:

```
Original: "xf llm-tldr gastown what llm apis are supported"
Expanded: ["LLM API integration supported models",
           "language model provider configuration"]
```

The original query gets **2x weight**; each expansion gets **1x weight**. This broadens recall — the original catches exact matches while expansions catch semantic variants.

If query expansion fails (LLM timeout, etc.), it falls back gracefully to the original query only.

#### 4b. Multi-Query BM25 + Vector Search (`search.rs:53-106`)

For **each** query variant (original + expansions), two parallel searches run:

- **BM25** (Tantivy full-text index) — keyword/term matching. Good for exact names, function signatures, error messages. Runs on a blocking thread via `spawn_blocking` since Tantivy is synchronous.
- **Vector** (embedding similarity via Ollama) — semantic matching. Embeds the query via the configured embedding model, then searches the HNSW vector store. Good for conceptual queries like "how does authentication work."

Each variant fetches up to `limit * 3 = 30` results from each index to give fusion enough candidates.

#### 4c. RRF Fusion (`search.rs:108-110`)

Reciprocal Rank Fusion merges all result sets (multiple BM25 lists + multiple vector lists, each weighted by query importance) into a single ranked list of 30 candidates.

RRF is position-based: a chunk appearing at rank 1 in multiple lists scores much higher than one appearing at rank 20 in only one list. The formula is:

```
score = Σ (weight / (k + rank))    where k = 60 (standard constant)
```

This deduplicates and normalizes across fundamentally different scoring systems (BM25 TF-IDF scores vs. cosine similarity).

#### 4d. LLM Re-ranking (`search.rs:112-125`)

The top 30 RRF candidates are sent to the LLM for relevance scoring against the **original** query. The LLM assigns a relevance score to each chunk, and results are re-sorted by a blended score that combines the RRF position with the LLM relevance judgment.

This catches cases where keyword/embedding similarity is high but actual relevance is low (e.g., a chunk mentions "LLM" many times but is about benchmarking, not API support).

The top 10 results survive (`CONTEXT_CHUNKS = 10`).

### Step 5: Prompt Assembly

**File:** `src/api/chat.rs:86-89`

Three pieces are assembled:

#### System Prompt (`build_system_prompt`, line 197)

Short behavioral rules only (~6 lines):

```
You are a code assistant. The user has cloned git repositories into this app.
Each user message includes source code retrieved from those repos.
Answer ONLY based on the provided code. Never use outside knowledge.
Never say you cannot access the code — it is included in the message.
If the snippets don't answer the question, say what you found and what's missing.
Reference file paths and line numbers. Use markdown code blocks with language tags.
```

Kept deliberately short so smaller models don't lose the instructions in noise.

#### Context Block (`build_context_block`, line 208)

The 10 code chunks formatted as plain text with `---` delimiters:

```
Here is source code from the user's repositories:

--- xf: README.md (lines 241-340) [text] ---
<actual file content>

--- xf: src/config.rs (lines 1-100) [rust] ---
<actual file content>
```

All code content is sanitized to strip ChatML tokens that might appear in source files.

#### Message Assembly (`build_messages`, line 233)

The context block is embedded **directly in the user message**, not the system message:

```
messages[0] = system: "You are a code assistant..." (short rules)
messages[1] = user: "q1"              (from history)
messages[2] = assistant: "a1"         (from history)
messages[3] = user: "Here is source code from the user's repositories:

    --- xf: README.md (lines 241-340) [text] ---
    <content>

    ...10 chunks...
    ---
    Question: what llm apis are supported"
```

**Why user-turn context?** This is the key design decision. Smaller models (llama3.2, etc.) attend much more strongly to content in user messages than to long system messages. When code context was in the system prompt, the model would ignore it and hallucinate from training data ("Google Cloud NLP, Amazon Comprehend..."). Moving it to the user turn — right before the question — forces the model to read it.

### Step 6: Sources SSE Event

**File:** `src/api/chat.rs:91-107`

Before the LLM starts streaming, a `context` SSE event is sent to the frontend with metadata about the retrieved sources (repo name, file path, line range, language). This is what the UI renders as the collapsible "Sources (10 files)" dropdown. No code content is sent — just references.

### Step 7: LLM Streaming

**File:** `src/api/chat.rs:109-118`

The assembled messages array is sent to Ollama (or any OpenAI-compatible endpoint) via `stream_chat` (`src/llm/chat_stream.rs`), which returns a `Stream<Item = Result<String>>` of token-by-token deltas.

The stream parser handles both:
- **Ollama format**: newline-delimited JSON with `message.content` field
- **OpenAI format**: SSE `data:` lines with `choices[0].delta.content` field

### Step 8: SSE Event Stream

**File:** `src/api/chat.rs:120-170`

The response is an SSE stream with three event types, always in this order:

```
event: context
data: {"sources":[{"repo_name":"xf","file_path":"README.md","start_line":241,"end_line":340,"language":"text"}]}

event: delta
data: {"content":"Based"}

event: delta
data: {"content":" on the code"}

...many more delta events...

event: done
data: {}
```

**Idle timeout:** If the LLM stops sending tokens for 30 seconds (`IDLE_TIMEOUT_SECS`), the stream emits an error event and closes. This handles hung LLM connections.

**Concurrency hold:** The semaphore permit is held for the entire stream lifetime via a `.map()` closure that captures `_permit` by reference (line 165). This ensures the 3-concurrent-chat limit applies to the full response duration, not just the initial setup.

**Error handling:** Pre-stream errors (validation, search failure) return normal HTTP error responses. Mid-stream errors (LLM failure, timeout) emit an SSE `error` event since HTTP headers are already sent.

## Conversation State

- **Stateless server:** No sessions. The client sends the full conversation history with each request.
- **History in `chatHistory` array:** Maintained client-side in the browser, sent as part of `ChatRequest`.
- **Capped at 10 turns:** To bound request size and context window usage.
- **Persisted via `sessionStorage`:** Survives page navigation within the tab but not tab close.

## Security Measures

| Threat | Mitigation |
|--------|-----------|
| Prompt injection via user message | `sanitize_for_prompt` strips ChatML tokens (`<\|im_start\|>`, `<\|im_end\|>`) |
| Prompt injection via history | System-role messages dropped; all content sanitized |
| Prompt injection via code content | Code chunks sanitized before embedding in prompt |
| XSS from LLM output | Frontend passes all rendered HTML through `DOMPurify.sanitize()` |
| Resource exhaustion | Semaphore caps concurrent streams at 3; idle timeout at 30s; message length cap at 2000 chars |

## Key Files

| File | Purpose |
|------|---------|
| `src/api/chat.rs` | Chat endpoint, prompt assembly, SSE streaming |
| `src/api/search.rs` | Reusable hybrid search pipeline (shared by search + chat) |
| `src/llm/chat_stream.rs` | Streaming LLM client (Ollama + OpenAI) |
| `src/llm/query_expand.rs` | LLM-based query expansion + `sanitize_for_prompt` |
| `src/llm/rerank.rs` | LLM-based result re-ranking |
| `src/llm/embeddings.rs` | Embedding generation for vector search |
| `src/search/bm25.rs` | Tantivy BM25 full-text index |
| `src/search/vector.rs` | HNSW vector store |
| `src/search/hybrid.rs` | RRF fusion algorithm |
| `src/state.rs` | Shared app state (repos, indexes, semaphores) |
| `static/index.html` | Frontend with Chat UI, SSE reader, markdown rendering |

## Configuration

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `CONTEXT_CHUNKS` | 10 | `chat.rs:19` | Number of code chunks to retrieve |
| `MAX_CHAT_MESSAGE_LEN` | 2000 | `chat.rs:17` | Max user message length (chars) |
| `MAX_HISTORY_TURNS` | 10 | `chat.rs:18` | Max conversation history turns |
| `IDLE_TIMEOUT_SECS` | 30 | `chat.rs:20` | Seconds before stream timeout |
| Chat concurrency | 3 | `state.rs:55` | Max concurrent chat streams |
