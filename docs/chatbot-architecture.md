# Plan: RAG Chatbot View

## Goal
Add a Chat tab that lets users ask questions about their repos. The backend searches for relevant code using the existing hybrid search pipeline, injects it as context into an LLM prompt, and streams the response via SSE.

## Architecture Overview

```
User types question
        |
   POST /api/chat (with message + history + repo_ids)
        |
   [1. Hybrid search: BM25 + vector, no reranking]
        |
   [2. Build prompt: system + code context + history + question]
        |
   [3. Stream LLM response via SSE]
        |
   Frontend renders streamed markdown with syntax highlighting
```

## Key Decisions

### Streaming: Server-Sent Events (SSE)
- Axum 0.8 has built-in `axum::response::sse::Sse`
- Ollama `/api/chat` supports `stream: true` (newline-delimited JSON)
- OpenAI `/v1/chat/completions` supports `stream: true` (SSE format)
- Frontend uses `fetch()` + `ReadableStream` reader (not `EventSource`, which is GET-only)

### Frontend libraries: marked.js + highlight.js + DOMPurify (vendored inline)
- LLM responses contain markdown with code blocks — rendering as plain text is unusable
- marked.min.js (~50KB), highlight.min.js (~40KB), and purify.min.js (~20KB) vendored directly in `index.html` inside `<script>` tags
- DOMPurify sanitizes all `marked.parse()` output to prevent XSS from LLM-generated content grounded in adversarial repo code (e.g. `<script>`, `<img onerror>` injected via file contents)
- Keeps the app fully self-contained with zero external dependencies (matching existing pattern)
- highlight.js github-dark theme CSS also vendored inline in `<style>`

### Conversation state: Client-side
- Server stays stateless — no sessions
- `chatHistory` array sent with each request
- Capped at last 10 turns to bound request size
- No persistence across page reloads (acceptable for a dev tool)

### Search integration: Reuse existing pipeline
- Extract core search logic from `src/api/search.rs` into a reusable function
- Chat uses BM25 + vector search (skip reranking to reduce latency)
- Default 5 context chunks (~500 lines of code)

## SSE Event Protocol

```
event: context
data: {"sources":[{"repo_name":"repo","file_path":"src/main.rs","start_line":1,"end_line":50,"language":"rust"}]}

event: delta
data: {"content":"The"}

event: delta
data: {"content":" main function"}

event: done
data: {}

event: error
data: {"message":"LLM connection failed"}
```

`context` fires first (before any tokens) so the UI can show sources immediately. `delta` events stream individual tokens. `done` signals completion.

## Changes

### 1. `Cargo.toml` — New dependencies
- Add `tokio-stream = "0.1"` (for `StreamExt` combinators)
- Add `futures-util = "0.3"` (for stream mapping)
- Add `stream` feature to reqwest: `features = ["json", "stream"]`

### 2. `src/models.rs` — Chat data types
- `ChatRequest { message, history, repo_ids, context_chunks }`
- `ChatMessage { role, content }` (reused in request and prompt building)
- `ContextSnippet { repo_name, file_path, start_line, end_line, language }` (for the `context` SSE event)

### 3. `src/api/search.rs` → Extract reusable search function
- Extract the core search pipeline (steps 1-3: query expansion, BM25+vector, RRF fusion) into a standalone `pub async fn run_search()` **within `src/api/search.rs`** (not `search/hybrid.rs` — that module is pure synchronous RRF fusion with zero async/HTTP/LLM deps, and must stay that way)
- The existing `search()` handler becomes a thin wrapper calling `run_search()`
- The chat handler imports `run_search()` from the `api` layer: `use crate::api::search::run_search`

### 4. `src/llm/chat_stream.rs` — New file: streaming LLM client
- `pub fn stream_chat(client, config, messages) -> impl Stream<Item = Result<String>>`
- Calls Ollama or OpenAI with `stream: true`
- Parses newline-delimited JSON (Ollama) or SSE `data:` lines (OpenAI)
- Extracts content deltas from each chunk

### 5. `src/api/chat.rs` — New file: chat handler
- `pub async fn chat()` — the SSE endpoint
  1. Validates input:
     - Enforce `MAX_CHAT_MESSAGE_LEN` (reuse `MAX_QUERY_LEN = 500` pattern from `query_expand.rs`)
     - Enforce request body size cap (e.g. 64KB via Axum `DefaultBodyLimit`)
     - Reject any history entries with `role: "system"` — only `"user"` and `"assistant"` allowed
     - Apply `sanitize_for_prompt()` to `message` and every `history[].content` (strips ChatML/Llama control tokens, consistent with existing `query_expand.rs` and `rerank.rs` usage)
     - Cap history at last 10 turns
  2. Acquires `chat_semaphore` permit (3–5 concurrent streams, prevents resource exhaustion from hung LLM connections)
  3. Calls `run_search()` to get top 5 code chunks
  4. Builds system prompt with code context + conversation history (also sanitize code context content)
  5. Emits `context` SSE event with source metadata
  6. Calls `stream_chat()` and maps each token to a `delta` SSE event
  7. Emits `done` event on completion, `error` on failure
- Pre-stream validation errors (steps 1–3) return `(StatusCode, String)` as normal HTTP errors
- Mid-stream errors (steps 5–7) emit SSE `error` event since headers are already sent

### 6. `src/state.rs` — Add concurrency semaphore
- Add `chat_semaphore: Arc<tokio::sync::Semaphore>` to `AppState` (3–5 permits) to cap concurrent LLM streams
- Reuse the existing `http_client` for chat streaming — use per-request `.timeout(Duration::from_secs(300))` to override the client's default 120s total timeout (no separate `chat_client` needed)
- Apply 30s idle timeout per SSE chunk read (via `tokio::time::timeout` around each `stream.next()`) to detect hung connections

### 7. `src/main.rs` + `src/api/mod.rs` + `src/llm/mod.rs` — Wiring
- Add `POST /api/chat` route
- Add `pub mod chat;` declarations

### 8. `static/index.html` — Chat UI
- Vendor marked.min.js + highlight.min.js inline in `<script>` tags (download from npm/CDN once, paste minified source)
- Add Chat nav button and `#page-chat` section
- Chat UI: scrollable message list + input area at bottom
- `sendChatMessage()`: POST to `/api/chat`, read SSE stream via `fetch()` + `ReadableStream`
- Render assistant messages with `DOMPurify.sanitize(marked.parse(...))` + `hljs.highlightElement()`
- Debounce re-renders during streaming (requestAnimationFrame) to avoid jank
- Collapsible "Sources" panel below each assistant message showing file paths
- "New Chat" button to clear history

### CSS additions
- `.chat-container` — full-height flex column layout
- `.chat-msg-user` — right-aligned blue bubble
- `.chat-msg-assistant` — left-aligned with markdown/code styling
- `.chat-sources` — collapsible source references
- `.chat-input-area` — textarea + send button at bottom

## System Prompt Template

```
You are a code assistant that answers questions about the user's repositories.
You have been given relevant code snippets as context.
Ground your answers in the provided code. If the context doesn't contain
enough information, say so rather than guessing.
When referencing code, mention the file path and line numbers.
Use markdown code blocks with language identifiers.

## Code Context

### {repo_name}: {file_path} (lines {start}-{end}) [{language}]
```{language}
{content}
```
...
```

Context budget: 5 chunks x ~100 lines x ~1500 chars = ~7500 chars (~2500 tokens), leaving room for history and response.

## Files modified
- `Cargo.toml`
- `src/models.rs`
- `src/api/search.rs` (refactor to extract reusable `run_search()` fn, kept in this file)
- `src/llm/chat_stream.rs` (new)
- `src/api/chat.rs` (new)
- `src/state.rs` (add `chat_semaphore`)
- `src/main.rs`
- `src/api/mod.rs`
- `src/llm/mod.rs`
- `static/index.html`

## Verification
- `cargo test` — all existing tests still pass
- Unit tests for: Ollama/OpenAI stream parsing, prompt construction, history truncation
- Unit tests for: chat history sanitization (ChatML stripping, role rejection, length limits)
- Unit tests for: SSE event serialization (exact wire format)
- Manual: start server with Ollama running, open Chat tab, ask "how does the search pipeline work?" — should see streamed markdown response with code references
- Manual: verify Sources panel shows the files the LLM grounded its answer in
- Manual: verify DOMPurify strips `<script>` / `<img onerror>` from LLM output
