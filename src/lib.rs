//! # repo-search
//!
//! A Rust web application for cloning git repositories and searching them
//! with a hybrid pipeline combining BM25 full-text search, vector semantic
//! search, and LLM re-ranking.
//!
//! ## Architecture
//!
//! The search pipeline is a directed acyclic graph (DAG):
//!
//! ```text
//!                          ┌─────────────┐
//!                          │  User Query  │
//!                          └──────┬───────┘
//!                                 │
//!                    ┌────────────┴────────────┐
//!                    ▼                         ▼
//!           ┌────────────────┐       ┌─────────────────┐
//!           │ Query Expansion│       │  Original Query  │
//!           │  (LLM: 2 alt)  │       │   (×2 weight)   │
//!           └───────┬────────┘       └────────┬────────┘
//!                   │ 2 variants              │
//!                   └────────────┬────────────┘
//!                                │ 3 queries total
//!            ┌───────────────────┼───────────────────┐
//!            ▼                   ▼                   ▼
//!     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
//!     │ Original Q   │    │ Expanded Q1 │    │ Expanded Q2 │
//!     │ BM25+Vector  │    │ BM25+Vector │    │ BM25+Vector │
//!     └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
//!            │                  │                  │
//!            └──────────────────┼──────────────────┘
//!                               │ 6 ranked lists
//!                               ▼
//!                  ┌───────────────────────┐
//!                  │  RRF Fusion + Bonus   │
//!                  │  Original: ×2 weight  │
//!                  │  Top-rank: +0.05      │
//!                  │  Keep top 30          │
//!                  └───────────┬───────────┘
//!                              │
//!                              ▼
//!                  ┌───────────────────────┐
//!                  │   LLM Re-ranking      │
//!                  │  yes/no + confidence   │
//!                  │  per-document scoring  │
//!                  └───────────┬───────────┘
//!                              │
//!                              ▼
//!                  ┌───────────────────────┐
//!                  │ Position-Aware Blend  │
//!                  │  Top 1-3:  75% RRF    │
//!                  │  Top 4-10: 60% RRF    │
//!                  │  Top 11+:  40% RRF    │
//!                  └───────────┬───────────┘
//!                              │
//!                              ▼
//!                  ┌───────────────────────┐
//!                  │   Final Results       │
//!                  └───────────────────────┘
//! ```
//!
//! ## Module Overview
//!
//! - [`config`] - Environment-based configuration for server, data dirs, and LLM settings
//! - [`models`] - Shared data types: `Repo`, `FileChunk`, `SearchHit`, request/response types
//! - [`git`] - Git clone operations and file walking/language detection (60+ languages)
//! - [`search::bm25`] - BM25 full-text index powered by tantivy
//! - [`search::vector`] - In-memory vector store with cosine similarity and disk persistence
//! - [`search::hybrid`] - Multi-query Reciprocal Rank Fusion with weighted scoring
//! - [`llm::embeddings`] - Batch embedding generation via Ollama or OpenAI-compatible APIs
//! - [`llm::query_expand`] - LLM-powered query expansion (generates 2 alternative phrasings)
//! - [`llm::rerank`] - Per-document yes/no relevance re-ranking with position-aware blending
//! - [`api`] - Axum HTTP handlers for repo CRUD, search, and config management
//! - [`state`] - Shared application state holding indexes, config, and persistence

pub mod api;
pub mod chunking;
pub mod config;
pub mod git;
pub mod llm;
pub mod models;
pub mod search;
pub mod state;
