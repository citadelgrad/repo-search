use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

/// A stored vector entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorEntry {
    repo_id: Uuid,
    repo_name: String,
    file_path: String,
    chunk_index: usize,
    content: String,
    language: String,
    start_line: usize,
    end_line: usize,
    embedding: Vec<f32>,
}

/// In-memory vector store with disk persistence and cosine similarity search.
pub struct VectorStore {
    entries: RwLock<Vec<VectorEntry>>,
    persist_path: std::path::PathBuf,
}

#[derive(Debug, Clone)]
pub struct VectorHit {
    pub repo_id: Uuid,
    pub repo_name: String,
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub language: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f32,
}

impl VectorStore {
    pub fn open_or_create(vector_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(vector_dir)?;
        let persist_path = vector_dir.join("vectors.json");

        let entries = if persist_path.exists() {
            let data = std::fs::read_to_string(&persist_path)
                .context("Failed to read vector store")?;
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(Self {
            entries: RwLock::new(entries),
            persist_path,
        })
    }

    /// Add vectors for file chunks. `embeddings` must be parallel with `chunks`.
    pub fn add_chunks(
        &self,
        repo_id: Uuid,
        repo_name: &str,
        chunks: &[(String, usize, String, String, usize, usize)], // (file_path, chunk_index, content, language, start_line, end_line)
        embeddings: Vec<Vec<f32>>,
    ) -> Result<()> {
        let mut entries = self.entries.write();

        for (i, (file_path, chunk_index, content, language, start_line, end_line)) in
            chunks.iter().enumerate()
        {
            if let Some(embedding) = embeddings.get(i) {
                entries.push(VectorEntry {
                    repo_id,
                    repo_name: repo_name.to_string(),
                    file_path: file_path.clone(),
                    chunk_index: *chunk_index,
                    content: content.clone(),
                    language: language.clone(),
                    start_line: *start_line,
                    end_line: *end_line,
                    embedding: embedding.clone(),
                });
            }
        }

        // Persist to disk
        let data = serde_json::to_string(&*entries)?;
        std::fs::write(&self.persist_path, data)?;

        Ok(())
    }

    /// Delete all vectors for a repo.
    pub fn delete_repo(&self, repo_id: &Uuid) -> Result<()> {
        let mut entries = self.entries.write();
        entries.retain(|e| &e.repo_id != repo_id);

        let data = serde_json::to_string(&*entries)?;
        std::fs::write(&self.persist_path, data)?;
        Ok(())
    }

    /// Search by cosine similarity against a query embedding.
    pub fn search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        repo_ids: Option<&[Uuid]>,
    ) -> Vec<VectorHit> {
        let entries = self.entries.read();

        let mut scored: Vec<(f32, &VectorEntry)> = entries
            .iter()
            .filter(|e| {
                if let Some(ids) = repo_ids {
                    ids.contains(&e.repo_id)
                } else {
                    true
                }
            })
            .map(|e| (cosine_similarity(query_embedding, &e.embedding), e))
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        scored
            .into_iter()
            .map(|(score, e)| VectorHit {
                repo_id: e.repo_id,
                repo_name: e.repo_name.clone(),
                file_path: e.file_path.clone(),
                chunk_index: e.chunk_index,
                content: e.content.clone(),
                language: e.language.clone(),
                start_line: e.start_line,
                end_line: e.end_line,
                score,
            })
            .collect()
    }

    pub fn entry_count(&self) -> usize {
        self.entries.read().len()
    }

    /// Get counts grouped by repo_id.
    pub fn repo_counts(&self) -> HashMap<Uuid, usize> {
        let entries = self.entries.read();
        let mut counts = HashMap::new();
        for e in entries.iter() {
            *counts.entry(e.repo_id).or_insert(0) += 1;
        }
        counts
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}
