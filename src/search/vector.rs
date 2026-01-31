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
    max_entries: usize,
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
        Self::open_or_create_with_limit(vector_dir, 0)
    }

    pub fn open_or_create_with_limit(vector_dir: &Path, max_entries: usize) -> Result<Self> {
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
            max_entries,
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

        // Enforce memory cap
        if self.max_entries > 0 && entries.len() + chunks.len() > self.max_entries {
            anyhow::bail!(
                "Vector store limit exceeded: {} existing + {} new > {} max",
                entries.len(),
                chunks.len(),
                self.max_entries
            );
        }

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

/// Compute cosine similarity between two vectors.
/// Returns 0.0 for mismatched lengths, empty vectors, or zero-norm vectors.
pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    // ── cosine_similarity tests ──────────────────────────

    #[test]
    fn test_cosine_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let score = cosine_similarity(&a, &a);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let score = cosine_similarity(&a, &b);
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_empty_vectors() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_known_value() {
        // cos([1,0], [1,1]) = 1 / (1 * sqrt(2)) = 0.7071...
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        let score = cosine_similarity(&a, &b);
        assert!((score - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    // ── VectorStore tests ────────────────────────────────

    fn make_chunks(n: usize) -> Vec<(String, usize, String, String, usize, usize)> {
        (0..n)
            .map(|i| {
                (
                    format!("file_{i}.rs"),
                    i,
                    format!("fn func_{i}() {{}}"),
                    "rust".to_string(),
                    i * 100 + 1,
                    (i + 1) * 100,
                )
            })
            .collect()
    }

    fn make_embeddings(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                // Each embedding differs in direction
                v[i % dim] = 1.0;
                v
            })
            .collect()
    }

    #[test]
    fn test_vector_store_create_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open_or_create(dir.path()).unwrap();
        assert_eq!(store.entry_count(), 0);
    }

    #[test]
    fn test_vector_store_add_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open_or_create(dir.path()).unwrap();
        let repo_id = Uuid::new_v4();

        let chunks = make_chunks(3);
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        store
            .add_chunks(repo_id, "test-repo", &chunks, embeddings)
            .unwrap();
        assert_eq!(store.entry_count(), 3);

        // Search for vector closest to [1, 0, 0]
        let results = store.search(&[1.0, 0.0, 0.0], 10, None);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].file_path, "file_0.rs");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_store_search_with_limit() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open_or_create(dir.path()).unwrap();
        let repo_id = Uuid::new_v4();

        let chunks = make_chunks(5);
        let embeddings = make_embeddings(5, 5);

        store
            .add_chunks(repo_id, "test-repo", &chunks, embeddings)
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0, 0.0, 0.0], 2, None);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_vector_store_delete_repo() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open_or_create(dir.path()).unwrap();
        let repo1 = Uuid::new_v4();
        let repo2 = Uuid::new_v4();

        store
            .add_chunks(
                repo1,
                "repo1",
                &make_chunks(2),
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            )
            .unwrap();
        store
            .add_chunks(
                repo2,
                "repo2",
                &make_chunks(2),
                vec![vec![0.5, 0.5], vec![0.3, 0.7]],
            )
            .unwrap();

        assert_eq!(store.entry_count(), 4);
        store.delete_repo(&repo1).unwrap();
        assert_eq!(store.entry_count(), 2);

        let counts = store.repo_counts();
        assert!(!counts.contains_key(&repo1));
        assert_eq!(*counts.get(&repo2).unwrap(), 2);
    }

    #[test]
    fn test_vector_store_filter_by_repo_id() {
        let dir = tempfile::tempdir().unwrap();
        let store = VectorStore::open_or_create(dir.path()).unwrap();
        let repo1 = Uuid::new_v4();
        let repo2 = Uuid::new_v4();

        store
            .add_chunks(
                repo1,
                "repo1",
                &make_chunks(1),
                vec![vec![1.0, 0.0, 0.0]],
            )
            .unwrap();
        store
            .add_chunks(
                repo2,
                "repo2",
                &make_chunks(1),
                vec![vec![0.9, 0.1, 0.0]],
            )
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 10, Some(&[repo2]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].repo_name, "repo2");
    }

    #[test]
    fn test_vector_store_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let repo_id = Uuid::new_v4();

        // Write
        {
            let store = VectorStore::open_or_create(dir.path()).unwrap();
            store
                .add_chunks(
                    repo_id,
                    "persisted",
                    &make_chunks(2),
                    vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                )
                .unwrap();
        }

        // Re-open and verify
        let store = VectorStore::open_or_create(dir.path()).unwrap();
        assert_eq!(store.entry_count(), 2);

        let results = store.search(&[1.0, 0.0], 10, None);
        assert_eq!(results.len(), 2);
    }
}
