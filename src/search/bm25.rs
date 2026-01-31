use anyhow::{Context, Result};
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy};
use uuid::Uuid;

use crate::models::FileChunk;

/// BM25 search index built on tantivy.
pub struct Bm25Index {
    index: Index,
    #[allow(dead_code)]
    schema: Schema,
    // Field handles
    f_repo_id: Field,
    f_repo_name: Field,
    f_file_path: Field,
    f_chunk_index: Field,
    f_content: Field,
    f_language: Field,
    f_start_line: Field,
    f_end_line: Field,
}

#[derive(Debug, Clone)]
pub struct Bm25Hit {
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

impl Bm25Index {
    /// Create or open a BM25 index at the given directory.
    pub fn open_or_create(index_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(index_dir)?;

        let mut schema_builder = Schema::builder();
        let f_repo_id = schema_builder.add_text_field("repo_id", STRING | STORED);
        let f_repo_name = schema_builder.add_text_field("repo_name", STRING | STORED);
        let f_file_path = schema_builder.add_text_field("file_path", TEXT | STORED);
        let f_chunk_index =
            schema_builder.add_u64_field("chunk_index", NumericOptions::default() | STORED);
        let f_content = schema_builder.add_text_field("content", TEXT | STORED);
        let f_language = schema_builder.add_text_field("language", STRING | STORED);
        let f_start_line =
            schema_builder.add_u64_field("start_line", NumericOptions::default() | STORED);
        let f_end_line =
            schema_builder.add_u64_field("end_line", NumericOptions::default() | STORED);

        let schema = schema_builder.build();

        let index = if index_dir.join("meta.json").exists() {
            Index::open_in_dir(index_dir).context("Failed to open existing tantivy index")?
        } else {
            Index::create_in_dir(index_dir, schema.clone())
                .context("Failed to create tantivy index")?
        };

        Ok(Self {
            index,
            schema,
            f_repo_id,
            f_repo_name,
            f_file_path,
            f_chunk_index,
            f_content,
            f_language,
            f_start_line,
            f_end_line,
        })
    }

    /// Index a batch of file chunks for a repo.
    pub fn index_chunks(&self, repo_name: &str, chunks: &[FileChunk]) -> Result<()> {
        let mut writer: IndexWriter = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        for chunk in chunks {
            writer.add_document(doc!(
                self.f_repo_id => chunk.repo_id.to_string(),
                self.f_repo_name => repo_name.to_string(),
                self.f_file_path => chunk.file_path.clone(),
                self.f_chunk_index => chunk.chunk_index as u64,
                self.f_content => chunk.content.clone(),
                self.f_language => chunk.language.clone(),
                self.f_start_line => chunk.start_line as u64,
                self.f_end_line => chunk.end_line as u64,
            ))?;
        }

        writer.commit().context("Failed to commit index")?;
        Ok(())
    }

    /// Delete all documents for a given repo.
    pub fn delete_repo(&self, repo_id: &Uuid) -> Result<()> {
        let mut writer: IndexWriter = self
            .index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        let term = tantivy::Term::from_field_text(self.f_repo_id, &repo_id.to_string());
        writer.delete_term(term);
        writer.commit().context("Failed to commit delete")?;
        Ok(())
    }

    /// Search the index and return scored hits.
    pub fn search(
        &self,
        query_str: &str,
        limit: usize,
        repo_ids: Option<&[Uuid]>,
    ) -> Result<Vec<Bm25Hit>> {
        let reader = self
            .index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .context("Failed to create reader")?;

        let searcher = reader.searcher();

        let query_parser =
            QueryParser::for_index(&self.index, vec![self.f_content, self.f_file_path]);
        let query = query_parser
            .parse_query(query_str)
            .context("Failed to parse search query")?;

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit * 2))
            .context("Search failed")?;

        let mut hits = Vec::new();

        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher
                .doc(doc_address)
                .context("Failed to retrieve document")?;

            let repo_id_str = doc
                .get_first(self.f_repo_id)
                .and_then(|v| v.as_str())
                .unwrap_or_default();

            let repo_id = match Uuid::parse_str(repo_id_str) {
                Ok(id) => id,
                Err(_) => continue,
            };

            // Filter by repo_ids if specified
            if let Some(ids) = repo_ids {
                if !ids.contains(&repo_id) {
                    continue;
                }
            }

            let repo_name = doc
                .get_first(self.f_repo_name)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let file_path = doc
                .get_first(self.f_file_path)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let chunk_index = doc
                .get_first(self.f_chunk_index)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            let content = doc
                .get_first(self.f_content)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let language = doc
                .get_first(self.f_language)
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let start_line = doc
                .get_first(self.f_start_line)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            let end_line = doc
                .get_first(self.f_end_line)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            hits.push(Bm25Hit {
                repo_id,
                repo_name,
                file_path,
                chunk_index,
                content,
                language,
                start_line,
                end_line,
                score,
            });

            if hits.len() >= limit {
                break;
            }
        }

        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::FileChunk;
    use uuid::Uuid;

    fn make_chunk(repo_id: Uuid, path: &str, idx: usize, content: &str) -> FileChunk {
        FileChunk {
            repo_id,
            file_path: path.to_string(),
            chunk_index: idx,
            content: content.to_string(),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 10,
        }
    }

    #[test]
    fn test_create_fresh_index() {
        let dir = tempfile::tempdir().unwrap();
        let index = Bm25Index::open_or_create(dir.path()).unwrap();
        let results = index.search("anything", 10, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_and_search_basic() {
        let dir = tempfile::tempdir().unwrap();
        let index = Bm25Index::open_or_create(dir.path()).unwrap();
        let repo_id = Uuid::new_v4();

        let chunks = vec![
            make_chunk(
                repo_id,
                "src/main.rs",
                0,
                "fn main() { println!(\"hello world\"); }",
            ),
            make_chunk(
                repo_id,
                "src/lib.rs",
                0,
                "pub fn add(a: i32, b: i32) -> i32 { a + b }",
            ),
            make_chunk(
                repo_id,
                "README.md",
                0,
                "This is a documentation file about the project",
            ),
        ];

        index.index_chunks("test-repo", &chunks).unwrap();

        let results = index.search("main", 10, None).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.file_path == "src/main.rs"));
    }

    #[test]
    fn test_search_respects_limit() {
        let dir = tempfile::tempdir().unwrap();
        let index = Bm25Index::open_or_create(dir.path()).unwrap();
        let repo_id = Uuid::new_v4();

        let chunks: Vec<_> = (0..20)
            .map(|i| {
                make_chunk(
                    repo_id,
                    &format!("file_{i}.rs"),
                    i,
                    &format!("function number {i} with common keyword"),
                )
            })
            .collect();

        index.index_chunks("test-repo", &chunks).unwrap();

        let results = index
            .search("function common keyword", 5, None)
            .unwrap();
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_returns_stored_fields() {
        let dir = tempfile::tempdir().unwrap();
        let index = Bm25Index::open_or_create(dir.path()).unwrap();
        let repo_id = Uuid::new_v4();

        let chunks = vec![make_chunk(
            repo_id,
            "src/handler.rs",
            3,
            "async fn handle_request(req: Request) -> Response { todo!() }",
        )];
        index.index_chunks("my-repo", &chunks).unwrap();

        let results = index.search("handle request", 10, None).unwrap();
        assert!(!results.is_empty());
        let hit = &results[0];
        assert_eq!(hit.repo_id, repo_id);
        assert_eq!(hit.repo_name, "my-repo");
        assert_eq!(hit.file_path, "src/handler.rs");
        assert_eq!(hit.chunk_index, 3);
        assert_eq!(hit.language, "rust");
        assert!(hit.score > 0.0);
    }

    #[test]
    fn test_delete_repo_removes_documents() {
        let dir = tempfile::tempdir().unwrap();
        let index = Bm25Index::open_or_create(dir.path()).unwrap();
        let repo1 = Uuid::new_v4();
        let repo2 = Uuid::new_v4();

        index
            .index_chunks(
                "repo1",
                &[make_chunk(repo1, "a.rs", 0, "unique_token_alpha function")],
            )
            .unwrap();
        index
            .index_chunks(
                "repo2",
                &[make_chunk(repo2, "b.rs", 0, "unique_token_beta function")],
            )
            .unwrap();

        assert!(!index
            .search("unique_token_alpha", 10, None)
            .unwrap()
            .is_empty());
        assert!(!index
            .search("unique_token_beta", 10, None)
            .unwrap()
            .is_empty());

        index.delete_repo(&repo1).unwrap();

        assert!(index
            .search("unique_token_alpha", 10, None)
            .unwrap()
            .is_empty());
        assert!(!index
            .search("unique_token_beta", 10, None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_search_filter_by_repo_ids() {
        let dir = tempfile::tempdir().unwrap();
        let index = Bm25Index::open_or_create(dir.path()).unwrap();
        let repo1 = Uuid::new_v4();
        let repo2 = Uuid::new_v4();

        index
            .index_chunks(
                "repo1",
                &[make_chunk(repo1, "a.rs", 0, "shared search term code")],
            )
            .unwrap();
        index
            .index_chunks(
                "repo2",
                &[make_chunk(repo2, "b.rs", 0, "shared search term code")],
            )
            .unwrap();

        let results = index
            .search("shared search term", 10, Some(&[repo2]))
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.repo_id == repo2));
    }

    #[test]
    fn test_reopen_existing_index() {
        let dir = tempfile::tempdir().unwrap();
        let repo_id = Uuid::new_v4();

        {
            let index = Bm25Index::open_or_create(dir.path()).unwrap();
            index
                .index_chunks(
                    "repo",
                    &[make_chunk(
                        repo_id,
                        "persisted.rs",
                        0,
                        "persistence test data content",
                    )],
                )
                .unwrap();
        }

        let index = Bm25Index::open_or_create(dir.path()).unwrap();
        let results = index.search("persistence test", 10, None).unwrap();
        assert!(!results.is_empty());
    }
}
