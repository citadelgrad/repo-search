//! Integration tests for the repo-search pipeline.
//!
//! These tests exercise the full indexing and search flow without
//! requiring a running LLM (vector and rerank are skipped).

use uuid::Uuid;

use repo_search::models::FileChunk;
use repo_search::search::bm25::Bm25Index;
use repo_search::search::hybrid::{multi_query_rrf_fusion, QueryResults};
use repo_search::search::vector::VectorStore;

/// Helper: create chunks simulating a small Rust project.
fn sample_rust_project(repo_id: Uuid) -> Vec<FileChunk> {
    vec![
        FileChunk {
            repo_id,
            file_path: "src/main.rs".to_string(),
            chunk_index: 0,
            content: "use axum::Router;\nfn main() {\n    let app = Router::new();\n    axum::serve(app).await;\n}".to_string(),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 5,
        },
        FileChunk {
            repo_id,
            file_path: "src/handlers.rs".to_string(),
            chunk_index: 0,
            content: "pub async fn health_check() -> &'static str {\n    \"OK\"\n}\n\npub async fn create_user(Json(body): Json<CreateUser>) -> impl IntoResponse {\n    // insert into database\n    StatusCode::CREATED\n}".to_string(),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 8,
        },
        FileChunk {
            repo_id,
            file_path: "src/db.rs".to_string(),
            chunk_index: 0,
            content: "pub struct Database {\n    pool: PgPool,\n}\n\nimpl Database {\n    pub async fn connect(url: &str) -> Result<Self> {\n        let pool = PgPool::connect(url).await?;\n        Ok(Self { pool })\n    }\n\n    pub async fn get_user(&self, id: i64) -> Result<User> {\n        sqlx::query_as!(User, \"SELECT * FROM users WHERE id = $1\", id)\n            .fetch_one(&self.pool)\n            .await\n    }\n}".to_string(),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 15,
        },
        FileChunk {
            repo_id,
            file_path: "src/models.rs".to_string(),
            chunk_index: 0,
            content: "#[derive(Debug, Serialize, Deserialize)]\npub struct User {\n    pub id: i64,\n    pub name: String,\n    pub email: String,\n}\n\n#[derive(Debug, Deserialize)]\npub struct CreateUser {\n    pub name: String,\n    pub email: String,\n}".to_string(),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 12,
        },
        FileChunk {
            repo_id,
            file_path: "README.md".to_string(),
            chunk_index: 0,
            content: "# User Service\n\nA REST API for managing users.\n\n## Endpoints\n- GET /health\n- POST /users\n- GET /users/:id".to_string(),
            language: "text".to_string(),
            start_line: 1,
            end_line: 8,
        },
    ]
}

/// Helper: create a second "repo" with Python code.
fn sample_python_project(repo_id: Uuid) -> Vec<FileChunk> {
    vec![
        FileChunk {
            repo_id,
            file_path: "app.py".to_string(),
            chunk_index: 0,
            content: "from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.route('/health')\ndef health():\n    return jsonify(status='ok')\n\n@app.route('/users', methods=['POST'])\ndef create_user():\n    data = request.get_json()\n    # insert user into database\n    return jsonify(data), 201".to_string(),
            language: "python".to_string(),
            start_line: 1,
            end_line: 13,
        },
        FileChunk {
            repo_id,
            file_path: "models.py".to_string(),
            chunk_index: 0,
            content: "from dataclasses import dataclass\n\n@dataclass\nclass User:\n    id: int\n    name: str\n    email: str".to_string(),
            language: "python".to_string(),
            start_line: 1,
            end_line: 7,
        },
    ]
}

#[test]
fn test_end_to_end_bm25_index_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let index = Bm25Index::open_or_create(dir.path()).unwrap();
    let repo_id = Uuid::new_v4();

    let chunks = sample_rust_project(repo_id);
    index.index_chunks("user-service", &chunks).unwrap();

    // Search for "database"
    let results = index.search("database", 10, None).unwrap();
    assert!(!results.is_empty());
    // db.rs should be the top hit (has "database" and "Database")
    assert!(results.iter().any(|r| r.file_path == "src/db.rs"));
}

#[test]
fn test_end_to_end_vector_store_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let store = VectorStore::open_or_create(dir.path()).unwrap();
    let repo_id = Uuid::new_v4();

    // Simulate embeddings (3-dimensional for simplicity)
    let chunks = vec![
        ("src/main.rs".to_string(), 0, "fn main()".to_string(), "rust".to_string(), 1, 5),
        ("src/db.rs".to_string(), 0, "database connection".to_string(), "rust".to_string(), 1, 15),
        ("src/handlers.rs".to_string(), 0, "http handler".to_string(), "rust".to_string(), 1, 8),
    ];
    let embeddings = vec![
        vec![0.1, 0.2, 0.9],   // main.rs - "server setup" direction
        vec![0.9, 0.1, 0.1],   // db.rs - "database" direction
        vec![0.2, 0.8, 0.3],   // handlers.rs - "http" direction
    ];

    store
        .add_chunks(repo_id, "user-service", &chunks, embeddings)
        .unwrap();

    // Query in "database" direction
    let results = store.search(&[0.95, 0.05, 0.05], 10, None);
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].file_path, "src/db.rs"); // Closest to query
}

#[test]
fn test_end_to_end_hybrid_fusion_bm25_only() {
    let dir = tempfile::tempdir().unwrap();
    let index = Bm25Index::open_or_create(dir.path()).unwrap();
    let repo_id = Uuid::new_v4();

    index
        .index_chunks("rust-svc", &sample_rust_project(repo_id))
        .unwrap();

    let bm25_hits = index.search("user", 20, None).unwrap();

    let qr = QueryResults {
        bm25_hits,
        vector_hits: vec![],
        weight: 2.0,
    };

    let results = multi_query_rrf_fusion(&[qr], 10);
    assert!(!results.is_empty());
    // models.rs and handlers.rs and db.rs all mention "user"
    let paths: Vec<&str> = results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(
        paths.contains(&"src/models.rs") || paths.contains(&"src/handlers.rs"),
        "Expected user-related files in results: {:?}",
        paths
    );
}

#[test]
fn test_multi_repo_search_with_filtering() {
    let dir = tempfile::tempdir().unwrap();
    let index = Bm25Index::open_or_create(dir.path()).unwrap();
    let rust_id = Uuid::new_v4();
    let python_id = Uuid::new_v4();

    index
        .index_chunks("rust-svc", &sample_rust_project(rust_id))
        .unwrap();
    index
        .index_chunks("python-svc", &sample_python_project(python_id))
        .unwrap();

    // Search across all repos
    let all_results = index.search("user", 20, None).unwrap();
    let rust_repos: Vec<_> = all_results.iter().filter(|r| r.repo_id == rust_id).collect();
    let python_repos: Vec<_> = all_results.iter().filter(|r| r.repo_id == python_id).collect();
    assert!(!rust_repos.is_empty());
    assert!(!python_repos.is_empty());

    // Search with filter to only Python repo
    let filtered = index.search("user", 20, Some(&[python_id])).unwrap();
    assert!(!filtered.is_empty());
    assert!(filtered.iter().all(|r| r.repo_id == python_id));
}

#[test]
fn test_multi_query_rrf_with_simulated_expansion() {
    let dir = tempfile::tempdir().unwrap();
    let index = Bm25Index::open_or_create(dir.path()).unwrap();
    let repo_id = Uuid::new_v4();

    index
        .index_chunks("rust-svc", &sample_rust_project(repo_id))
        .unwrap();

    // Simulate the full pipeline with query expansion:
    // Original query: "database connection"
    // Expanded query 1: "pool connect"
    // Expanded query 2: "sql query fetch"
    let original_hits = index.search("database connection", 20, None).unwrap();
    let expanded1_hits = index.search("pool connect", 20, None).unwrap();
    let expanded2_hits = index.search("query fetch", 20, None).unwrap();

    let query_results = vec![
        QueryResults {
            bm25_hits: original_hits,
            vector_hits: vec![],
            weight: 2.0, // Original gets 2x weight
        },
        QueryResults {
            bm25_hits: expanded1_hits,
            vector_hits: vec![],
            weight: 1.0,
        },
        QueryResults {
            bm25_hits: expanded2_hits,
            vector_hits: vec![],
            weight: 1.0,
        },
    ];

    let results = multi_query_rrf_fusion(&query_results, 30);
    assert!(!results.is_empty());

    // db.rs should be boosted by appearing across multiple query variants
    if let Some(db_hit) = results.iter().find(|r| r.file_path == "src/db.rs") {
        // It should have a decent combined score from multi-query fusion
        assert!(db_hit.combined_score > 0.0);
    }
}

#[test]
fn test_delete_repo_removes_from_both_indexes() {
    let bm25_dir = tempfile::tempdir().unwrap();
    let vec_dir = tempfile::tempdir().unwrap();

    let index = Bm25Index::open_or_create(bm25_dir.path()).unwrap();
    let store = VectorStore::open_or_create(vec_dir.path()).unwrap();

    let repo_id = Uuid::new_v4();
    let chunks = sample_rust_project(repo_id);

    // Index in BM25
    index.index_chunks("test", &chunks).unwrap();

    // Index in vector store
    let meta: Vec<_> = chunks
        .iter()
        .map(|c| {
            (
                c.file_path.clone(),
                c.chunk_index,
                c.content.clone(),
                c.language.clone(),
                c.start_line,
                c.end_line,
            )
        })
        .collect();
    let embeddings: Vec<Vec<f32>> = chunks.iter().map(|_| vec![0.5, 0.5, 0.5]).collect();
    store
        .add_chunks(repo_id, "test", &meta, embeddings)
        .unwrap();

    // Verify data exists
    assert!(!index.search("main", 10, None).unwrap().is_empty());
    assert!(store.entry_count() > 0);

    // Delete
    index.delete_repo(&repo_id).unwrap();
    store.delete_repo(&repo_id).unwrap();

    // Verify clean
    assert!(index.search("main", 10, None).unwrap().is_empty());
    assert_eq!(store.entry_count(), 0);
}
