use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::LlmConfig;

/// Maximum characters to send per text to the embedding API.
/// nomic-embed-text has an 8 192-token context.  Most code tokenises at
/// ~1 token per 2-3 chars, but dense content (JSON blobs, minified JS) can
/// hit ~2.3 tokens/char.  3 000 chars × 2.3 ≈ 6 900 tokens — safely under 8 192.
/// We also pass `truncate: true` to Ollama, but it has a known bug where it
/// still returns 400 for inputs that exceed the context length.
const MAX_EMBED_CHARS: usize = 3_000;

/// Task type for asymmetric embedding models (e.g. nomic-embed-text, E5).
/// These models were trained with different prefixes for queries vs documents,
/// which improves the embedding space geometry for retrieval tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedTask {
    /// Indexing: text being stored for later retrieval.
    SearchDocument,
    /// Querying: text used to search against stored documents.
    SearchQuery,
}

impl EmbedTask {
    /// Return the prefix string for the given embedding model.
    /// The trailing space after the colon is required by most models.
    pub fn prefix_for_model(&self, model_name: &str) -> &'static str {
        let lower = model_name.to_lowercase();
        if lower.contains("nomic") {
            match self {
                EmbedTask::SearchDocument => "search_document: ",
                EmbedTask::SearchQuery => "search_query: ",
            }
        } else if lower.contains("e5") {
            match self {
                EmbedTask::SearchDocument => "passage: ",
                EmbedTask::SearchQuery => "query: ",
            }
        } else {
            "" // Unknown model: no prefix (backwards-compatible)
        }
    }
}

/// Truncate `text` to at most `max_chars`, splitting on a UTF-8 char boundary.
fn truncate_for_embedding(text: &str, max_chars: usize) -> &str {
    if text.len() <= max_chars {
        return text;
    }
    // Find the last char boundary at or before the limit
    let mut end = max_chars;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Generate embeddings for a batch of texts using the configured LLM provider.
pub async fn embed_batch(
    client: &reqwest::Client,
    config: &LlmConfig,
    texts: &[String],
    task: EmbedTask,
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let prefix = task.prefix_for_model(&config.embedding_model);
    let prefix_len = prefix.len();

    // Apply prefix before truncation so the prefix is never truncated away.
    let truncated: Vec<String> = texts
        .iter()
        .map(|t| {
            let body = truncate_for_embedding(t, MAX_EMBED_CHARS.saturating_sub(prefix_len));
            format!("{prefix}{body}")
        })
        .collect();

    match config.provider.as_str() {
        "ollama" => embed_ollama(client, config, &truncated).await,
        "openai" => embed_openai(client, config, &truncated).await,
        other => anyhow::bail!("Unknown LLM provider: {other}"),
    }
}

/// Generate embedding for a single text.
pub async fn embed_single(
    client: &reqwest::Client,
    config: &LlmConfig,
    text: &str,
    task: EmbedTask,
) -> Result<Vec<f32>> {
    let results = embed_batch(client, config, &[text.to_string()], task).await?;
    results
        .into_iter()
        .next()
        .context("No embedding returned")
}

// ─── Ollama ──────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: Vec<String>,
    /// Ask Ollama to silently truncate inputs that exceed the model's context
    /// length instead of returning a 400 error.
    truncate: bool,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

async fn embed_ollama(
    client: &reqwest::Client,
    config: &LlmConfig,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let url = format!("{}/api/embed", config.base_url);

    // Ollama supports batch embedding with the /api/embed endpoint
    let batch_size = 32;
    let mut all_embeddings = Vec::new();

    for chunk in texts.chunks(batch_size) {
        let req = OllamaEmbedRequest {
            model: config.embedding_model.clone(),
            input: chunk.to_vec(),
            truncate: true,
        };

        let resp = client
            .post(&url)
            .json(&req)
            .send()
            .await
            .context("Failed to call Ollama embed API")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama embed API returned {status}: {body}");
        }

        let body: OllamaEmbedResponse = resp
            .json()
            .await
            .context("Failed to parse Ollama embed response")?;

        all_embeddings.extend(body.embeddings);
    }

    Ok(all_embeddings)
}

// ─── OpenAI-compatible ───────────────────────────────────

#[derive(Serialize)]
struct OpenAiEmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct OpenAiEmbedResponse {
    data: Vec<OpenAiEmbedData>,
}

#[derive(Deserialize)]
struct OpenAiEmbedData {
    embedding: Vec<f32>,
}

async fn embed_openai(
    client: &reqwest::Client,
    config: &LlmConfig,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    let url = format!("{}/v1/embeddings", config.base_url);
    let api_key = config
        .api_key
        .as_deref()
        .unwrap_or_default();

    let batch_size = 64;
    let mut all_embeddings = Vec::new();

    for chunk in texts.chunks(batch_size) {
        let req = OpenAiEmbedRequest {
            model: config.embedding_model.clone(),
            input: chunk.to_vec(),
        };

        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&req)
            .send()
            .await
            .context("Failed to call OpenAI embed API")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenAI embed API returned {status}: {body}");
        }

        let body: OpenAiEmbedResponse = resp
            .json()
            .await
            .context("Failed to parse OpenAI embed response")?;

        let mut embeddings: Vec<Vec<f32>> = body.data.into_iter().map(|d| d.embedding).collect();
        all_embeddings.append(&mut embeddings);
    }

    Ok(all_embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── EmbedTask prefix tests ──────────────────────────

    #[test]
    fn test_nomic_document_prefix() {
        let prefix = EmbedTask::SearchDocument.prefix_for_model("nomic-embed-text");
        assert_eq!(prefix, "search_document: ");
    }

    #[test]
    fn test_nomic_query_prefix() {
        let prefix = EmbedTask::SearchQuery.prefix_for_model("nomic-embed-text");
        assert_eq!(prefix, "search_query: ");
    }

    #[test]
    fn test_nomic_case_insensitive() {
        let prefix = EmbedTask::SearchQuery.prefix_for_model("Nomic-Embed-Text-v1.5");
        assert_eq!(prefix, "search_query: ");
    }

    #[test]
    fn test_e5_document_prefix() {
        let prefix = EmbedTask::SearchDocument.prefix_for_model("e5-large-v2");
        assert_eq!(prefix, "passage: ");
    }

    #[test]
    fn test_e5_query_prefix() {
        let prefix = EmbedTask::SearchQuery.prefix_for_model("e5-large-v2");
        assert_eq!(prefix, "query: ");
    }

    #[test]
    fn test_unknown_model_no_prefix() {
        let prefix = EmbedTask::SearchDocument.prefix_for_model("all-minilm-l6-v2");
        assert_eq!(prefix, "");
        let prefix = EmbedTask::SearchQuery.prefix_for_model("all-minilm-l6-v2");
        assert_eq!(prefix, "");
    }

    // ── truncation tests ────────────────────────────────

    #[test]
    fn test_truncate_short_text() {
        let text = "short text";
        assert_eq!(truncate_for_embedding(text, 100), "short text");
    }

    #[test]
    fn test_truncate_at_limit() {
        let text = "a".repeat(100);
        assert_eq!(truncate_for_embedding(&text, 100).len(), 100);
    }

    #[test]
    fn test_truncate_over_limit() {
        let text = "a".repeat(200);
        assert_eq!(truncate_for_embedding(&text, 100).len(), 100);
    }

    #[test]
    fn test_truncate_respects_utf8_boundary() {
        // é is 2 bytes in UTF-8
        let text = "é".repeat(100); // 200 bytes
        let result = truncate_for_embedding(&text, 150);
        assert!(result.len() <= 150);
        // Should be on a char boundary (even number of bytes for this char)
        assert!(result.len() % 2 == 0);
    }
}
