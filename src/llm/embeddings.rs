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

/// Truncate `text` to at most `MAX_EMBED_CHARS`, splitting on a UTF-8 char boundary.
fn truncate_for_embedding(text: &str) -> &str {
    if text.len() <= MAX_EMBED_CHARS {
        return text;
    }
    // Find the last char boundary at or before the limit
    let mut end = MAX_EMBED_CHARS;
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
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let truncated: Vec<String> = texts
        .iter()
        .map(|t| truncate_for_embedding(t).to_string())
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
) -> Result<Vec<f32>> {
    let results = embed_batch(client, config, &[text.to_string()]).await?;
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
