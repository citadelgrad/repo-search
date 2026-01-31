use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::LlmConfig;

/// Generate embeddings for a batch of texts using the configured LLM provider.
pub async fn embed_batch(
    client: &reqwest::Client,
    config: &LlmConfig,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    match config.provider.as_str() {
        "ollama" => embed_ollama(client, config, texts).await,
        "openai" => embed_openai(client, config, texts).await,
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
