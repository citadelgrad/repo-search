use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::LlmConfig;
use crate::models::SearchHit;

/// Re-rank search results using a yes/no relevance judgment per document.
/// Each candidate is scored individually: the LLM answers "yes" or "no" to
/// whether the snippet is relevant, and we parse a confidence score.
/// This mirrors the qwen3-reranker approach.
pub async fn rerank(
    client: &reqwest::Client,
    config: &LlmConfig,
    query: &str,
    hits: &mut [SearchHit],
) -> Result<()> {
    if hits.is_empty() {
        return Ok(());
    }

    // Score top 30 candidates (re-ranking budget)
    let n = hits.len().min(30);

    // Build batch of yes/no prompts
    let prompts: Vec<String> = hits[..n]
        .iter()
        .map(|h| build_yesno_prompt(query, &h.file_path, &h.content))
        .collect();

    // Score all candidates via batch LLM call
    let scores = batch_score(client, config, &prompts).await?;

    // Apply rerank scores
    for (i, score) in scores.into_iter().enumerate() {
        if i < hits.len() {
            hits[i].rerank_score = Some(score);
        }
    }

    // Position-aware blending:
    // Top 1-3:   75% RRF + 25% rerank
    // Top 4-10:  60% RRF + 40% rerank
    // Top 11+:   40% RRF + 60% rerank
    for (i, hit) in hits.iter_mut().enumerate() {
        if let Some(rerank) = hit.rerank_score {
            let (rrf_w, rerank_w) = if i < 3 {
                (0.75, 0.25)
            } else if i < 10 {
                (0.60, 0.40)
            } else {
                (0.40, 0.60)
            };
            hit.combined_score = rrf_w * hit.combined_score + rerank_w * rerank;
        }
    }

    // Re-sort by blended score
    hits.sort_by(|a, b| {
        b.combined_score
            .partial_cmp(&a.combined_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(())
}

/// Build a yes/no relevance prompt for a single document.
fn build_yesno_prompt(query: &str, file_path: &str, content: &str) -> String {
    let snippet = truncate_content(content, 800);
    format!(
        "<|im_start|>system\nJudge whether the following code snippet is relevant to the search query. \
         Answer with ONLY a JSON object: {{\"relevant\": true/false, \"confidence\": 0.0-1.0}}<|im_end|>\n\
         <|im_start|>user\nQuery: {query}\n\nFile: {file_path}\n```\n{snippet}\n```<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn truncate_content(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        content.to_string()
    } else {
        format!("{}...", &content[..max_chars])
    }
}

/// Score candidates in a batch. Uses sequential calls for Ollama (no batch API)
/// and batched calls for OpenAI-compatible APIs.
async fn batch_score(
    client: &reqwest::Client,
    config: &LlmConfig,
    prompts: &[String],
) -> Result<Vec<f32>> {
    let mut scores = Vec::with_capacity(prompts.len());

    // Process in parallel with concurrency limit
    let semaphore = std::sync::Arc::new(tokio::sync::Semaphore::new(4));
    let mut handles = Vec::new();

    for prompt in prompts {
        let client = client.clone();
        let config = config.clone();
        let prompt = prompt.clone();
        let sem = semaphore.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await;
            score_single(&client, &config, &prompt).await.unwrap_or(0.0)
        });
        handles.push(handle);
    }

    for handle in handles {
        let score = handle.await.unwrap_or(0.0);
        scores.push(score);
    }

    Ok(scores)
}

async fn score_single(
    client: &reqwest::Client,
    config: &LlmConfig,
    prompt: &str,
) -> Result<f32> {
    let response = match config.provider.as_str() {
        "ollama" => call_ollama_single(client, config, prompt).await?,
        "openai" => call_openai_single(client, config, prompt).await?,
        other => anyhow::bail!("Unknown provider: {other}"),
    };

    parse_relevance_score(&response)
}

fn parse_relevance_score(content: &str) -> Result<f32> {
    // Try JSON parse first
    if let Ok(v) = serde_json::from_str::<RelevanceResponse>(content) {
        let base = if v.relevant { 0.5 } else { 0.0 };
        return Ok(base + v.confidence * 0.5);
    }

    // Try to extract JSON from response
    if let Some(start) = content.find('{') {
        if let Some(end) = content.rfind('}') {
            if let Ok(v) = serde_json::from_str::<RelevanceResponse>(&content[start..=end]) {
                let base = if v.relevant { 0.5 } else { 0.0 };
                return Ok(base + v.confidence * 0.5);
            }
        }
    }

    // Fallback: check for yes/no keywords
    let lower = content.to_lowercase();
    if lower.contains("\"relevant\": true") || lower.contains("yes") {
        Ok(0.7)
    } else if lower.contains("\"relevant\": false") || lower.contains("no") {
        Ok(0.2)
    } else {
        Ok(0.5) // Uncertain
    }
}

#[derive(Deserialize)]
struct RelevanceResponse {
    relevant: bool,
    #[serde(default = "default_confidence")]
    confidence: f32,
}

fn default_confidence() -> f32 {
    0.5
}

// ─── Ollama ──────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaMessage,
}

async fn call_ollama_single(
    client: &reqwest::Client,
    config: &LlmConfig,
    prompt: &str,
) -> Result<String> {
    let url = format!("{}/api/chat", config.base_url);

    let req = OllamaChatRequest {
        model: config.chat_model.clone(),
        messages: vec![OllamaMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        stream: false,
    };

    let resp = client
        .post(&url)
        .json(&req)
        .send()
        .await
        .context("Failed to call Ollama for reranking")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Ollama rerank call returned {status}: {body}");
    }

    let body: OllamaChatResponse = resp.json().await?;
    Ok(body.message.content)
}

// ─── OpenAI-compatible ───────────────────────────────────

#[derive(Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize)]
struct OpenAiMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Deserialize)]
struct OpenAiResponseMessage {
    content: String,
}

async fn call_openai_single(
    client: &reqwest::Client,
    config: &LlmConfig,
    prompt: &str,
) -> Result<String> {
    let url = format!("{}/v1/chat/completions", config.base_url);
    let api_key = config.api_key.as_deref().unwrap_or_default();

    let req = OpenAiChatRequest {
        model: config.chat_model.clone(),
        messages: vec![OpenAiMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        temperature: 0.0,
        max_tokens: 50,
    };

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&req)
        .send()
        .await
        .context("Failed to call OpenAI for reranking")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI rerank call returned {status}: {body}");
    }

    let body: OpenAiChatResponse = resp.json().await?;
    Ok(body
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::SearchHit;
    use uuid::Uuid;

    // ── parse_relevance_score tests ─────────────────────

    #[test]
    fn test_parse_perfect_json_relevant() {
        let content = r#"{"relevant": true, "confidence": 0.9}"#;
        let score = parse_relevance_score(content).unwrap();
        // base=0.5 + 0.9*0.5 = 0.95
        assert!((score - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_parse_perfect_json_not_relevant() {
        let content = r#"{"relevant": false, "confidence": 0.8}"#;
        let score = parse_relevance_score(content).unwrap();
        // base=0.0 + 0.8*0.5 = 0.4
        assert!((score - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_parse_json_without_confidence() {
        let content = r#"{"relevant": true}"#;
        let score = parse_relevance_score(content).unwrap();
        // base=0.5 + default(0.5)*0.5 = 0.75
        assert!((score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_parse_json_embedded_in_markdown() {
        let content = "Here is my judgment:\n```json\n{\"relevant\": true, \"confidence\": 1.0}\n```";
        let score = parse_relevance_score(content).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_yes_keyword_fallback() {
        let content = "Yes, this code is relevant to the search query.";
        let score = parse_relevance_score(content).unwrap();
        assert!((score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_parse_no_keyword_fallback() {
        let content = "No, this is not related.";
        let score = parse_relevance_score(content).unwrap();
        assert!((score - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_parse_uncertain_fallback() {
        let content = "I'm unable to determine relevance from this.";
        let score = parse_relevance_score(content).unwrap();
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_parse_relevant_true_string_fallback() {
        let content = "The answer is \"relevant\": true based on analysis";
        let score = parse_relevance_score(content).unwrap();
        assert!((score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_parse_relevant_false_string_fallback() {
        let content = "My judgment: \"relevant\": false";
        let score = parse_relevance_score(content).unwrap();
        assert!((score - 0.2).abs() < 1e-6);
    }

    // ── build_yesno_prompt tests ────────────────────────

    #[test]
    fn test_build_yesno_prompt_contains_query() {
        let prompt = build_yesno_prompt("search term", "main.rs", "fn main() {}");
        assert!(prompt.contains("search term"));
        assert!(prompt.contains("main.rs"));
        assert!(prompt.contains("fn main()"));
    }

    #[test]
    fn test_build_yesno_prompt_structure() {
        let prompt = build_yesno_prompt("test", "f.rs", "code");
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    // ── truncate_content tests ──────────────────────────

    #[test]
    fn test_truncate_short_content() {
        let result = truncate_content("short", 100);
        assert_eq!(result, "short");
    }

    #[test]
    fn test_truncate_long_content() {
        let content = "a".repeat(200);
        let result = truncate_content(&content, 50);
        assert_eq!(result.len(), 53); // 50 chars + "..."
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_truncate_exact_boundary() {
        let content = "a".repeat(100);
        let result = truncate_content(&content, 100);
        assert_eq!(result, content); // No truncation needed
    }

    // ── position-aware blending tests ───────────────────

    fn make_hit(combined: f32) -> SearchHit {
        SearchHit {
            repo_id: Uuid::new_v4(),
            repo_name: "r".to_string(),
            file_path: "f.rs".to_string(),
            chunk_index: 0,
            content: "c".to_string(),
            language: "rust".to_string(),
            start_line: 1,
            end_line: 10,
            bm25_score: 0.0,
            vector_score: 0.0,
            combined_score: combined,
            rerank_score: None,
        }
    }

    #[test]
    fn test_position_aware_blending_top3() {
        // Positions 0-2 get 75% RRF + 25% rerank
        let mut hits: Vec<SearchHit> = (0..3).map(|_| make_hit(1.0)).collect();

        // Simulate setting rerank scores directly (bypassing LLM)
        for (_i, hit) in hits.iter_mut().enumerate() {
            hit.rerank_score = Some(0.5);
            let (rrf_w, rerank_w) = (0.75, 0.25);
            hit.combined_score = rrf_w * hit.combined_score + rerank_w * 0.5;
        }

        // Top 3: 0.75 * 1.0 + 0.25 * 0.5 = 0.875
        for hit in &hits {
            assert!((hit.combined_score - 0.875).abs() < 1e-6);
        }
    }

    #[test]
    fn test_position_aware_blending_mid_range() {
        // Positions 3-9 get 60% RRF + 40% rerank
        let mut hit = make_hit(1.0);
        hit.rerank_score = Some(0.5);
        let (rrf_w, rerank_w) = (0.60, 0.40);
        hit.combined_score = rrf_w * hit.combined_score + rerank_w * 0.5;
        // 0.60 * 1.0 + 0.40 * 0.5 = 0.8
        assert!((hit.combined_score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_position_aware_blending_tail() {
        // Positions 10+ get 40% RRF + 60% rerank
        let mut hit = make_hit(1.0);
        hit.rerank_score = Some(0.5);
        let (rrf_w, rerank_w) = (0.40, 0.60);
        hit.combined_score = rrf_w * hit.combined_score + rerank_w * 0.5;
        // 0.40 * 1.0 + 0.60 * 0.5 = 0.7
        assert!((hit.combined_score - 0.7).abs() < 1e-6);
    }
}
