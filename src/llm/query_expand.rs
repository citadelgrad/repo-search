use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::LlmConfig;

/// Expand a user query into 2 alternative phrasings using the LLM.
pub async fn expand_query(
    client: &reqwest::Client,
    config: &LlmConfig,
    original_query: &str,
) -> Result<Vec<String>> {
    let prompt = format!(
        "You are a code search query expander. Given a search query, generate exactly 2 \
         alternative phrasings that capture different aspects or synonyms of the intent. \
         The alternatives should help find relevant code that the original query might miss.\n\n\
         Original query: \"{original_query}\"\n\n\
         Respond with ONLY a JSON array of 2 strings. No explanation.\n\
         Example: [\"alternative phrasing 1\", \"alternative phrasing 2\"]"
    );

    let response = match config.provider.as_str() {
        "ollama" => call_ollama(client, config, &prompt).await?,
        "openai" => call_openai(client, config, &prompt).await?,
        other => anyhow::bail!("Unknown LLM provider: {other}"),
    };

    parse_expanded_queries(&response)
}

fn parse_expanded_queries(content: &str) -> Result<Vec<String>> {
    // Extract JSON array from response
    let json_str = if let Some(start) = content.find('[') {
        if let Some(end) = content.rfind(']') {
            &content[start..=end]
        } else {
            content
        }
    } else {
        content
    };

    match serde_json::from_str::<Vec<String>>(json_str) {
        Ok(queries) => Ok(queries.into_iter().take(2).collect()),
        Err(e) => {
            tracing::warn!("Failed to parse expanded queries: {e}. Raw: {content}");
            Ok(Vec::new())
        }
    }
}

// ─── Ollama ──────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: Message,
}

async fn call_ollama(
    client: &reqwest::Client,
    config: &LlmConfig,
    prompt: &str,
) -> Result<String> {
    let url = format!("{}/api/chat", config.base_url);

    let req = OllamaChatRequest {
        model: config.chat_model.clone(),
        messages: vec![Message {
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
        .context("Failed to call Ollama chat API for query expansion")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Ollama chat API returned {status}: {body}");
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

async fn call_openai(
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
        temperature: 0.3,
    };

    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&req)
        .send()
        .await
        .context("Failed to call OpenAI chat API for query expansion")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI chat API returned {status}: {body}");
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

    #[test]
    fn test_parse_clean_json_array() {
        let input = r#"["authentication middleware", "auth handler function"]"#;
        let result = parse_expanded_queries(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "authentication middleware");
        assert_eq!(result[1], "auth handler function");
    }

    #[test]
    fn test_parse_json_embedded_in_text() {
        let input = "Here are 2 alternatives:\n[\"error handling\", \"exception management\"]\nHope that helps!";
        let result = parse_expanded_queries(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "error handling");
        assert_eq!(result[1], "exception management");
    }

    #[test]
    fn test_parse_json_in_markdown_code_block() {
        let input = "```json\n[\"database query\", \"SQL statement\"]\n```";
        let result = parse_expanded_queries(input).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_truncates_to_two() {
        let input = r#"["a", "b", "c", "d"]"#;
        let result = parse_expanded_queries(input).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_single_item() {
        let input = r#"["only one"]"#;
        let result = parse_expanded_queries(input).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_parse_empty_array() {
        let input = "[]";
        let result = parse_expanded_queries(input).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_garbage_returns_empty() {
        let input = "I don't understand the question.";
        let result = parse_expanded_queries(input).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_no_closing_bracket() {
        // Only opening bracket, no closing - should fallback gracefully
        let input = "[\"partial";
        let result = parse_expanded_queries(input).unwrap();
        assert!(result.is_empty()); // parse fails gracefully
    }

    #[test]
    fn test_parse_unicode_queries() {
        let input = r#"["函数定义", "関数の実装"]"#;
        let result = parse_expanded_queries(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "函数定义");
    }
}
