use anyhow::{Context, Result};
use futures_util::stream::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::time::Duration;

use crate::config::LlmConfig;
use crate::models::ChatMessage;

type ChatStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// Stream chat completions from Ollama or OpenAI.
/// Returns a stream of content delta strings (one per token/chunk).
pub async fn stream_chat(
    client: &reqwest::Client,
    config: &LlmConfig,
    messages: Vec<ChatMessage>,
) -> Result<ChatStream> {
    match config.provider.as_str() {
        "ollama" => stream_ollama(client, config, messages).await,
        "openai" => stream_openai(client, config, messages).await,
        other => anyhow::bail!("Unsupported LLM provider for chat: {other}"),
    }
}

// ─── Ollama streaming ────────────────────────────────────

#[derive(Serialize)]
struct OllamaStreamRequest {
    model: String,
    messages: Vec<StreamMessage>,
    stream: bool,
}

#[derive(Serialize, Deserialize)]
struct StreamMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OllamaStreamChunk {
    message: StreamMessage,
    done: bool,
}

async fn stream_ollama(
    client: &reqwest::Client,
    config: &LlmConfig,
    messages: Vec<ChatMessage>,
) -> Result<ChatStream> {
    let url = format!("{}/api/chat", config.base_url);

    let req = OllamaStreamRequest {
        model: config.chat_model.clone(),
        messages: messages
            .into_iter()
            .map(|m| StreamMessage {
                role: m.role,
                content: m.content,
            })
            .collect(),
        stream: true,
    };

    let resp = client
        .post(&url)
        .timeout(Duration::from_secs(300))
        .json(&req)
        .send()
        .await
        .context("Failed to connect to Ollama for chat streaming")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Ollama chat API returned {status}: {body}");
    }

    let stream = stream_lines(resp.bytes_stream()).filter_map(|line_result| async move {
        match line_result {
            Ok(line) => parse_ollama_line(&line),
            Err(e) => Some(Err(e)),
        }
    });

    Ok(Box::pin(stream))
}

/// Parse a single Ollama streaming line. Returns:
/// - Some(Ok(content)) for content deltas
/// - Some(Err(e)) for parse errors
/// - None to skip (empty content or done signal)
fn parse_ollama_line(line: &str) -> Option<Result<String>> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    match serde_json::from_str::<OllamaStreamChunk>(line) {
        Ok(chunk) => {
            if chunk.done {
                return None;
            }
            let content = chunk.message.content;
            if content.is_empty() {
                return None;
            }
            Some(Ok(content))
        }
        Err(e) => Some(Err(anyhow::anyhow!("Failed to parse Ollama chunk: {e}"))),
    }
}

// ─── OpenAI streaming ────────────────────────────────────

#[derive(Serialize)]
struct OpenAiStreamRequest {
    model: String,
    messages: Vec<StreamMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
}

#[derive(Deserialize)]
struct OpenAiStreamDelta {
    content: Option<String>,
}

async fn stream_openai(
    client: &reqwest::Client,
    config: &LlmConfig,
    messages: Vec<ChatMessage>,
) -> Result<ChatStream> {
    let url = format!("{}/v1/chat/completions", config.base_url);

    let req = OpenAiStreamRequest {
        model: config.chat_model.clone(),
        messages: messages
            .into_iter()
            .map(|m| StreamMessage {
                role: m.role,
                content: m.content,
            })
            .collect(),
        stream: true,
    };

    let resp = client
        .post(&url)
        .timeout(Duration::from_secs(300))
        .header(
            "Authorization",
            format!("Bearer {}", config.api_key.as_deref().unwrap_or("")),
        )
        .json(&req)
        .send()
        .await
        .context("Failed to connect to OpenAI for chat streaming")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI chat API returned {status}: {body}");
    }

    let stream = stream_lines(resp.bytes_stream()).filter_map(|line_result| async move {
        match line_result {
            Ok(line) => parse_openai_line(&line),
            Err(e) => Some(Err(e)),
        }
    });

    Ok(Box::pin(stream))
}

/// Parse a single OpenAI SSE line. Returns:
/// - Some(Ok(content)) for content deltas
/// - Some(Err(e)) for parse errors
/// - None to skip (empty lines, [DONE], role-only chunks)
fn parse_openai_line(line: &str) -> Option<Result<String>> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    let data = if let Some(d) = line.strip_prefix("data: ") {
        d.trim()
    } else {
        return None;
    };

    if data == "[DONE]" {
        return None;
    }

    match serde_json::from_str::<OpenAiStreamChunk>(data) {
        Ok(chunk) => {
            let content = chunk
                .choices
                .first()
                .and_then(|c| c.delta.content.clone())
                .unwrap_or_default();
            if content.is_empty() {
                return None;
            }
            Some(Ok(content))
        }
        Err(e) => Some(Err(anyhow::anyhow!("Failed to parse OpenAI chunk: {e}"))),
    }
}

// ─── Line buffering ──────────────────────────────────────

/// Convert a byte stream into a stream of complete lines.
fn stream_lines(
    byte_stream: impl Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
) -> impl Stream<Item = Result<String>> + Send {
    futures_util::stream::unfold(
        (Box::pin(byte_stream), String::new()),
        |(mut stream, mut buffer)| async move {
            loop {
                // First, try to extract a complete line from the buffer
                if let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].to_string();
                    buffer = buffer[newline_pos + 1..].to_string();
                    if !line.trim().is_empty() {
                        return Some((Ok(line), (stream, buffer)));
                    }
                    continue;
                }

                // Buffer has no complete line — read more bytes
                match stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(anyhow::anyhow!("Stream read error: {e}")),
                            (stream, buffer),
                        ));
                    }
                    None => {
                        // Stream ended — emit remaining buffer if non-empty
                        if !buffer.trim().is_empty() {
                            let remaining = std::mem::take(&mut buffer);
                            return Some((Ok(remaining), (stream, buffer)));
                        }
                        return None;
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Ollama parsing ──────────────────────────────────

    #[test]
    fn test_parse_ollama_chunk() {
        let line = r#"{"message":{"role":"assistant","content":"The main"},"done":false}"#;
        let result = parse_ollama_line(line);
        assert_eq!(result.unwrap().unwrap(), "The main");
    }

    #[test]
    fn test_parse_ollama_done() {
        let line = r#"{"message":{"role":"assistant","content":""},"done":true}"#;
        let result = parse_ollama_line(line);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_ollama_empty_content() {
        let line = r#"{"message":{"role":"assistant","content":""},"done":false}"#;
        let result = parse_ollama_line(line);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_ollama_malformed() {
        let line = "not valid json{{{";
        let result = parse_ollama_line(line);
        assert!(result.unwrap().is_err());
    }

    // ─── OpenAI parsing ──────────────────────────────────

    #[test]
    fn test_parse_openai_data_line() {
        let line = r#"data: {"choices":[{"delta":{"content":"Hello"}}]}"#;
        let result = parse_openai_line(line);
        assert_eq!(result.unwrap().unwrap(), "Hello");
    }

    #[test]
    fn test_parse_openai_done() {
        let line = "data: [DONE]";
        let result = parse_openai_line(line);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_openai_empty_delta() {
        let line = r#"data: {"choices":[{"delta":{"content":null}}]}"#;
        let result = parse_openai_line(line);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_openai_role_only_chunk() {
        let line = r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#;
        let result = parse_openai_line(line);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_openai_malformed() {
        let line = "data: {broken json";
        let result = parse_openai_line(line);
        assert!(result.unwrap().is_err());
    }

    // ─── Edge cases ──────────────────────────────────────

    #[test]
    fn test_parse_empty_line() {
        assert!(parse_ollama_line("").is_none());
        assert!(parse_openai_line("").is_none());
    }

    #[test]
    fn test_parse_whitespace_line() {
        assert!(parse_ollama_line("   ").is_none());
        assert!(parse_openai_line("   ").is_none());
    }

    #[test]
    fn test_parse_openai_non_data_line() {
        assert!(parse_openai_line("event: message").is_none());
    }
}
