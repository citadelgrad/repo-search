//! Code chunking module: AST-aware for supported languages, line-based fallback for the rest.

pub mod ast;
pub mod fallback;

/// Output of the chunking process.
#[derive(Debug, Clone)]
pub struct ChunkOutput {
    pub content: String,
    /// 1-based start line in the original file.
    pub start_line: usize,
    /// 1-based end line in the original file.
    pub end_line: usize,
}

/// Languages with tree-sitter AST support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    JavaScript,
    TypeScript,
    Tsx,
    Unknown,
}

impl Language {
    /// Map a file extension / language name string to a Language variant.
    pub fn from_str(lang: &str) -> Self {
        match lang.to_lowercase().as_str() {
            "rust" | "rs" => Language::Rust,
            "javascript" | "js" | "jsx" => Language::JavaScript,
            "typescript" | "ts" => Language::TypeScript,
            "tsx" => Language::Tsx,
            _ => Language::Unknown,
        }
    }

    /// Return the tree-sitter Language for this variant, or None if unsupported.
    pub fn tree_sitter_language(&self) -> Option<tree_sitter::Language> {
        match self {
            Language::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
            Language::JavaScript => Some(tree_sitter_javascript::LANGUAGE.into()),
            Language::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            Language::Tsx => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
            Language::Unknown => None,
        }
    }
}

/// Chunk source code for the given language.
/// Tries AST-aware chunking first, falls back to line-based.
pub fn chunk_code(content: &str, language_str: &str) -> Vec<ChunkOutput> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    let language = Language::from_str(language_str);

    // Try AST chunking for supported languages
    if let Some(chunks) = ast::chunk_with_ast(content, language) {
        return chunks;
    }

    // Fallback to line-based
    fallback::chunk_fallback(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_from_str() {
        assert_eq!(Language::from_str("rust"), Language::Rust);
        assert_eq!(Language::from_str("rs"), Language::Rust);
        assert_eq!(Language::from_str("javascript"), Language::JavaScript);
        assert_eq!(Language::from_str("js"), Language::JavaScript);
        assert_eq!(Language::from_str("typescript"), Language::TypeScript);
        assert_eq!(Language::from_str("ts"), Language::TypeScript);
        assert_eq!(Language::from_str("tsx"), Language::Tsx);
        assert_eq!(Language::from_str("python"), Language::Unknown);
    }

    #[test]
    fn test_chunk_code_empty() {
        assert!(chunk_code("", "rust").is_empty());
        assert!(chunk_code("  \n  ", "rust").is_empty());
    }

    #[test]
    fn test_chunk_code_rust_uses_ast() {
        let source = "fn hello() {\n    println!(\"hi\");\n}\n";
        let chunks = chunk_code(source, "rust");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("fn hello"));
    }

    #[test]
    fn test_chunk_code_unknown_uses_fallback() {
        let source = "some plain text\nmore text\n";
        let chunks = chunk_code(source, "markdown");
        assert_eq!(chunks.len(), 1);
    }
}
