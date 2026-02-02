//! AST-aware code chunking using tree-sitter.
//!
//! Implements a simplified cAST (EMNLP 2025) algorithm:
//! 1. Parse source into AST
//! 2. Walk top-level nodes, splitting at syntactic boundaries
//! 3. If a single node exceeds the budget, recurse into its children
//! 4. Merge adjacent small nodes up to the budget
//!
//! Falls back to line-based chunking when:
//! - File exceeds MAX_FILE_SIZE (500 KB)
//! - Parse produces >30% error nodes
//! - Language is not supported

use super::{ChunkOutput, Language};

/// Maximum non-whitespace characters per chunk.
const CHAR_BUDGET: usize = 1500;

/// Files larger than this skip AST parsing entirely.
const MAX_FILE_SIZE: usize = 500 * 1024; // 500 KB

/// If more than this fraction of AST nodes are error nodes, fall back.
const ERROR_THRESHOLD: f64 = 0.30;

/// Chunk source code using AST-aware splitting for supported languages.
/// Returns None if the language is unsupported or AST parsing should be skipped.
#[allow(unused_assignments)] // current_chars reset is needed for loop correctness
pub fn chunk_with_ast(content: &str, language: Language) -> Option<Vec<ChunkOutput>> {
    if content.len() > MAX_FILE_SIZE {
        return None;
    }

    let mut parser = tree_sitter::Parser::new();
    let ts_language = language.tree_sitter_language()?;
    parser.set_language(&ts_language).ok()?;

    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    // Check error rate
    let (total, errors) = count_nodes(root);
    if total > 0 && (errors as f64 / total as f64) > ERROR_THRESHOLD {
        tracing::warn!(
            "AST error rate {:.0}% exceeds threshold, falling back to line-based chunking",
            (errors as f64 / total as f64) * 100.0
        );
        return None;
    }

    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    let mut current_lines: Vec<usize> = Vec::new(); // line indices
    let mut current_chars: usize = 0;

    // Walk top-level children of root
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        let node_lines = child.start_position().row..=child.end_position().row;
        let node_chars = non_ws_chars_in_range(&lines, *node_lines.start(), *node_lines.end());

        if node_chars > CHAR_BUDGET {
            // Flush current accumulator first
            if !current_lines.is_empty() {
                chunks.push(make_chunk(&lines, &current_lines));
                current_lines.clear();
                current_chars = 0;
            }
            // Recurse into this large node's children
            split_large_node(&lines, child, &mut chunks);
        } else if current_chars + node_chars > CHAR_BUDGET && !current_lines.is_empty() {
            // Adding this node would exceed budget — flush and start new chunk
            chunks.push(make_chunk(&lines, &current_lines));
            current_lines.clear();
            current_chars = 0;

            // Start new accumulator with this node
            for row in *node_lines.start()..=*node_lines.end() {
                current_lines.push(row);
            }
            current_chars = node_chars;
        } else {
            // Merge into current chunk
            // Fill any gap lines between previous content and this node
            let start = if let Some(&last) = current_lines.last() {
                last + 1
            } else {
                *node_lines.start()
            };
            for row in start..=*node_lines.end() {
                current_lines.push(row);
            }
            current_chars += node_chars;
        }
    }

    // Flush remaining
    if !current_lines.is_empty() {
        chunks.push(make_chunk(&lines, &current_lines));
    }

    // If AST produced nothing (e.g. empty file parsed by tree-sitter), return None
    // to let the fallback handle it.
    if chunks.is_empty() {
        return None;
    }

    Some(chunks)
}

/// Recursively split a node that exceeds the budget.
#[allow(unused_assignments)]
fn split_large_node(
    lines: &[&str],
    node: tree_sitter::Node,
    chunks: &mut Vec<ChunkOutput>,
) {
    let mut cursor = node.walk();
    let children: Vec<tree_sitter::Node> = node.children(&mut cursor).collect();

    if children.is_empty() {
        // Leaf node too large — just emit it as-is
        let node_lines: Vec<usize> = (node.start_position().row..=node.end_position().row).collect();
        chunks.push(make_chunk(lines, &node_lines));
        return;
    }

    let mut current_lines: Vec<usize> = Vec::new();
    let mut current_chars: usize = 0;

    for child in children {
        let child_lines = child.start_position().row..=child.end_position().row;
        let child_chars = non_ws_chars_in_range(lines, *child_lines.start(), *child_lines.end());

        if child_chars > CHAR_BUDGET {
            // Flush and recurse
            if !current_lines.is_empty() {
                chunks.push(make_chunk(lines, &current_lines));
                current_lines.clear();
                current_chars = 0;
            }
            split_large_node(lines, child, chunks);
        } else if current_chars + child_chars > CHAR_BUDGET && !current_lines.is_empty() {
            chunks.push(make_chunk(lines, &current_lines));
            current_lines.clear();
            current_chars = 0;

            for row in *child_lines.start()..=*child_lines.end() {
                current_lines.push(row);
            }
            current_chars = child_chars;
        } else {
            let start = if let Some(&last) = current_lines.last() {
                last + 1
            } else {
                *child_lines.start()
            };
            for row in start..=*child_lines.end() {
                current_lines.push(row);
            }
            current_chars += child_chars;
        }
    }

    if !current_lines.is_empty() {
        chunks.push(make_chunk(lines, &current_lines));
    }
}

/// Build a ChunkOutput from a set of line indices.
fn make_chunk(lines: &[&str], line_indices: &[usize]) -> ChunkOutput {
    let start_line = line_indices.first().copied().unwrap_or(0);
    let end_line = line_indices.last().copied().unwrap_or(0);

    let content: String = (start_line..=end_line)
        .filter_map(|i| lines.get(i))
        .cloned()
        .collect::<Vec<&str>>()
        .join("\n");

    ChunkOutput {
        content,
        start_line: start_line + 1, // 1-based
        end_line: end_line + 1,     // 1-based
    }
}

/// Count non-whitespace characters in a range of lines.
fn non_ws_chars_in_range(lines: &[&str], start: usize, end: usize) -> usize {
    (start..=end)
        .filter_map(|i| lines.get(i))
        .flat_map(|line| line.chars())
        .filter(|c| !c.is_whitespace())
        .count()
}

/// Count total nodes and error nodes in a tree.
fn count_nodes(node: tree_sitter::Node) -> (usize, usize) {
    let mut total = 1usize;
    let mut errors = if node.is_error() { 1usize } else { 0 };

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let (t, e) = count_nodes(child);
        total += t;
        errors += e;
    }

    (total, errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_rust_functions() {
        let source = r#"
fn hello() {
    println!("hello");
}

fn world() {
    println!("world");
}
"#;
        let chunks = chunk_with_ast(source, Language::Rust).unwrap();
        assert!(!chunks.is_empty());
        // Both functions are small — should be merged into one chunk
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("fn hello"));
        assert!(chunks[0].content.contains("fn world"));
    }

    #[test]
    fn test_chunk_rust_large_function_splits() {
        // Create a function with ~2000 non-whitespace chars (exceeds 1500 budget)
        let body: String = (0..150)
            .map(|i| format!("    let var_{i} = {i};\n"))
            .collect();
        let source = format!("fn big() {{\n{body}}}\n\nfn small() {{}}\n");

        let chunks = chunk_with_ast(&source, Language::Rust).unwrap();
        assert!(chunks.len() >= 2, "Large function should be split: got {} chunks", chunks.len());
    }

    #[test]
    fn test_chunk_rust_preserves_all_content() {
        let source = r#"use std::io;

struct Foo {
    bar: i32,
}

impl Foo {
    fn new() -> Self {
        Foo { bar: 0 }
    }
}

fn main() {
    let f = Foo::new();
    println!("{}", f.bar);
}
"#;
        let chunks = chunk_with_ast(source, Language::Rust).unwrap();
        let reassembled: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        // All key elements should be present
        assert!(reassembled.contains("use std::io"));
        assert!(reassembled.contains("struct Foo"));
        assert!(reassembled.contains("fn new"));
        assert!(reassembled.contains("fn main"));
    }

    #[test]
    fn test_chunk_skips_large_files() {
        let large = "x".repeat(MAX_FILE_SIZE + 1);
        assert!(chunk_with_ast(&large, Language::Rust).is_none());
    }

    #[test]
    fn test_chunk_unsupported_language_returns_none() {
        assert!(chunk_with_ast("some text", Language::Unknown).is_none());
    }

    #[test]
    fn test_line_numbers_are_one_based() {
        let source = "fn a() {}\nfn b() {}\n";
        let chunks = chunk_with_ast(source, Language::Rust).unwrap();
        assert_eq!(chunks[0].start_line, 1);
    }

    // ── JavaScript tests ────────────────────────────────

    #[test]
    fn test_chunk_javascript_functions() {
        let source = r#"
function hello() {
    console.log("hello");
}

function world() {
    console.log("world");
}
"#;
        let chunks = chunk_with_ast(source, Language::JavaScript).unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks[0].content.contains("function hello"));
        assert!(chunks[0].content.contains("function world"));
    }

    #[test]
    fn test_chunk_javascript_class() {
        let source = r#"
class MyComponent {
    constructor(name) {
        this.name = name;
    }

    render() {
        return `<div>${this.name}</div>`;
    }
}

function standalone() {
    return 42;
}
"#;
        let chunks = chunk_with_ast(source, Language::JavaScript).unwrap();
        assert!(!chunks.is_empty());
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("class MyComponent"));
        assert!(all_content.contains("function standalone"));
    }

    // ── TypeScript tests ────────────────────────────────

    #[test]
    fn test_chunk_typescript() {
        let source = r#"
interface Config {
    name: string;
    value: number;
}

function processConfig(config: Config): string {
    return config.name;
}

export class Service {
    private config: Config;

    constructor(config: Config) {
        this.config = config;
    }

    process(): string {
        return processConfig(this.config);
    }
}
"#;
        let chunks = chunk_with_ast(source, Language::TypeScript).unwrap();
        assert!(!chunks.is_empty());
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("interface Config"));
        assert!(all_content.contains("function processConfig"));
        assert!(all_content.contains("class Service"));
    }

    #[test]
    fn test_chunk_tsx() {
        let source = r#"
import React from 'react';

interface Props {
    name: string;
}

function Greeting({ name }: Props) {
    return <div>Hello, {name}!</div>;
}

export default Greeting;
"#;
        let chunks = chunk_with_ast(source, Language::Tsx).unwrap();
        assert!(!chunks.is_empty());
        let all_content: String = chunks.iter().map(|c| c.content.as_str()).collect::<Vec<_>>().join("\n");
        assert!(all_content.contains("interface Props"));
        assert!(all_content.contains("function Greeting"));
    }
}
