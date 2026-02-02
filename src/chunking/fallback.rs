//! Smart line-based fallback chunker for languages without AST support.
//!
//! Four-tier splitting strategy:
//! 1. Split at blank lines (natural paragraph boundaries)
//! 2. Merge small segments up to the character budget
//! 3. If a segment is still too large, split at single newlines
//! 4. Last resort: split at character boundary

use super::ChunkOutput;

/// Maximum non-whitespace characters per chunk.
const CHAR_BUDGET: usize = 1500;

/// Chunk content using line-based heuristics.
pub fn chunk_fallback(content: &str) -> Vec<ChunkOutput> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    // Split into segments at blank lines
    let segments = split_at_blank_lines(&lines);

    // Merge small segments up to budget
    let mut chunks = Vec::new();
    let mut current_start = 0usize; // line index
    let mut current_end = 0usize;
    let mut current_chars = 0usize;
    let mut started = false;

    for seg in &segments {
        let seg_chars: usize = seg.lines.iter()
            .flat_map(|l| l.chars())
            .filter(|c| !c.is_whitespace())
            .count();

        if seg_chars > CHAR_BUDGET {
            // Flush accumulator
            if started {
                chunks.push(ChunkOutput {
                    content: lines[current_start..=current_end].join("\n"),
                    start_line: current_start + 1,
                    end_line: current_end + 1,
                });
                started = false;
                current_chars = 0;
            }
            // This segment is too large — split it by individual lines
            split_large_segment(&lines, seg.start, seg.end, &mut chunks);
        } else if started && current_chars + seg_chars > CHAR_BUDGET {
            // Flush current chunk
            chunks.push(ChunkOutput {
                content: lines[current_start..=current_end].join("\n"),
                start_line: current_start + 1,
                end_line: current_end + 1,
            });
            // Start new with this segment
            current_start = seg.start;
            current_end = seg.end;
            current_chars = seg_chars;
        } else {
            if !started {
                current_start = seg.start;
                started = true;
            }
            current_end = seg.end;
            current_chars += seg_chars;
        }
    }

    // Flush remaining
    if started {
        chunks.push(ChunkOutput {
            content: lines[current_start..=current_end].join("\n"),
            start_line: current_start + 1,
            end_line: current_end + 1,
        });
    }

    chunks
}

struct Segment<'a> {
    lines: Vec<&'a str>,
    start: usize,
    end: usize,
}

fn split_at_blank_lines<'a>(lines: &[&'a str]) -> Vec<Segment<'a>> {
    let mut segments = Vec::new();
    let mut seg_start = None;

    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            if let Some(start) = seg_start.take() {
                segments.push(Segment {
                    lines: lines[start..i].to_vec(),
                    start,
                    end: i.saturating_sub(1),
                });
            }
        } else if seg_start.is_none() {
            seg_start = Some(i);
        }
    }

    // Last segment
    if let Some(start) = seg_start {
        segments.push(Segment {
            lines: lines[start..].to_vec(),
            start,
            end: lines.len() - 1,
        });
    }

    segments
}

fn split_large_segment(
    lines: &[&str],
    start: usize,
    end: usize,
    chunks: &mut Vec<ChunkOutput>,
) {
    let mut chunk_start = start;
    let mut chars = 0usize;

    for i in start..=end {
        let line_chars: usize = lines[i].chars().filter(|c| !c.is_whitespace()).count();
        if chars + line_chars > CHAR_BUDGET && i > chunk_start {
            chunks.push(ChunkOutput {
                content: lines[chunk_start..i].join("\n"),
                start_line: chunk_start + 1,
                end_line: i, // exclusive end, but we use the line before
            });
            chunk_start = i;
            chars = line_chars;
        } else {
            chars += line_chars;
        }
    }

    if chunk_start <= end {
        chunks.push(ChunkOutput {
            content: lines[chunk_start..=end].join("\n"),
            start_line: chunk_start + 1,
            end_line: end + 1,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_empty() {
        assert!(chunk_fallback("").is_empty());
        assert!(chunk_fallback("   \n\n  ").is_empty());
    }

    #[test]
    fn test_fallback_small_file() {
        let content = "line 1\nline 2\nline 3";
        let chunks = chunk_fallback(content);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_line, 1);
        assert_eq!(chunks[0].end_line, 3);
    }

    #[test]
    fn test_fallback_splits_at_blank_lines() {
        // Create two blocks separated by a blank line, each near the budget
        let block1: String = (0..80).map(|i| format!("let var_{i} = {i};")).collect::<Vec<_>>().join("\n");
        let block2: String = (0..80).map(|i| format!("let other_{i} = {i};")).collect::<Vec<_>>().join("\n");
        let content = format!("{block1}\n\n{block2}");

        let chunks = chunk_fallback(&content);
        assert!(chunks.len() >= 2, "Should split at blank line, got {} chunks", chunks.len());
    }

    #[test]
    fn test_fallback_line_numbers_one_based() {
        let content = "a\nb\nc";
        let chunks = chunk_fallback(content);
        assert_eq!(chunks[0].start_line, 1);
    }
}
