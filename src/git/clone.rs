use anyhow::{Context, Result};
use std::path::Path;
use walkdir::WalkDir;

/// A file extracted from a repo
#[derive(Debug, Clone)]
pub struct RepoFile {
    pub relative_path: String,
    pub content: String,
    pub language: String,
}

/// Clone a git repository to the target directory.
pub fn clone_repo(url: &str, target: &Path) -> Result<()> {
    tracing::info!("Cloning {} into {}", url, target.display());
    git2::Repository::clone(url, target)
        .with_context(|| format!("Failed to clone {url}"))?;
    tracing::info!("Clone complete: {}", target.display());
    Ok(())
}

/// Walk all text files in a cloned repo and return their contents.
pub fn walk_repo_files(repo_dir: &Path) -> Vec<RepoFile> {
    let mut files = Vec::new();

    for entry in WalkDir::new(repo_dir)
        .into_iter()
        .filter_entry(|e| !is_hidden_or_ignored(e))
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();

        // Skip binary / non-text files
        if !is_indexable_file(path) {
            continue;
        }

        // Skip very large files (>1MB)
        if let Ok(meta) = std::fs::metadata(path) {
            if meta.len() > 1_048_576 {
                continue;
            }
        }

        let relative = path
            .strip_prefix(repo_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        match std::fs::read_to_string(path) {
            Ok(content) => {
                let language = detect_language(path);
                files.push(RepoFile {
                    relative_path: relative,
                    content,
                    language,
                });
            }
            Err(_) => {
                // Skip files that can't be read as UTF-8
                continue;
            }
        }
    }

    files
}

fn is_hidden_or_ignored(entry: &walkdir::DirEntry) -> bool {
    let name = entry.file_name().to_string_lossy();
    if name.starts_with('.') {
        return true;
    }
    // Skip common non-code directories
    matches!(
        name.as_ref(),
        "node_modules"
            | "target"
            | "dist"
            | "build"
            | "__pycache__"
            | ".git"
            | "vendor"
            | "venv"
            | ".venv"
            | "env"
    )
}

fn is_indexable_file(path: &Path) -> bool {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    // File name checks for common config files without extensions
    let filename = path
        .file_name()
        .map(|f| f.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    if matches!(
        filename.as_ref(),
        "makefile"
            | "dockerfile"
            | "rakefile"
            | "gemfile"
            | "cmakelists.txt"
            | "cargo.toml"
            | "cargo.lock"
            | "package.json"
            | "tsconfig.json"
            | "readme"
            | "license"
    ) {
        return true;
    }

    matches!(
        ext.as_str(),
        "rs" | "py"
            | "js"
            | "ts"
            | "tsx"
            | "jsx"
            | "go"
            | "java"
            | "c"
            | "cpp"
            | "cc"
            | "h"
            | "hpp"
            | "cs"
            | "rb"
            | "php"
            | "swift"
            | "kt"
            | "kts"
            | "scala"
            | "r"
            | "lua"
            | "sh"
            | "bash"
            | "zsh"
            | "fish"
            | "ps1"
            | "bat"
            | "cmd"
            | "sql"
            | "html"
            | "css"
            | "scss"
            | "less"
            | "xml"
            | "json"
            | "yaml"
            | "yml"
            | "toml"
            | "ini"
            | "cfg"
            | "conf"
            | "md"
            | "rst"
            | "txt"
            | "tex"
            | "proto"
            | "graphql"
            | "gql"
            | "vue"
            | "svelte"
            | "ex"
            | "exs"
            | "erl"
            | "hs"
            | "ml"
            | "mli"
            | "clj"
            | "cljs"
            | "el"
            | "vim"
            | "tf"
            | "hcl"
            | "nix"
            | "zig"
            | "dart"
            | "v"
            | "nim"
            | "cr"
            | "jl"
    )
}

fn detect_language(path: &Path) -> String {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "rs" => "rust",
        "py" => "python",
        "js" => "javascript",
        "ts" => "typescript",
        "tsx" => "tsx",
        "jsx" => "jsx",
        "go" => "go",
        "java" => "java",
        "c" => "c",
        "cpp" | "cc" => "cpp",
        "h" | "hpp" => "cpp",
        "cs" => "csharp",
        "rb" => "ruby",
        "php" => "php",
        "swift" => "swift",
        "kt" | "kts" => "kotlin",
        "scala" => "scala",
        "r" => "r",
        "lua" => "lua",
        "sh" | "bash" | "zsh" | "fish" => "shell",
        "sql" => "sql",
        "html" => "html",
        "css" | "scss" | "less" => "css",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "toml" => "toml",
        "xml" => "xml",
        "md" | "rst" | "txt" => "text",
        "proto" => "protobuf",
        "graphql" | "gql" => "graphql",
        "vue" => "vue",
        "svelte" => "svelte",
        "zig" => "zig",
        "dart" => "dart",
        "nim" => "nim",
        "jl" => "julia",
        "ex" | "exs" => "elixir",
        "hs" => "haskell",
        "clj" | "cljs" => "clojure",
        "tf" | "hcl" => "hcl",
        "nix" => "nix",
        _ => "text",
    }
    .to_string()
}
