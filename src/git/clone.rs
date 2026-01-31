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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // ── is_indexable_file ─────────────────────────────────

    #[test]
    fn test_indexable_rust_file() {
        assert!(is_indexable_file(Path::new("src/main.rs")));
    }

    #[test]
    fn test_indexable_python_file() {
        assert!(is_indexable_file(Path::new("app.py")));
    }

    #[test]
    fn test_indexable_typescript_file() {
        assert!(is_indexable_file(Path::new("index.tsx")));
    }

    #[test]
    fn test_not_indexable_image() {
        assert!(!is_indexable_file(Path::new("photo.png")));
        assert!(!is_indexable_file(Path::new("logo.jpg")));
    }

    #[test]
    fn test_not_indexable_binary() {
        assert!(!is_indexable_file(Path::new("app.exe")));
        assert!(!is_indexable_file(Path::new("lib.so")));
        assert!(!is_indexable_file(Path::new("archive.zip")));
    }

    #[test]
    fn test_indexable_config_filenames() {
        assert!(is_indexable_file(Path::new("Makefile")));
        assert!(is_indexable_file(Path::new("Dockerfile")));
        assert!(is_indexable_file(Path::new("Cargo.toml")));
        assert!(is_indexable_file(Path::new("package.json")));
    }

    #[test]
    fn test_indexable_many_extensions() {
        let extensions = vec![
            "go", "java", "c", "cpp", "h", "cs", "rb", "php", "swift", "kt",
            "scala", "lua", "sh", "sql", "html", "css", "scss", "json", "yaml",
            "yml", "toml", "xml", "md", "txt", "proto", "graphql", "vue",
            "svelte", "zig", "dart", "nim", "jl", "ex", "hs", "clj", "tf", "nix",
        ];
        for ext in extensions {
            assert!(
                is_indexable_file(Path::new(&format!("file.{ext}"))),
                "Expected .{ext} to be indexable"
            );
        }
    }

    // ── detect_language ──────────────────────────────────

    #[test]
    fn test_detect_rust() {
        assert_eq!(detect_language(Path::new("main.rs")), "rust");
    }

    #[test]
    fn test_detect_python() {
        assert_eq!(detect_language(Path::new("app.py")), "python");
    }

    #[test]
    fn test_detect_javascript() {
        assert_eq!(detect_language(Path::new("index.js")), "javascript");
    }

    #[test]
    fn test_detect_typescript() {
        assert_eq!(detect_language(Path::new("app.ts")), "typescript");
        assert_eq!(detect_language(Path::new("comp.tsx")), "tsx");
    }

    #[test]
    fn test_detect_go() {
        assert_eq!(detect_language(Path::new("main.go")), "go");
    }

    #[test]
    fn test_detect_cpp_variants() {
        assert_eq!(detect_language(Path::new("foo.cpp")), "cpp");
        assert_eq!(detect_language(Path::new("foo.cc")), "cpp");
        assert_eq!(detect_language(Path::new("foo.h")), "cpp");
        assert_eq!(detect_language(Path::new("foo.hpp")), "cpp");
    }

    #[test]
    fn test_detect_shell() {
        assert_eq!(detect_language(Path::new("run.sh")), "shell");
        assert_eq!(detect_language(Path::new("init.bash")), "shell");
        assert_eq!(detect_language(Path::new("config.zsh")), "shell");
    }

    #[test]
    fn test_detect_unknown_extension() {
        assert_eq!(detect_language(Path::new("file.xyz")), "text");
    }

    #[test]
    fn test_detect_no_extension() {
        assert_eq!(detect_language(Path::new("somefile")), "text");
    }

    // ── walk_repo_files ──────────────────────────────────
    // Note: tempfile creates dirs named `.tmpXXXXXX` which triggers
    // the hidden-dir filter. We use a "repo" subdirectory as the walk root.

    fn make_repo_dir() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("repo");
        fs::create_dir_all(&root).unwrap();
        (dir, root)
    }

    #[test]
    fn test_walk_repo_files_basic() {
        let (_dir, root) = make_repo_dir();

        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn hello() {}").unwrap();
        fs::write(root.join("README.md"), "# Hello").unwrap();

        let files = walk_repo_files(&root);
        assert_eq!(files.len(), 3);

        let paths: Vec<&str> = files.iter().map(|f| f.relative_path.as_str()).collect();
        assert!(paths.contains(&"src/main.rs"));
        assert!(paths.contains(&"src/lib.rs"));
    }

    #[test]
    fn test_walk_repo_files_skips_hidden_dirs() {
        let (_dir, root) = make_repo_dir();

        fs::create_dir_all(root.join(".git")).unwrap();
        fs::write(root.join(".git/config"), "git config data").unwrap();
        fs::write(root.join("visible.rs"), "fn visible() {}").unwrap();

        let files = walk_repo_files(&root);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative_path, "visible.rs");
    }

    #[test]
    fn test_walk_repo_files_skips_node_modules() {
        let (_dir, root) = make_repo_dir();

        fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        fs::write(root.join("node_modules/pkg/index.js"), "// dep").unwrap();
        fs::write(root.join("index.js"), "// app").unwrap();

        let files = walk_repo_files(&root);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative_path, "index.js");
    }

    #[test]
    fn test_walk_repo_files_skips_binary_extensions() {
        let (_dir, root) = make_repo_dir();

        fs::write(root.join("app.rs"), "fn main() {}").unwrap();
        fs::write(root.join("image.png"), "fake png data").unwrap();
        fs::write(root.join("archive.zip"), "fake zip data").unwrap();

        let files = walk_repo_files(&root);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative_path, "app.rs");
    }

    #[test]
    fn test_walk_repo_files_detects_language() {
        let (_dir, root) = make_repo_dir();

        fs::write(root.join("server.py"), "print('hi')").unwrap();

        let files = walk_repo_files(&root);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].language, "python");
    }

    #[test]
    fn test_walk_empty_directory() {
        let (_dir, root) = make_repo_dir();
        let files = walk_repo_files(&root);
        assert!(files.is_empty());
    }
}
