---
title: "feat: Native repo sync (git pull + full re-index)"
type: feat
date: 2026-01-31
reviewed: 2026-01-31
---

# Native Repo Sync (git pull + full re-index)

## Overview

Add native git-pull and re-indexing so repos already added to repo-search stay up to date. Instead of requiring users to delete and re-add a repo when its upstream changes, they can trigger a sync that fetches new commits and re-indexes the repo.

Inspired by [repo_updater (ru)](https://github.com/Dicklesworthstone/repo_updater). Rather than adding `ru` as an external dependency, we implement sync natively in Rust using the existing `git2` crate -- consistent with the current clone pipeline and avoiding new system dependencies.

## Problem Statement

Once a repo is cloned and indexed, its search index becomes stale as upstream evolves. The only way to refresh is delete + re-add, which requires a full re-clone, loses embedding vectors, and is tedious for many repos.

## Proposed Solution

1. **Git fetch + hard reset** using `git2` (always `reset --hard origin/HEAD` -- repos are read-only mirrors)
2. **Full re-index**: delete all BM25 + vector entries for the repo, re-walk, re-chunk, re-index
3. **One new API endpoint**: `POST /api/repos/{id}/sync`
4. **Frontend**: per-repo Sync button

**HEAD comparison optimization**: Compare old HEAD to new HEAD after fetch. If unchanged, skip re-indexing entirely.

## Prerequisites (Separate PRs)

### Security: Symlink Path Traversal

`walk_repo_files()` in `src/git/clone.rs:44` and `dir_size_bytes()` in `src/git/clone.rs:270` use `WalkDir` which follows symlinks by default. A malicious repo could symlink to host files (e.g., `src/secrets.rs -> ~/.ssh/id_rsa`), which then get indexed and returned in search results.

**Fix** (separate PR, applies to existing clone path too):

```rust
// walk_repo_files
for entry in WalkDir::new(repo_dir)
    .follow_links(false)  // ADD THIS
    .into_iter()
    .filter_entry(|e| !is_hidden_or_ignored(e))
    .filter_map(|e| e.ok())
{
    if entry.path_is_symlink() { continue; }  // ADD THIS (defense in depth)
    if !entry.file_type().is_file() { continue; }
    // ...
}

// dir_size_bytes -- also add follow_links(false)
pub fn dir_size_bytes(path: &Path) -> u64 {
    WalkDir::new(path)
        .follow_links(false)  // ADD THIS
        .into_iter()
        // ...
}
```

### Security: SSRF via Git Remote URL Mutation

`git fetch` reads the remote URL from `.git/config`, not from our stored `Repo.url`. After clone-time URL validation, `.git/config` could be modified to `file:///` or `ssh://attacker.com/`, bypassing SSRF checks.

**Fix**: In `pull_repo()`, set the remote URL from the stored `Repo.url` before fetching:

```rust
repo.remote_set_url("origin", &stored_url)?;
```

## Technical Approach

### Git Operations

**git2 fetch + reset pattern** (from [official pull.rs example](https://github.com/rust-lang/git2-rs/blob/master/examples/pull.rs)):

```rust
pub fn pull_repo(repo_dir: &Path, stored_url: &str, git_token: Option<&str>) -> Result<String> {
    let repo = git2::Repository::open(repo_dir)?;

    // SSRF prevention: always set remote URL from our stored value
    repo.remote_set_url("origin", stored_url)?;

    let mut remote = repo.find_remote("origin")?;
    let mut fo = git2::FetchOptions::new();
    if let Some(token) = git_token {
        let token = token.to_string();
        let mut callbacks = git2::RemoteCallbacks::new();
        callbacks.credentials(move |_url, _username, _allowed| {
            git2::Cred::userpass_plaintext("x-access-token", &token)
        });
        fo.remote_callbacks(callbacks);
    }
    remote.fetch(&["HEAD"], Some(&mut fo), None)?;

    let fetch_head = repo.find_reference("FETCH_HEAD")?;
    let fetch_commit = repo.reference_to_annotated_commit(&fetch_head)?;
    let new_head = fetch_commit.id().to_string();

    let object = repo.find_object(fetch_commit.id(), None)?;
    repo.reset(&object, git2::ResetType::Hard, None)?;

    Ok(new_head)
}
```

**Why always hard-reset**: A single `reset --hard` handles normal updates, force pushes, rebases, and branch switches. No benefit to trying fast-forward first on read-only clones.

**Submodules**: Not handled (current clone doesn't handle them either).

### Sync Pipeline

```
git fetch + hard reset -> new_head
       |
   old_head == new_head?  (if old_head is None, always re-index)
      / \
    yes   no
     |     |
  no-op   delete all BM25 chunks for repo (sub-second)
           |
         delete all vector entries for repo
           |
         walk_repo_files() (existing function)
           |
         chunk_file() for each file (existing function)
           |
         index_chunks() into BM25 (existing method)
           |
         embed_batch() + add_chunks() into vector store (existing methods)
           |
         update repo metadata
         (head_commit, indexed_at, file_count, status -> Ready)
```

Reuses 100% of existing code paths. The only new function is `pull_repo()` (~25 lines).

**Search gap during re-index**: Between delete-all and re-index-complete, the repo has no search results. BM25 delete+reindex is sub-second. Vector embedding is the slow path (80-400s for large repos). Acceptable for v1 since syncs are user-initiated.

### Concurrency & State

- **Semaphore**: Reuse existing `clone_semaphore` (2 permits) for the entire sync pipeline
- **Status transitions**: `Ready` -> `Indexing` -> `Ready` (success) or `Error(msg)` (failure). Reuse `Indexing` -- the frontend already handles it with a spinner.
- **Concurrent sync guard**: If repo is already `Indexing`, `Cloning`, or `Embedding`, return 409 Conflict
- **Search during sync**: Allowed (empty results during the gap)
- **Delete during sync**: Allowed (background task checks repo existence before each step)
- **Sync from Error state**: Allowed if the repo directory still exists on disk
- **Sync timeout**: Apply `clone_timeout_secs` (300s)

### Failure & Rollback

- **Git fetch fails**: Set `Error("Fetch failed: {details}")`. Directory and index preserved (no deletions yet). User can retry.
- **Re-indexing fails after delete**: Index empty but on-disk files intact. Set `Error("Re-index failed: {details}")`. User can retry.
- **Never delete repo directory on sync failure** -- use `update_repo_status` directly, NOT `set_error_and_cleanup`.
- **Sync on repo with missing directory**: Return error suggesting user delete and re-add the repo.

### Data Model Changes

```rust
// src/models.rs -- add one field

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Repo {
    pub id: Uuid,
    pub url: String,
    pub name: String,
    pub status: RepoStatus,
    pub added_at: DateTime<Utc>,
    pub indexed_at: Option<DateTime<Utc>>,
    pub file_count: usize,
    #[serde(default)]
    pub head_commit: Option<String>,  // NEW: SHA of HEAD after clone/sync
}
```

`#[serde(default)]` ensures backward compatibility with existing `repos.json` files.

### API Endpoint

**POST /api/repos/{id}/sync**
- Validates repo exists and status is `Ready` or `Error` (with directory present)
- Sets status to `Indexing`, spawns background task
- Returns `200 OK` with full `Repo` JSON (consistent with `add_repo`)
- Returns `409 Conflict` if already `Indexing`, `Cloning`, or `Embedding`
- Returns `404` if repo not found
- Returns `400` if repo is in `Error` state with missing directory

### Frontend Changes

- **Per-repo Sync button**: Visible when status is `Ready` or `Error`. Disabled when `Indexing`/`Cloning`/`Embedding`. Calls `POST /api/repos/{id}/sync`.
- **Status indicator**: Existing `Indexing` spinner already works.
- **Last indexed display**: Show `indexed_at` relative time (e.g., "Indexed 2 hours ago") next to each repo.

## Acceptance Criteria

- [ ] `POST /api/repos/{id}/sync` fetches latest commits and re-indexes the repo
- [ ] Repos with no upstream changes (HEAD unchanged) return quickly with no re-indexing
- [ ] Force-pushed branches are handled via hard reset
- [ ] Failed syncs preserve the repo directory and return a clear error
- [ ] Syncing a repo with missing directory returns a clear error
- [ ] `Repo` model includes `head_commit` field
- [ ] `head_commit` is set during initial clone and during sync
- [ ] Sync button is present in the UI
- [ ] Private repos (using `git_token`) can be synced
- [ ] Concurrent sync requests to the same repo return 409
- [ ] Existing `repos.json` files without `head_commit` load correctly

## Dependencies & Risks

**Dependencies:**
- `git2` 0.20 -- already in use; fetch/reset APIs available
- No new dependencies for this feature

**Risks:**
- **Embedding cost on full re-index**: For large repos, re-embedding all chunks is the primary bottleneck. Mitigated by HEAD comparison (skip if no changes). Selective re-indexing would address this for incremental updates (deferred).
- **Brief search gap during re-index**: Between delete-all and re-index-complete, the repo has no search results. Acceptable for v1 since syncs are user-initiated.

## Implementation Sequence

1. Add `head_commit: Option<String>` to `Repo` model with `#[serde(default)]`
2. Capture HEAD after clone in `clone_and_index` (`src/api/repos.rs`)
3. Add `pull_repo()` function in `src/git/clone.rs` (~25 lines)
4. Add `sync_repo` handler in `src/api/repos.rs`:
   - Validate state, set `Indexing`, spawn background task
   - Pull, compare HEAD, skip if unchanged
   - `bm25.delete_repo()` + `vectors.delete_repo()`, then re-walk, re-chunk, re-index
   - Update repo metadata (`head_commit`, `indexed_at`, `file_count`, status -> `Ready`)
5. Wire `POST /api/repos/{id}/sync` route in `src/main.rs`
6. Add Sync button to frontend
7. Tests: pull logic, HEAD comparison, sync handler

## References

### Internal
- Clone auth pattern: `src/git/clone.rs:22-31`
- BM25 delete_repo: `src/search/bm25.rs:106`
- Vector delete_repo: `src/search/vector.rs`
- Repo model: `src/models.rs`
- Clone semaphore: `src/state.rs`
- Reindex async pattern: `src/api/repos.rs:186`
- Error cleanup (do NOT follow for sync): `src/api/repos.rs:474`

### External
- [repo_updater (ru)](https://github.com/Dicklesworthstone/repo_updater) -- inspiration
- [git2 pull.rs example](https://github.com/rust-lang/git2-rs/blob/master/examples/pull.rs)
- [git2 fetch.rs example](https://github.com/rust-lang/git2-rs/blob/master/examples/fetch.rs)

### Deferred Work
- Selective re-indexing with per-file diff detection (requires BM25 schema migration)
- Background auto-sync on configurable interval
- `POST /api/repos/sync-all` bulk endpoint
- Atomic persistence for `vectors.json` and `repos.json`
- `RepoStatus::Syncing` status variant
- Chunk content hashing (skip re-embedding unchanged chunks)
