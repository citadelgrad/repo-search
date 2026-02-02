---
title: "feat: Add repo pinning and drag-and-drop reorder"
type: feat
date: 2026-02-01
---

# feat: Add repo pinning and drag-and-drop reorder

## Overview

Add the ability to pin repos to the top of the repos list and manually reorder repos via drag-and-drop. A sort dropdown provides predefined sorting (name, date added, file count). Pinned repos always float above unpinned repos regardless of sort. All state persists server-side in `repos.json`.

## Problem Statement

Repos currently display in insertion order with no way to organize them. Users with many repos cannot prioritize frequently-used repos or group them meaningfully.

## Proposed Solution

### Data Model

Add a `pinned: bool` field to the `Repo` struct. Use `#[serde(default)]` for backward compatibility with existing `repos.json` data (defaults to `false`). Do NOT add a `sort_order` field -- rely on Vec position in `repos.json` as the canonical manual order (current behavior, just made explicit).

```rust
// src/models.rs
pub struct Repo {
    // ...existing fields...
    #[serde(default)]
    pub pinned: bool,
}
```

### API Endpoints

Two new endpoints:

**`PATCH /api/repos/{id}/pin`** -- Toggle pin status

- Request: empty body (toggle) or `{ "pinned": true/false }`
- When pinning: move repo to end of pinned group in the Vec
- When unpinning: move repo to top of unpinned group in the Vec
- Response: `200 OK` with updated `Repo` JSON
- Persist immediately

**`PUT /api/repos/order`** -- Bulk reorder

- Request: `{ "order": ["uuid1", "uuid2", "uuid3", ...] }`
  - Full ordered list of all repo IDs, pinned first then unpinned
- Server validates all IDs present and pinned/unpinned grouping is maintained
- Response: `204 No Content`
- Called once on `drop`, never during drag

**Modify `list_repos`** to sort pinned repos first (preserving relative order within each group).

### Frontend

#### Sort Dropdown

Add a `<select>` above the repo list with options:
- **Custom** (default) -- shows server-persisted manual order
- **Name (A-Z)** / **Name (Z-A)**
- **Date Added (newest)** / **Date Added (oldest)**
- **File Count (most)** / **File Count (least)**

Sorting is **client-side only** -- it does not modify the persisted order. Sort preference stored in `sessionStorage`. Drag handles are **disabled** when a non-Custom sort is active.

Sorting applies **within each group independently** -- pinned repos are sorted among themselves, unpinned among themselves. Pinned always float above unpinned.

#### Pin Toggle

Each repo card gets a pin icon button (thumbtack SVG). Filled when pinned, outlined when unpinned. Positioned left of the action buttons (Sync, Re-index, Delete).

#### Drag-and-Drop

Use **SortableJS** via CDN for drag-and-drop:
- Handles touch/mobile natively
- Works well with vanilla JS
- Minimal footprint (~10KB gzipped)

Two separate sortable containers:
1. **Pinned group** -- repos where `pinned === true`
2. **Unpinned group** -- repos where `pinned === false`

Drag is constrained within each group (dragging does NOT cross the pinned/unpinned boundary). To change pin status, use the pin button.

Each card gets a drag handle (grip dots icon, left side of card). Handle visible always to signal reorder capability. Handle hidden when sort is not "Custom".

On `drop`: send `PUT /api/repos/order` with the new full order.

#### Visual Design

- **Pinned section**: subtle separator line below pinned repos (label: "Pinned" in small muted text). Hidden when no repos are pinned.
- **Pinned card indicator**: filled pin icon on the card
- **Drag handle**: 6-dot grip icon (3 rows of 2 dots), `color: var(--text-muted)`, left side of each card
- **Drag feedback**: SortableJS default ghost (semi-transparent clone) + colored drop indicator line

#### Auto-Poll Conflict Resolution

Set `isDragging = true` on `dragstart`. In `loadRepos()`, early-return if `isDragging` is true. On `dragend`, set `isDragging = false` and call `loadRepos()` immediately to sync state.

#### Newly Added Repos

New repos appear at the bottom of the unpinned group (preserving current `push` behavior). They are unpinned by default.

## Technical Approach

### Phase 1: Backend -- Model + API

**Files to modify:**

1. **`src/models.rs`** -- Add `pinned: bool` with `#[serde(default)]` to `Repo` struct
2. **`src/api/repos.rs`** -- Add `pin_repo` and `reorder_repos` handlers. Modify `list_repos` to sort pinned first.
3. **`src/main.rs`** -- Register new routes: `.route("/api/repos/{id}/pin", patch(pin_repo))` and `.route("/api/repos/order", put(reorder_repos))`
4. **`src/state.rs`** -- Make `persist_repos` atomic (write to temp file, then rename). This prevents corruption from rapid reorder writes or crashes.

**Request/response types (add to `src/models.rs`):**

```rust
#[derive(Debug, Deserialize)]
pub struct ReorderRequest {
    pub order: Vec<Uuid>,
}
```

### Phase 2: Frontend -- Pin + Sort

**File to modify:** `static/index.html`

1. **CSS** (~30 lines):
   - `.pin-btn` -- pin toggle button styles (filled/outlined states)
   - `.pinned-separator` -- divider between groups
   - `.drag-handle` -- grip dots icon
   - `.sort-controls` -- dropdown container

2. **HTML** -- Add sort dropdown to `#page-repos`, above `#repo-list`

3. **JS** -- `loadRepos()`:
   - Split repos into pinned/unpinned arrays
   - Apply client-side sort if sort dropdown is not "Custom"
   - Render pinned group + separator + unpinned group
   - Add `pinRepo(id)` function calling `PATCH /api/repos/{id}/pin`
   - Persist sort preference in `sessionStorage`

### Phase 3: Frontend -- Drag-and-Drop

**File to modify:** `static/index.html`

1. **Add SortableJS CDN** `<script>` tag (before app JS)
2. **Initialize two Sortable instances** in `loadRepos()`:
   - `#pinned-list` and `#unpinned-list` containers
   - `handle: '.drag-handle'`
   - `animation: 150`
   - `onStart`: set `isDragging = true`
   - `onEnd`: set `isDragging = false`, collect new order, call `PUT /api/repos/order`, then `loadRepos()`
3. **Disable Sortable** when sort dropdown is not "Custom"
4. **Auto-poll guard**: check `isDragging` flag in `loadRepos()`, skip innerHTML rebuild if true

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Drag boundary | Constrained within group | Avoids implicit pin/unpin. Pin button is explicit. |
| Sort mechanism | Client-side view only | Preserves manual order. No server roundtrip for sort. |
| Drag during sort | Disabled | Avoids contradicting sort with manual order. |
| Order storage | Vec position (no sort_order field) | Simpler. One source of truth. Consistent with current behavior. |
| Drag library | SortableJS via CDN | Touch support, small footprint, vanilla JS compatible. |
| Reorder API | Full ordered ID list | Idempotent, simple, no edge cases with relative positioning. |
| persist_repos | Atomic write (tmp+rename) | Prevents corruption from rapid writes. |

## Edge Cases

- **Unpin last pinned repo**: pinned separator disappears, repo moves to top of unpinned group
- **Pin while processing (cloning/indexing)**: allowed, status badge still shows on pinned card
- **Delete a pinned repo**: removed from pinned group, separator hides if group becomes empty
- **Auto-poll during drag**: suppressed, resumes on drop
- **Rapid pin/unpin clicks**: each is a separate PATCH, last write wins (acceptable for single-user app)

## Acceptance Criteria

- [x] Repos can be pinned/unpinned via a toggle button on each card
- [x] Pinned repos always appear above unpinned repos
- [x] Repos can be drag-and-drop reordered within pinned and unpinned groups
- [x] Sort dropdown sorts repos by name, date added, or file count
- [x] Sort is client-side only and does not modify persisted order
- [x] Drag handles are hidden when a non-Custom sort is active
- [x] Pin status and manual order persist across page reloads (server-side)
- [x] Existing `repos.json` files load without error (backward compatible)
- [x] Auto-poll does not disrupt in-progress drag operations
- [x] Touch/mobile drag works via SortableJS

## References

- `src/models.rs:7-17` -- current `Repo` struct
- `src/state.rs:30-65` -- repo load/persist logic
- `src/api/repos.rs:11-14` -- `list_repos` handler
- `src/main.rs:31-35` -- route registration
- `static/index.html:2852-2896` -- `loadRepos()` function
- `static/index.html:409-499` -- repo card CSS
- [SortableJS](https://sortablejs.github.io/Sortable/) -- drag library
