use axum::response::Html;
use axum::routing::{delete, get, patch, post, put};
use axum::Router;
use tracing_subscriber::EnvFilter;

use repo_search::api;
use repo_search::config::Config;
use repo_search::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env();
    tracing::info!("Data directory: {}", config.data_dir.display());
    tracing::info!("LLM provider: {} ({})", config.llm.provider, config.llm.base_url);

    let state = AppState::new(config.clone())?;

    // No CORS layer: the SPA is served from the same origin so cross-origin
    // access is unnecessary. This prevents drive-by attacks from malicious pages.
    let app = Router::new()
        // Serve frontend
        .route("/", get(serve_index))
        // API routes
        .route("/api/repos", get(api::repos::list_repos))
        .route("/api/repos", post(api::repos::add_repo))
        .route("/api/repos/{id}", delete(api::repos::delete_repo))
        .route("/api/repos/{id}/reindex", post(api::repos::reindex_repo))
        .route("/api/repos/{id}/sync", post(api::repos::sync_repo))
        .route("/api/repos/{id}/pin", patch(api::repos::pin_repo))
        .route("/api/repos/order", put(api::repos::reorder_repos))
        .route("/api/search", post(api::search::search))
        .route("/api/chat", post(api::chat::chat))
        .route("/api/config", get(api::repos::get_config))
        .route("/api/config", put(api::repos::update_config))
        .with_state(state)
        .fallback(get(serve_index));

    let listener = tokio::net::TcpListener::bind(&config.bind_addr).await?;
    tracing::info!("Server listening on {}", config.bind_addr);

    axum::serve(listener, app).await?;
    Ok(())
}

async fn serve_index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}
