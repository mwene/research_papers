mod core;
mod llm;
mod api;
mod narrow;
mod extended;
mod web;

use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;
use std::net::SocketAddr;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    
    let app = Router::new()
        .nest("/api", api::routes())
        .nest_service("/", ServeDir::new("ui"))
        .layer(CorsLayer::permissive());
    
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("Server running on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
