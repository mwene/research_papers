//! REST API for the diagnostic framework (Axum port of the FastAPI service).
//!
//! Run with:
//!     ai_diagnostic serve --host 0.0.0.0 --port 8000
//!
//! or embedded:
//!     let app = ai_diagnostic_rs::api::create_app();
//!     let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
//!     axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()).await?;

use crate::bayesian::{BayesianDiagnostic, DiagnosisResult};
use crate::diagnose::{find_preset, PRESETS};
use crate::evidence::{evidence_description, Evidence, EVIDENCE_PARAMS};
use crate::likelihoods::failure_class_label;
use crate::sensitivity::{compute_sensitivity, SensitivityReport};

use axum::extract::{ConnectInfo, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::{Json as AxumJson, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tower_http::cors::{Any, CorsLayer};

// ── Request/response models ───────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct DiagnoseRequest {
    pub evidence: Evidence,
    #[serde(default)]
    pub priors: Option<HashMap<String, f64>>,
    #[serde(default = "default_threshold")]
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BatchCase {
    pub evidence: Evidence,
    #[serde(default)]
    pub priors: Option<HashMap<String, f64>>,
    #[serde(default = "default_threshold")]
    pub confidence_threshold: f64,
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub cases: Vec<BatchCase>,
}

#[derive(Debug, Deserialize)]
pub struct SensitivityRequest {
    pub evidence: Evidence,
    #[serde(default)]
    pub priors: Option<HashMap<String, f64>>,
    #[serde(default = "default_threshold")]
    pub confidence_threshold: f64,
}

#[derive(Debug, Deserialize)]
pub struct CompareRequest {
    pub evidence_a: Evidence,
    pub evidence_b: Evidence,
    #[serde(default)]
    pub priors: Option<HashMap<String, f64>>,
    #[serde(default = "default_threshold")]
    pub confidence_threshold: f64,
}

fn default_threshold() -> f64 {
    0.6
}

// ── Rate limiter (in-memory, per client IP) ──────────────────────────

struct RateLimiter {
    limit: usize,
    window: Duration,
    hits: Mutex<HashMap<String, Vec<Instant>>>,
}

impl RateLimiter {
    fn new(limit: usize, window: Duration) -> Self {
        RateLimiter {
            limit,
            window,
            hits: Mutex::new(HashMap::new()),
        }
    }

    fn check(&self, key: &str) -> bool {
        let mut hits = self.hits.lock().unwrap();
        let now = Instant::now();
        let cutoff = now - self.window;
        let entry = hits.entry(key.to_string()).or_default();
        entry.retain(|t| *t > cutoff);
        if entry.len() >= self.limit {
            false
        } else {
            entry.push(now);
            true
        }
    }
}

// ── Shared state ──────────────────────────────────────────────────────

struct AppState {
    engine: BayesianDiagnostic,
    rate_limiter: RateLimiter,
}

type SharedState = Arc<AppState>;

// ── Helpers ───────────────────────────────────────────────────────────

fn result_response(result: &DiagnosisResult) -> Value {
    json!({
        "posteriors": result
            .posteriors
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect::<Map<String, Value>>(),
        "diagnosis": result.diagnosis,
        "diagnosis_label": failure_class_label(&result.diagnosis),
        "confidence": result.confidence,
        "needs_investigation": result.needs_investigation,
        "log_likelihoods": result
            .log_likelihoods
            .iter()
            .map(|(k, v)| (k.clone(), json!(v)))
            .collect::<Map<String, Value>>(),
    })
}

fn sensitivity_response(report: &SensitivityReport) -> Value {
    Value::Object(report.to_dict())
}

fn error(status: StatusCode, message: String) -> Response {
    (status, AxumJson(json!({ "detail": message }))).into_response()
}

fn boxed_error(status: StatusCode, message: String) -> Box<Response> {
    Box::new(error(status, message))
}

// ── App factory ───────────────────────────────────────────────────────

pub fn create_app() -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_credentials(false)
        .allow_methods(Any)
        .allow_headers(Any);

    let state = Arc::new(AppState {
        engine: BayesianDiagnostic::new(),
        rate_limiter: RateLimiter::new(100, Duration::from_secs(60)),
    });

    Router::new()
        .route("/health", get(health))
        .route("/presets", get(list_presets))
        .route("/evidence/params", get(evidence_params))
        .route("/diagnose", post(diagnose).layer(cors.clone()))
        .route("/diagnose/batch", post(diagnose_batch))
        .route("/diagnose/preset/{preset_name}", post(diagnose_preset))
        .route("/sensitivity", post(sensitivity))
        .route("/compare", post(compare))
        .layer(cors)
        .with_state(state)
}

async fn health(State(_state): State<SharedState>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "presets_loaded": PRESETS.len(),
        "evidence_params": EVIDENCE_PARAMS.len(),
    }))
}

async fn diagnose(
    State(state): State<SharedState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumJson(req): AxumJson<DiagnoseRequest>,
) -> Response {
    match check_rate_limit_inner(&state, &addr) {
        Ok(()) => {}
        Err(resp) => return *resp,
    }
    match state.engine.diagnose(
        &req.evidence,
        req.priors.as_ref(),
        req.confidence_threshold,
    ) {
        Ok(result) => AxumJson(result_response(&result)).into_response(),
        Err(e) => error(StatusCode::UNPROCESSABLE_ENTITY, e),
    }
}

async fn diagnose_batch(
    State(state): State<SharedState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumJson(req): AxumJson<BatchRequest>,
) -> Response {
    if let Err(resp) = check_rate_limit_inner(&state, &addr) {
        return *resp;
    }
    let mut results = Vec::with_capacity(req.cases.len());
    for case in &req.cases {
        match state.engine.diagnose(
            &case.evidence,
            case.priors.as_ref(),
            case.confidence_threshold,
        ) {
            Ok(r) => results.push(result_response(&r)),
            Err(e) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
        }
    }
    AxumJson(json!(results)).into_response()
}

async fn diagnose_preset(
    State(state): State<SharedState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    axum::extract::Path(preset_name): axum::extract::Path<String>,
) -> Response {
    if let Err(resp) = check_rate_limit_inner(&state, &addr) {
        return *resp;
    }
    match find_preset(&preset_name) {
        None => error(
            StatusCode::NOT_FOUND,
            format!(
                "Unknown preset: {preset_name}. Available: {}",
                PRESETS.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")
            ),
        ),
        Some(preset) => {
            let evidence = match preset.evidence() {
                Ok(ev) => ev,
                Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
            };
            match state.engine.diagnose(&evidence, preset.priors_map().as_ref(), 0.6) {
                Ok(result) => AxumJson(result_response(&result)).into_response(),
                Err(e) => error(StatusCode::UNPROCESSABLE_ENTITY, e),
            }
        }
    }
}

async fn list_presets() -> Json<Value> {
    Json(json!(
        PRESETS
            .iter()
            .map(|p| json!({ "name": p.name, "description": p.description }))
            .collect::<Vec<Value>>()
    ))
}

async fn sensitivity(
    State(state): State<SharedState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumJson(req): AxumJson<SensitivityRequest>,
) -> Response {
    if let Err(resp) = check_rate_limit_inner(&state, &addr) {
        return *resp;
    }
    let result = match state.engine.diagnose(
        &req.evidence,
        req.priors.as_ref(),
        req.confidence_threshold,
    ) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
    };
    match compute_sensitivity(&result, &state.engine) {
        Ok(report) => AxumJson(sensitivity_response(&report)).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn compare(
    State(state): State<SharedState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    AxumJson(req): AxumJson<CompareRequest>,
) -> Response {
    if let Err(resp) = check_rate_limit_inner(&state, &addr) {
        return *resp;
    }
    let run = |ev: &Evidence| {
        state
            .engine
            .diagnose(ev, req.priors.as_ref(), req.confidence_threshold)
    };
    let (ra, rb) = match (run(&req.evidence_a), run(&req.evidence_b)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
    };
    AxumJson(json!({
        "case_a": result_response(&ra),
        "case_b": result_response(&rb),
    }))
    .into_response()
}

async fn evidence_params() -> Json<Value> {
    Json(json!(
        EVIDENCE_PARAMS
            .iter()
            .map(|name| {
                #[derive(Serialize)]
                struct ParamInfo<'a> {
                    name: &'a str,
                    description: &'a str,
                }
                ParamInfo {
                    name,
                    description: evidence_description(name),
                }
            })
            .collect::<Vec<_>>()
    ))
}

fn check_rate_limit_inner(state: &AppState, addr: &SocketAddr) -> Result<(), Box<Response>> {
    if state.rate_limiter.check(&addr.ip().to_string()) {
        Ok(())
    } else {
        Err(boxed_error(
            StatusCode::TOO_MANY_REQUESTS,
            "Rate limit exceeded. Max 100 requests per minute.".to_string(),
        ))
    }
}

/// Serve the API (used by the `serve` CLI command).
pub async fn serve(host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_app();
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    println!("AI Diagnostic API listening on http://{host}:{port}");
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await?;
    Ok(())
}
