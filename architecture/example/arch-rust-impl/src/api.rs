use axum::{
    routing::{get, post},
    Router, Json, extract::State,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::core::{AnalysisResult, Environment, Agent, Game};
use crate::llm::LlmClient;
use crate::narrow::NarrowAlgorithm;
use crate::extended::ExtendedAlgorithm;

#[derive(Clone)]
pub struct AppState {
    llm: Arc<Mutex<Option<LlmClient>>>,
    narrow: Arc<NarrowAlgorithm>,
    extended: Arc<ExtendedAlgorithm>,
    questions_asked: Arc<Mutex<Vec<String>>>,
}

impl AppState {
    pub fn new() -> Self {
        let llm_client = LlmClient::from_env().ok();
        Self {
            llm: Arc::new(Mutex::new(llm_client)),
            narrow: Arc::new(NarrowAlgorithm::new()),
            extended: Arc::new(ExtendedAlgorithm::new()),
            questions_asked: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct AnalyzeRequest {
    pub phenomenon: String,
    pub agents: Vec<Agent>,
    pub environment: Environment,
    pub use_llm: bool,
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub query: String,
    pub context: String,
    pub use_llm: bool,
}

#[derive(Debug, Deserialize)]
pub struct ClarifyRequest {
    pub question: String,
    pub info: String,
    pub use_llm: bool,
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub result: String,
    pub needs_clarification: bool,
    pub clarification_questions: Vec<String>,
}

pub fn routes() -> Router {
    let state = AppState::new();
    Router::new()
        .route("/health", get(health))
        .route("/analyze", post(analyze))
        .route("/query", post(query))
        .route("/clarify", post(clarify))
        .route("/classify", post(classify))
        .with_state(state)
}

async fn health() -> &'static str {
    "OK"
}

async fn analyze(
    State(state): State<AppState>,
    Json(req): Json<AnalyzeRequest>,
) -> Json<AnalysisResult> {
    let game = Game {
        agents: req.agents.clone(),
        equilibria: Vec::new(),
    };
    
    let ext_state = crate::extended::ExtendedState::new(game, req.environment);
    let result = state.extended.analyze(&ext_state);
    Json(result)
}

async fn query(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    let mut result_text = String::new();
    let mut needs_clarification = false;
    let mut clarification_questions = Vec::new();
    
    if req.use_llm {
        let llm_guard = state.llm.lock().await;
        if let Some(ref client) = *llm_guard {
            if let Ok(resp) = client.search(&req.query, &req.context).await {
                result_text = resp;
            }
        }
    }
    
    if result_text.is_empty() {
        result_text = format!("Analysis: {} (context: {})", req.query, req.context);
    }
    
    // Check if more info needed
    if req.query.to_lowercase().contains("architecture") && req.context.len() < 50 {
        needs_clarification = true;
        clarification_questions.push("What is the purpose/goal of this architecture?".to_string());
        clarification_questions.push("What are the key parts/elements?".to_string());
        clarification_questions.push("What environment does it operate in?".to_string());
        clarification_questions.push("Are there multiple agents with purposes of their own?".to_string());
        clarification_questions.push("Which of the 5 principles are most critical?".to_string());
    }
    
    Json(QueryResponse {
        result: result_text,
        needs_clarification,
        clarification_questions,
    })
}

async fn clarify(
    State(state): State<AppState>,
    Json(req): Json<ClarifyRequest>,
) -> Json<QueryResponse> {
    let mut result_text = String::new();
    if req.use_llm {
        let llm_guard = state.llm.lock().await;
        if let Some(ref client) = *llm_guard {
            if let Ok(resp) = client.clarify(&req.question, &req.info).await {
                result_text = resp;
            }
        }
    }
    
    if result_text.is_empty() {
        result_text = format!("Clarification for '{}': based on theory, consider purpose, parts, environment, agency (5-principle framework).", req.question);
    }
    
    Json(QueryResponse {
        result: result_text,
        needs_clarification: false,
        clarification_questions: Vec::new(),
    })
}

async fn classify(
    State(_state): State<AppState>,
    Json(agents): Json<Vec<Agent>>,
) -> Json<AnalysisResult> {
    let has_turing = agents.iter().any(|a| a.is_turing_complete);
    let has_strategic = agents.iter().any(|a| a.has_purpose && (a.has_private_info || a.has_commitment_problem || a.can_best_respond));
    
    let (classification, mode, rationale) = if has_turing {
        (Classification::Undecidable, Mode::Undecidable, "Turing-complete agents present - barrier mode".to_string())
    } else if has_strategic && agents.len() > 1 {
        (Classification::Relation, Mode::Extended, "Multiple strategic agents with conflicting purposes - extended mode".to_string())
    } else if agents.is_empty() || !agents.iter().any(|a| a.has_purpose) {
        (Classification::Tool, Mode::Narrow, "No strategic agents with purposes - tool, narrow mode".to_string())
    } else {
        (Classification::Tool, Mode::Narrow, "Single/trivial agency - tool, narrow mode".to_string())
    };
    
    Json(AnalysisResult {
        classification,
        mode,
        confidence: 0.85,
        rationale,
        recommendations: vec![
            "Consider 5 principles: function, speed/efficiency, safety/security, scale/capacity/load, aesthetics".to_string(),
            "Assess co-evolution with environment".to_string(),
            "Apply classifier logic from theory".to_string(),
        ],
    })
}
