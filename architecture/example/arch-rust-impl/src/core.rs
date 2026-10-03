use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartType {
    Physical,
    Informational,
    Conceptual,
    Factual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    pub id: Uuid,
    pub name: String,
    pub part_type: PartType,
    pub attributes: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub from: Uuid,
    pub to: Uuid,
    pub relation_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arrangement {
    pub id: Uuid,
    pub parts: Vec<Part>,
    pub relations: Vec<Relation>,
    pub attributes: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    pub passive: Vec<Element>,
    pub active: Vec<Element>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Element {
    pub id: Uuid,
    pub name: String,
    pub is_living: bool,
    pub is_active: bool,
    pub attributes: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Principles {
    pub function: f64,
    pub speed_efficiency: f64,
    pub safety_security: f64,
    pub scale_capacity_load: f64,
    pub aesthetics: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub id: Uuid,
    pub description: String,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: Uuid,
    pub name: String,
    pub has_purpose: bool,
    pub has_private_info: bool,
    pub has_commitment_problem: bool,
    pub can_best_respond: bool,
    pub is_turing_complete: bool,
    pub utilities: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Equilibrium {
    pub id: Uuid,
    pub description: String,
    pub is_peace: bool,
    pub is_trap: bool,
    pub is_fragile: bool,
    pub is_chaos: bool,
    pub stability: f64,
    pub basin_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Game {
    pub agents: Vec<Agent>,
    pub equilibria: Vec<Equilibrium>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Mode {
    Narrow,
    Extended,
    Undecidable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Classification {
    Tool,
    Relation,
    Undecidable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub classification: Classification,
    pub mode: Mode,
    pub confidence: f64,
    pub rationale: String,
    pub recommendations: Vec<String>,
}
