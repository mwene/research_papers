mod advice;
mod five;
mod mathutil;
mod params;
mod quantities;
pub mod verdict;

#[cfg(feature = "wasm")]
mod wasm;

pub use params::DecisionParams;
pub use verdict::{decide, Action, Scores, Verdict};

pub fn decide_from_params(p: &DecisionParams) -> Verdict {
    let warnings = p.validate();
    let mut v = decide(&p.resolved());
    if !warnings.is_empty() {
        v.advice.push(format!(
            "Input note: {} warning(s) on parameter ranges (see logs).",
            warnings.len()
        ));
    }
    v
}

pub fn validate_params(p: &DecisionParams) -> Vec<String> {
    p.validate()
}
