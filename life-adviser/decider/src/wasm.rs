use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn decide_json(params_json: &str) -> Result<String, JsValue> {
    let p: crate::DecisionParams = serde_json::from_str(params_json)
        .map_err(|e| JsValue::from_str(&format!("invalid parameters JSON: {e}")))?;
    let v = crate::decide_from_params(&p);
    serde_json::to_string(&v).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn validate_json(params_json: &str) -> Result<String, JsValue> {
    let p: crate::DecisionParams = serde_json::from_str(params_json)
        .map_err(|e| JsValue::from_str(&format!("invalid parameters JSON: {e}")))?;
    serde_json::to_string(&crate::validate_params(&p))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn decider_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
