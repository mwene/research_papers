use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub model: String,
    pub messages: Vec<LlmMessage>,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub message: LlmMessage,
}

#[derive(Clone)]
pub struct LlmClient {
    config: LlmConfig,
    client: reqwest::Client,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }
    
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let api_key = env::var("LLM_API_KEY").unwrap_or_default();
        let base_url = env::var("LLM_BASE_URL").unwrap_or("https://api.openai.com/v1".to_string());
        let model = env::var("LLM_MODEL").unwrap_or("gpt-3.5-turbo".to_string());
        
        Ok(Self::new(LlmConfig {
            api_key,
            base_url,
            model,
        }))
    }
    
    pub async fn search(&self, query: &str, context: &str) -> Result<String, Box<dyn std::error::Error>> {
        let prompt = format!(
            "You are an expert on the Theory of Architecture (Macharia Barii). Context: {}\n\nQuery: {}\n\nProvide a concise, relevant response.",
            context, query
        );
        
        let request = LlmRequest {
            model: self.config.model.clone(),
            messages: vec![
                LlmMessage {
                    role: "user".to_string(),
                    content: prompt,
                }
            ],
            temperature: 0.7,
        };
        
        let response = self.client
            .post(format!("{}/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&request)
            .send()
            .await?;
        
        let resp: LlmResponse = response.json().await?;
        if let Some(choice) = resp.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Ok("No response".to_string())
        }
    }
    
    pub async fn clarify(&self, question: &str, info: &str) -> Result<String, Box<dyn std::error::Error>> {
        let prompt = format!(
            "Based on the theory of architecture (5 principles: function, speed/efficiency, safety/security, scale/capacity/load, aesthetics), help clarify: {}\n\nCurrent info: {}",
            question, info
        );
        
        let request = LlmRequest {
            model: self.config.model.clone(),
            messages: vec![
                LlmMessage {
                    role: "user".to_string(),
                    content: prompt,
                }
            ],
            temperature: 0.3,
        };
        
        let response = self.client
            .post(format!("{}/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&request)
            .send()
            .await?;
        
        let resp: LlmResponse = response.json().await?;
        if let Some(choice) = resp.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Ok("No clarification available".to_string())
        }
    }
}
