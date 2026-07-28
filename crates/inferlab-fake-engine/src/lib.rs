//! Executable token-only Engine fixture for the specialized-engine workflow.
//!
//! The core deliberately has no HTTP, tokenizer, chat-template, placement, or
//! process-lifecycle concepts. The SMG-facing transport adapts those external
//! concerns to this token request/result boundary when the optional
//! `smg-transport` feature is enabled.

use thiserror::Error;

#[cfg(feature = "smg-transport")]
pub mod smg;

/// The complete input understood by the deterministic fake execution core.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerateRequest {
    pub prompt_token_ids: Vec<u32>,
    pub max_output_tokens: u32,
}

/// Why the fake Engine stopped producing token IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
}

impl FinishReason {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Length => "length",
        }
    }
}

/// The terminal token result returned by the fake execution core.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerateOutput {
    pub token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
}

/// A request rejected at the token execution boundary.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum EngineError {
    #[error("prompt_token_ids must not be empty")]
    EmptyPrompt,
}

/// Minimal execution contract shared by the fixture core and its transports.
pub trait TokenEngine {
    fn generate(&self, request: &GenerateRequest) -> Result<GenerateOutput, EngineError>;
}

/// A deterministic Engine used to pin the token execution boundary.
#[derive(Clone, Copy, Debug, Default)]
pub struct EchoEngine;

impl TokenEngine for EchoEngine {
    fn generate(&self, request: &GenerateRequest) -> Result<GenerateOutput, EngineError> {
        if request.prompt_token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }

        let token_ids = request
            .prompt_token_ids
            .iter()
            .copied()
            .cycle()
            .take(request.max_output_tokens as usize)
            .collect();
        Ok(GenerateOutput {
            token_ids,
            finish_reason: FinishReason::Length,
        })
    }
}
