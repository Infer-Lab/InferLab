use inferlab_fake_engine::{
    EchoEngine, EngineError, FinishReason, GenerateOutput, GenerateRequest, TokenEngine,
};

#[test]
fn echo_engine_cycles_prompt_token_ids_to_the_requested_limit() {
    let output = EchoEngine.generate(&GenerateRequest {
        prompt_token_ids: vec![11, 22, 33],
        max_output_tokens: 5,
    });

    assert_eq!(
        output,
        Ok(GenerateOutput {
            token_ids: vec![11, 22, 33, 11, 22],
            finish_reason: FinishReason::Length,
        })
    );
}

#[test]
fn echo_engine_rejects_an_empty_token_prompt() {
    let output = EchoEngine.generate(&GenerateRequest {
        prompt_token_ids: Vec::new(),
        max_output_tokens: 1,
    });

    assert_eq!(output, Err(EngineError::EmptyPrompt));
}
