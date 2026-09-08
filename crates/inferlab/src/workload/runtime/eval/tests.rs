use super::{repeated_eval_result_error, validate_openai_completion_body};
use crate::workspace::{EvalDefinition, EvalTaskSource};
use inferlab_protocol::{
    ClientStatus, EvalClientResult, EvalMetricComparison, EvalMetricGate, EvalMetricGateConclusion,
    EvalNormalizedMetric, EvalTrialSummary,
};
use std::collections::BTreeMap;

#[test]
fn repeated_eval_rejects_a_gate_conclusion_that_disagrees_with_its_threshold() {
    let definition = EvalDefinition::LmEval {
        task: EvalTaskSource::BuiltIn("fixture".to_owned()),
        prompt: Default::default(),
        request_body: BTreeMap::new(),
        limit: Some(1),
        few_shot: None,
        seed: Some(1234),
        trials: 2,
        max_tokens: None,
        concurrency: Some(1),
        metric: "exact_match".to_owned(),
        metric_filter: Some("strict".to_owned()),
        threshold: 0.75,
        timeout_seconds: 30,
    };
    let normalized = EvalNormalizedMetric {
        source_identity: "fixture".to_owned(),
        metric: "exact_match".to_owned(),
        filter: Some("strict".to_owned()),
        native_metric_key: "inferlab:pass_rate".to_owned(),
        value: 0.5,
        higher_is_better: true,
        prompt_authority: inferlab_protocol::EvalPromptInput::Flat,
    };
    let result = EvalClientResult {
        schema_version: 1,
        status: ClientStatus::Succeeded,
        metrics: BTreeMap::from([("fixture:pass_rate".to_owned(), 0.5)]),
        normalized_metrics: BTreeMap::new(),
        gate: Some(EvalMetricGate {
            metric: normalized,
            threshold: 0.75,
            comparison: EvalMetricComparison::AtLeast,
            conclusion: EvalMetricGateConclusion::Passed,
        }),
        trial_summary: Some(EvalTrialSummary {
            requested_trials: 2,
            issued_trials: 2,
            unissued_trials: 0,
            completed_trials: 2,
            request_failure_trials: 0,
            passed_trials: 1,
            pass_rate: Some(0.5),
            per_trial_metric: "exact_match".to_owned(),
            per_trial_filter: Some("strict".to_owned()),
            higher_is_better: true,
        }),
        native_command: vec!["lm_eval".to_owned()],
        native_exit_code: None,
        native_timed_out: false,
        raw_artifacts: Vec::new(),
        failure_kind: None,
        error: None,
    };

    assert!(
        repeated_eval_result_error(&definition, &result)
            .is_some_and(|error| error.contains("pass-rate threshold semantics"))
    );
}

#[test]
fn openai_smoke_requires_a_nonempty_choices_array_with_text() {
    assert_eq!(
        validate_openai_completion_body(br#"{"choices":[{"text":"ok"}]}"#),
        Ok(1)
    );
    for body in [
        br#"not-json"#.as_slice(),
        br#"{}"#.as_slice(),
        br#"{"choices":[]}"#.as_slice(),
        br#"{"choices":[{}]}"#.as_slice(),
        br#"{"choices":[{"text":1}]}"#.as_slice(),
    ] {
        assert!(validate_openai_completion_body(body).is_err());
    }
}

#[test]
fn vision_smoke_requires_a_nonempty_choices_array_with_message_content() {
    assert_eq!(
        super::validate_openai_chat_vision_body(
            br#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#
        ),
        Ok(1)
    );
    for body in [
        br#"not-json"#.as_slice(),
        br#"{}"#.as_slice(),
        br#"{"choices":[]}"#.as_slice(),
        br#"{"choices":[{}]}"#.as_slice(),
        br#"{"choices":[{"text":"ok"}]}"#.as_slice(),
        br#"{"choices":[{"message":{"role":"assistant"}}]}"#.as_slice(),
        br#"{"choices":[{"message":{"content":1}}]}"#.as_slice(),
    ] {
        assert!(super::validate_openai_chat_vision_body(body).is_err());
    }
}

#[test]
fn vision_smoke_request_body_carries_one_text_part_and_the_fixed_image()
-> Result<(), Box<dyn std::error::Error>> {
    let body = super::OpenAiSmokeRequestBody::ChatVision(super::OpenAiChatVisionRequest {
        model: "qwen3-vl",
        messages: [super::OpenAiChatVisionMessage {
            role: "user",
            content: [
                super::OpenAiChatContentPart::Text {
                    text: "Describe the image.",
                },
                super::OpenAiChatContentPart::ImageUrl {
                    image_url: super::OpenAiImageUrl {
                        url: super::vision_smoke_image_data_uri(),
                    },
                },
            ],
        }],
        max_tokens: 16,
        temperature: 0.0,
        stream: false,
        n: 1,
    });

    let value = serde_json::to_value(&body)?;
    assert_eq!(value["model"], "qwen3-vl");
    assert_eq!(value["max_tokens"], 16);
    assert_eq!(value["temperature"], 0.0);
    assert_eq!(value["stream"], false);
    assert_eq!(value["n"], 1);
    let messages = value["messages"]
        .as_array()
        .ok_or("request has no messages array")?;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    let content = messages[0]["content"]
        .as_array()
        .ok_or("message content is not a parts array")?;
    assert_eq!(
        content[0],
        serde_json::json!({"type": "text", "text": "Describe the image."})
    );
    assert_eq!(content[1]["type"], "image_url");
    let url = content[1]["image_url"]["url"]
        .as_str()
        .ok_or("image part has no url")?;
    let prefix = "data:image/png;base64,";
    assert!(url.starts_with(prefix), "{url}");
    use base64::Engine;
    let decoded = base64::engine::general_purpose::STANDARD.decode(&url[prefix.len()..])?;
    assert_eq!(decoded, super::VISION_SMOKE_IMAGE_PNG);
    Ok(())
}

#[test]
fn non_vision_smoke_request_body_keeps_the_completions_shape()
-> Result<(), Box<dyn std::error::Error>> {
    let body = super::OpenAiSmokeRequestBody::Completion(super::OpenAiCompletionRequest {
        model: "deepseek-v4-flash",
        prompt: "San Francisco is a city in",
        max_tokens: 16,
        temperature: 0.0,
        stream: false,
        n: 1,
    });

    assert_eq!(
        serde_json::to_string(&body)?,
        r#"{"model":"deepseek-v4-flash","prompt":"San Francisco is a city in","max_tokens":16,"temperature":0.0,"stream":false,"n":1}"#
    );
    Ok(())
}
