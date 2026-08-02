use inferlab::InferlabError;
use std::path::PathBuf;

fn is_error_code(text: &str) -> bool {
    text.len() == 5 && text.starts_with('E') && text[1..].chars().all(|c| c.is_ascii_digit())
}

fn representative_errors() -> Vec<InferlabError> {
    vec![
        InferlabError::WorkspaceNotFound {
            start: PathBuf::from("fixture"),
        },
        InferlabError::Read {
            path: PathBuf::from("fixture"),
            source: std::io::Error::other("fixture"),
        },
        InferlabError::ParseToml {
            path: PathBuf::from("fixture"),
            source: <toml::de::Error as serde::de::Error>::custom("fixture"),
        },
        InferlabError::InvalidConfig {
            message: "fixture".to_owned(),
        },
        InferlabError::InvalidOverride {
            value: "fixture".to_owned(),
            message: "fixture".to_owned(),
        },
        InferlabError::Git {
            root: PathBuf::from("fixture"),
            source: inferlab::GitError::InvalidOutput {
                operation: "fixture".to_owned(),
                detail: "fixture".to_owned(),
            },
        },
        InferlabError::EnvironmentLifecycle {
            message: "fixture".to_owned(),
        },
        InferlabError::UnsupportedToolchainPlatform {
            platform: "fixture".to_owned(),
        },
        InferlabError::AdapterTimeout {
            integration: "fixture".to_owned(),
            seconds: 1,
        },
        InferlabError::InsufficientDevices {
            machine: "fixture".to_owned(),
            required: 1,
            available: 0,
        },
        InferlabError::RecipeFailed {
            record_id: "fixture".to_owned(),
        },
        InferlabError::ServerLifecycle {
            message: "fixture".to_owned(),
        },
        InferlabError::ImageBuild {
            message: "fixture".to_owned(),
        },
        InferlabError::RecordIo {
            path: PathBuf::from("fixture"),
            source: std::io::Error::other("fixture"),
        },
        InferlabError::OperationObservationIo {
            operation: "read",
            path: PathBuf::from("fixture"),
            source: std::io::Error::other("fixture"),
        },
        InferlabError::Scratchpad {
            message: "fixture".to_owned(),
        },
        InferlabError::Agent {
            message: "fixture".to_owned(),
        },
        InferlabError::AdHocRun {
            message: "fixture".to_owned(),
        },
        InferlabError::WriteOutput {
            source: std::io::Error::other("fixture"),
        },
    ]
}

/// The registry table shipped in the rendered specification and the codes
/// returned by representative public errors must agree exactly. This lives
/// outside src/ so the published crate carries no test that reads outside the
/// package. A semantic remapping within an existing code remains review's job
/// under the clause's append-only and MAY-join rules.
#[test]
fn the_shipped_registry_and_the_emitted_codes_agree() {
    let errors = representative_errors();
    let emitted: std::collections::BTreeSet<&str> =
        errors.iter().map(InferlabError::code).collect();

    let documented: std::collections::BTreeSet<&str> =
        include_str!("../../../docs/rfc/RFC-0001.md")
            .lines()
            .filter_map(|line| line.strip_prefix('|'))
            .filter_map(|rest| rest.split('|').next())
            .map(str::trim)
            .filter(|cell| is_error_code(cell))
            .collect();

    assert!(
        !emitted.is_empty() && !documented.is_empty(),
        "extraction found nothing; the registry table or code() moved"
    );
    assert_eq!(
        emitted, documented,
        "the registry in docs/rfc/RFC-0001.md and InferlabError::code() disagree"
    );
}
