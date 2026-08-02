//! Prefix-cache reset action and bounded HTTP evidence.

use crate::workload::domain::{WorkloadEndpoint, WorkloadHttpAction};
use crate::workload::record::PrefixCacheResetEvidence;
use inferlab_runtime::operation_bound::{OperationBound, Remaining};
use std::time::Duration;

pub fn reset_prefix_cache(
    endpoint: &WorkloadEndpoint,
    action: &WorkloadHttpAction,
    bound: &OperationBound,
) -> PrefixCacheResetEvidence {
    let url = format!("http://{}:{}{}", endpoint.host, endpoint.port, action.path);
    let result = post_empty(&url, bound);
    match result {
        Ok(status) if is_successful_cache_reset_status(status) => PrefixCacheResetEvidence {
            method: action.method,
            url,
            succeeded: true,
            http_status: Some(status),
            error: None,
        },
        Ok(status) => PrefixCacheResetEvidence {
            method: action.method,
            url,
            succeeded: false,
            http_status: Some(status),
            error: Some(format!("prefix-cache reset returned HTTP {status}")),
        },
        Err(error) => PrefixCacheResetEvidence {
            method: action.method,
            url,
            succeeded: false,
            http_status: None,
            error: Some(error.to_string()),
        },
    }
}

fn is_successful_cache_reset_status(status: u16) -> bool {
    (200..300).contains(&status) && status != 206
}

#[derive(Debug, thiserror::Error)]
enum PrefixCacheResetError {
    #[error("measurement-case budget expired")]
    Deadline,
    #[error("prefix-cache reset request failed: {source}")]
    Request {
        #[source]
        source: reqwest::Error,
    },
}

fn post_empty(url: &str, bound: &OperationBound) -> Result<u16, PrefixCacheResetError> {
    let attempt = bound.attempt(Some(Duration::from_secs(2)));
    let timeout = match attempt.remaining() {
        Remaining::Finite(duration) => duration,
        Remaining::Expired => return Err(PrefixCacheResetError::Deadline),
        Remaining::Unbounded => Duration::from_secs(2),
    };
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .map_err(|source| PrefixCacheResetError::Request { source })?;
    client
        .post(url)
        .send()
        .map(|response| response.status().as_u16())
        .map_err(|source| {
            if source.is_timeout() {
                PrefixCacheResetError::Deadline
            } else {
                PrefixCacheResetError::Request { source }
            }
        })
}
