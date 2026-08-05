//! Prefix-cache reset action and bounded HTTP evidence.

use crate::workload::domain::{WorkloadEndpoint, WorkloadHttpAction};
use crate::workload::record::PrefixCacheResetEvidence;
use inferlab_runtime::operation_bound::{OperationBound, Remaining};

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
    #[error("prefix-cache reset requires a finite measurement-case budget")]
    UnboundedBudget,
    #[error("prefix-cache reset request failed: {source}")]
    Request {
        #[source]
        source: reqwest::Error,
    },
}

fn post_empty(url: &str, bound: &OperationBound) -> Result<u16, PrefixCacheResetError> {
    let timeout = match bound.remaining() {
        Remaining::Finite(duration) => duration,
        Remaining::Expired => return Err(PrefixCacheResetError::Deadline),
        Remaining::Unbounded => return Err(PrefixCacheResetError::UnboundedBudget),
    };
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .map_err(|source| PrefixCacheResetError::Request { source })?;
    let mut response = client.post(url).send().map_err(|source| {
        if source.is_timeout() {
            PrefixCacheResetError::Deadline
        } else {
            PrefixCacheResetError::Request { source }
        }
    })?;
    let status = response.status().as_u16();
    response.copy_to(&mut std::io::sink()).map_err(|source| {
        if source.is_timeout() {
            PrefixCacheResetError::Deadline
        } else {
            PrefixCacheResetError::Request { source }
        }
    })?;
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn reset_can_complete_after_the_former_private_cap() -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            thread::sleep(Duration::from_millis(2_100));
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        });

        let status = post_empty(
            &format!("http://{address}/reset_prefix_cache"),
            &OperationBound::finite(Duration::from_secs(3)),
        )?;

        assert_eq!(status, 200);
        server.join().map_err(|_| "fixture server panicked")??;
        Ok(())
    }

    #[test]
    fn reset_deadline_includes_the_complete_response_body() -> Result<(), Box<dyn std::error::Error>>
    {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nx",
            )?;
            thread::sleep(Duration::from_millis(1_500));
            Ok(())
        });

        let started = Instant::now();
        let result = post_empty(
            &format!("http://{address}/reset_prefix_cache"),
            &OperationBound::finite(Duration::from_secs(1)),
        );
        let elapsed = started.elapsed();
        server.join().map_err(|_| "fixture server panicked")??;

        assert!(matches!(result, Err(PrefixCacheResetError::Deadline)));
        assert!(
            elapsed < Duration::from_millis(1_500),
            "elapsed {elapsed:?}"
        );
        Ok(())
    }
}
