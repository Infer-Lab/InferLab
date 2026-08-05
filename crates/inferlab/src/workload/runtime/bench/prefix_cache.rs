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
    #[error("failed to initialize prefix-cache reset HTTP runtime: {source}")]
    Runtime {
        #[source]
        source: std::io::Error,
    },
    #[error("prefix-cache reset request failed: {source}")]
    Request {
        #[source]
        source: reqwest::Error,
    },
}

fn post_empty(url: &str, bound: &OperationBound) -> Result<u16, PrefixCacheResetError> {
    match bound.remaining() {
        Remaining::Finite(_) => {}
        Remaining::Expired => return Err(PrefixCacheResetError::Deadline),
        Remaining::Unbounded => return Err(PrefixCacheResetError::UnboundedBudget),
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    if bound.is_expired() {
        return Err(PrefixCacheResetError::Deadline);
    }
    let runtime = runtime.map_err(|source| PrefixCacheResetError::Runtime { source })?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .tcp_keepalive(None)
        .tcp_keepalive_interval(None)
        .tcp_keepalive_retries(None);
    #[cfg(any(target_os = "android", target_os = "fuchsia", target_os = "linux"))]
    let client = client.tcp_user_timeout(None);
    let client = client.build();
    if bound.is_expired() {
        return Err(PrefixCacheResetError::Deadline);
    }
    let client = client.map_err(|source| PrefixCacheResetError::Request { source })?;
    let remaining = match bound.remaining() {
        Remaining::Finite(duration) => duration,
        Remaining::Expired => return Err(PrefixCacheResetError::Deadline),
        Remaining::Unbounded => return Err(PrefixCacheResetError::UnboundedBudget),
    };
    let outcome = runtime.block_on(async {
        let request = async {
            let mut response = client
                .post(url)
                .send()
                .await
                .map_err(|source| PrefixCacheResetError::Request { source })?;
            let status = response.status().as_u16();
            loop {
                if bound.is_expired() {
                    return Err(PrefixCacheResetError::Deadline);
                }
                let chunk = response
                    .chunk()
                    .await
                    .map_err(|source| PrefixCacheResetError::Request { source })?;
                if chunk.is_none() {
                    return Ok(status);
                }
            }
        };
        tokio::select! {
            biased;
            () = tokio::time::sleep(remaining) => Err(PrefixCacheResetError::Deadline),
            result = request => result,
        }
    });
    if bound.is_expired() {
        Err(PrefixCacheResetError::Deadline)
    } else {
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::thread;
    use std::time::Duration;

    fn read_request_headers(stream: &mut TcpStream) -> std::io::Result<()> {
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line)? {
                0 => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "request ended before its header terminator",
                    ));
                }
                _ if line == "\r\n" => return Ok(()),
                _ => {}
            }
        }
    }

    #[test]
    fn reset_can_complete_after_the_former_private_cap() -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            read_request_headers(&mut stream)?;
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
            read_request_headers(&mut stream)?;
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nx")?;
            thread::sleep(Duration::from_millis(300));
            stream.write_all(b"y")?;
            thread::sleep(Duration::from_millis(300));
            Ok(())
        });

        let bound = OperationBound::finite(Duration::from_millis(500));
        let result = post_empty(&format!("http://{address}/reset_prefix_cache"), &bound);
        server.join().map_err(|_| "fixture server panicked")??;

        assert!(
            matches!(result, Err(PrefixCacheResetError::Deadline)),
            "result={result:?}, remaining={:?}, elapsed_ms={}",
            bound.remaining(),
            bound.elapsed_ms()
        );
        Ok(())
    }

    #[test]
    fn reset_rejects_a_complete_response_after_the_owner_deadline()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            read_request_headers(&mut stream)?;
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nx")?;
            thread::sleep(Duration::from_millis(300));
            stream.write_all(b"y")?;
            thread::sleep(Duration::from_millis(300));
            match stream.write_all(b"z") {
                Ok(()) => Ok(()),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
                    ) =>
                {
                    Ok(())
                }
                Err(error) => Err(error),
            }
        });

        let bound = OperationBound::finite(Duration::from_millis(500));
        let result = post_empty(&format!("http://{address}/reset_prefix_cache"), &bound);
        server.join().map_err(|_| "fixture server panicked")??;

        assert!(
            matches!(result, Err(PrefixCacheResetError::Deadline)),
            "result={result:?}, remaining={:?}, elapsed_ms={}",
            bound.remaining(),
            bound.elapsed_ms()
        );
        Ok(())
    }

    #[test]
    fn reset_preserves_a_transport_failure_observed_before_the_deadline()
    -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> std::io::Result<()> {
            let (mut stream, _) = listener.accept()?;
            read_request_headers(&mut stream)?;
            drop(stream);
            Ok(())
        });

        let result = post_empty(
            &format!("http://{address}/reset_prefix_cache"),
            &OperationBound::finite(Duration::from_secs(1)),
        );
        server.join().map_err(|_| "fixture server panicked")??;

        assert!(matches!(result, Err(PrefixCacheResetError::Request { .. })));
        Ok(())
    }
}
