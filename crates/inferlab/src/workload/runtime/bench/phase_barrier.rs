//! A runtime-local acknowledgement between AIPerf's profiling setup and the
//! Rust-owned capture window.

use super::super::CLIENT_POLL_INTERVAL;
use crate::InferlabError;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::{self, ScopedJoinHandle};

pub(super) const PROFILE_BARRIER_ENV: &str = "INFERLAB_AIPERF_PROFILE_BARRIER";
pub(super) const PROFILE_BARRIER_REQUIRES_WARMUP_ENV: &str =
    "INFERLAB_AIPERF_PROFILE_BARRIER_REQUIRES_WARMUP";
const PROFILE_READY: &[u8] = b"profiling-ready\n";
const CAPTURE_OPEN: &[u8] = b"capture-open\n";

pub(super) struct ProfileBarrier {
    listener: TcpListener,
    address: String,
}

impl ProfileBarrier {
    pub(super) fn bind() -> Result<Self, InferlabError> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|source| {
            InferlabError::ProfileBarrierIo {
                operation: "bind its loopback listener",
                source,
            }
        })?;
        listener
            .set_nonblocking(true)
            .map_err(|source| InferlabError::ProfileBarrierIo {
                operation: "configure its loopback listener",
                source,
            })?;
        let address = listener
            .local_addr()
            .map_err(|source| InferlabError::ProfileBarrierIo {
                operation: "resolve its loopback address",
                source,
            })?
            .to_string();
        Ok(Self { listener, address })
    }

    pub(super) fn address(&self) -> &str {
        &self.address
    }

    pub(super) fn wait_for_ready<T>(
        self,
        client: &ScopedJoinHandle<'_, T>,
    ) -> Result<Option<ProfileRelease>, InferlabError> {
        let mut connection = None;
        let mut received = Vec::with_capacity(PROFILE_READY.len());
        loop {
            if connection.is_none() {
                match self.listener.accept() {
                    Ok((stream, _)) => {
                        stream.set_nonblocking(true).map_err(|source| {
                            InferlabError::ProfileBarrierIo {
                                operation: "configure the AIPerf connection",
                                source,
                            }
                        })?;
                        connection = Some(stream);
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        if client.is_finished() {
                            return Ok(None);
                        }
                    }
                    Err(source) => {
                        return Err(InferlabError::ProfileBarrierIo {
                            operation: "accept the AIPerf connection",
                            source,
                        });
                    }
                }
            }
            if let Some(stream) = connection.as_mut() {
                let mut buffer = [0_u8; 32];
                match stream.read(&mut buffer) {
                    Ok(0) => {
                        return Err(InferlabError::ProfileBarrierProtocol {
                            message: "AIPerf closed before reporting profiling readiness"
                                .to_owned(),
                        });
                    }
                    Ok(count) => {
                        received.extend_from_slice(&buffer[..count]);
                        if received == PROFILE_READY {
                            return Ok(connection.map(|stream| ProfileRelease { stream }));
                        }
                        if received.ends_with(b"\n") || received.len() >= PROFILE_READY.len() {
                            return Err(InferlabError::ProfileBarrierProtocol {
                                message: format!(
                                    "AIPerf reported an invalid readiness message {:?}",
                                    String::from_utf8_lossy(&received)
                                ),
                            });
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        if client.is_finished() {
                            return Err(InferlabError::ProfileBarrierProtocol {
                                message: "AIPerf exited before completing its readiness message"
                                    .to_owned(),
                            });
                        }
                    }
                    Err(source) => {
                        return Err(InferlabError::ProfileBarrierIo {
                            operation: "read AIPerf profiling readiness",
                            source,
                        });
                    }
                }
            }
            thread::sleep(CLIENT_POLL_INTERVAL);
        }
    }
}

pub(super) struct ProfileRelease {
    stream: TcpStream,
}

impl ProfileRelease {
    pub(super) fn acknowledge(mut self) -> Result<(), InferlabError> {
        self.stream
            .write_all(CAPTURE_OPEN)
            .map_err(|source| InferlabError::ProfileBarrierIo {
                operation: "acknowledge the open capture window",
                source,
            })
    }
}
