use std::fmt;

/// Lifecycle/setup error for standing a built-in proxy up; per-request
/// failures are [`crate::core::ProxyHttpError`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProxyError {
    Io { message: String },
    ExternalTool { message: String },
    Invalid { message: String },
    Lifecycle { message: String },
}

impl fmt::Display for ProxyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { message }
            | Self::ExternalTool { message }
            | Self::Invalid { message }
            | Self::Lifecycle { message } => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ProxyError {}
