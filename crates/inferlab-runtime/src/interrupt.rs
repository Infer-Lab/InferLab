use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static SIGNALS: AtomicU64 = AtomicU64::new(0);
static CTRL_C_HANDLER: OnceLock<Result<(), InterruptInstallError>> = OnceLock::new();

#[derive(Clone, Debug, thiserror::Error)]
pub enum InterruptInstallError {
    #[error(
        "failed to install Ctrl-C handler: Ctrl-C error: Signal could not be found from the system ({signal})"
    )]
    NoSuchSignal { signal: String },
    #[error(
        "failed to install Ctrl-C handler: Ctrl-C error: Ctrl-C signal handler already registered"
    )]
    MultipleHandlers,
    #[error("failed to install Ctrl-C handler: Ctrl-C error: Unexpected system error")]
    System {
        #[source]
        source: Arc<std::io::Error>,
    },
}

impl From<ctrlc::Error> for InterruptInstallError {
    fn from(source: ctrlc::Error) -> Self {
        match source {
            ctrlc::Error::NoSuchSignal(signal) => Self::NoSuchSignal {
                signal: format!("{signal:?}"),
            },
            ctrlc::Error::MultipleHandlers => Self::MultipleHandlers,
            ctrlc::Error::System(source) => Self::System {
                source: Arc::new(source),
            },
        }
    }
}

pub fn prepare() -> Result<(), InterruptInstallError> {
    INTERRUPTED.store(false, Ordering::SeqCst);
    CTRL_C_HANDLER
        .get_or_init(|| {
            ctrlc::set_handler(|| {
                SIGNALS.fetch_add(1, Ordering::SeqCst);
                INTERRUPTED.store(true, Ordering::SeqCst);
            })
            .map_err(InterruptInstallError::from)
        })
        .clone()
}

pub fn received() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

/// How many operator interrupt signals this process has received. A wait
/// that must end only on an interrupt arriving while it runs — such as a
/// cleanup completion wait, which often runs because of an earlier interrupt
/// ([[RFC-0009:C-COMPLETION-WAITS]]) — compares this count with the one it
/// started from. A stop request raised through [`request`] is not counted.
pub fn signals() -> u64 {
    SIGNALS.load(Ordering::SeqCst)
}

/// Raise the same interruption an operator's Ctrl-C raises, for a stop
/// request delivered by other means ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub fn request() {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::InterruptInstallError;

    #[test]
    fn ctrlc_categories_remain_typed() {
        let missing =
            InterruptInstallError::from(ctrlc::Error::NoSuchSignal(ctrlc::SignalType::Ctrlc));
        assert!(matches!(
            missing,
            InterruptInstallError::NoSuchSignal { .. }
        ));

        let duplicate = InterruptInstallError::from(ctrlc::Error::MultipleHandlers);
        assert!(matches!(duplicate, InterruptInstallError::MultipleHandlers));

        let system = InterruptInstallError::from(ctrlc::Error::System(std::io::Error::other(
            "handler installation failed",
        )));
        assert!(matches!(system, InterruptInstallError::System { .. }));
    }
}
