use crate::InferlabError;
use serde::Serialize;
use std::path::Path;
use time::OffsetDateTime;

/// The workspace-local state root as a literal: `concat!` accepts only
/// literals, so the per-subdirectory state constants derive from this single
/// spelling rather than from the `STATE_DIR` constant.
macro_rules! state_dir {
    () => {
        ".inferlab"
    };
}
pub(crate) use state_dir;

pub(crate) const STATE_DIR: &str = state_dir!();
pub(crate) const CACHE_DIR: &str = concat!(state_dir!(), "/cache");
pub(crate) const RECORDS_DIR: &str = concat!(state_dir!(), "/records");
pub(crate) const RUNTIME_DIR: &str = concat!(state_dir!(), "/runtime");
pub(crate) const RECORD_FILE: &str = "record.json";

/// This is the shared writer for the workload and recipe records, whose on-disk
/// shape is identical. The server and image record writers deliberately keep
/// their own shapes (a dotfile temp and a non-atomic rewrite, respectively) and
/// do not route through here.
pub(crate) fn write_json(path: &Path, value: &impl Serialize) -> Result<(), InferlabError> {
    crate::atomic_json::write(path, value).map_err(|error| match error {
        crate::atomic_json::AtomicJsonError::Encode(source) => {
            InferlabError::RecordEncode { source }
        }
        crate::atomic_json::AtomicJsonError::Io { path, source, .. } => {
            InferlabError::RecordIo { path, source }
        }
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RecordIdentity<'a> {
    Serve {
        server: &'a str,
        case: Option<&'a str>,
    },
    Recipe {
        recipe: &'a str,
        case: Option<&'a str>,
    },
    Bench {
        bench: &'a str,
    },
    Image {
        image: &'a str,
    },
}

pub(crate) fn new_record_id(identity: RecordIdentity<'_>) -> Result<String, InferlabError> {
    record_id(identity, now_unix_ms()?)
}

pub(crate) fn now_unix_ms() -> Result<u64, InferlabError> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(inferlab_runtime::operation_bound::duration_millis)
        .map_err(|error| InferlabError::ServerLifecycle {
            message: format!("system clock is before Unix epoch: {error}"),
        })
}

/// The lenient version header every record family decodes before its strict
/// parse: only the version, no field policy, so an old record reaches the
/// version gate even when its body predates the current record shape.
#[derive(serde::Deserialize)]
pub(crate) struct SchemaVersionHeader {
    pub(crate) schema_version: u32,
}

pub(crate) fn validate_record_id(kind: &str, id: &str) -> Result<(), InferlabError> {
    if !matches!(id, "." | "..")
        && !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        Ok(())
    } else {
        Err(InferlabError::ServerLifecycle {
            message: format!("invalid {kind} id {id:?}"),
        })
    }
}

pub(crate) fn record_id(
    identity: RecordIdentity<'_>,
    unix_ms: u64,
) -> Result<String, InferlabError> {
    let timestamp = utc_timestamp(unix_ms)?;
    let pid = std::process::id();
    Ok(match identity {
        RecordIdentity::Serve { server, case } => case.map_or_else(
            || format!("{timestamp}-serve-{server}-{pid}"),
            |case| format!("{timestamp}-serve-{server}-{case}-{pid}"),
        ),
        RecordIdentity::Recipe { recipe, case } => case.map_or_else(
            || format!("{timestamp}-recipe-{recipe}-{pid}"),
            |case| format!("{timestamp}-recipe-{recipe}-{case}-{pid}"),
        ),
        RecordIdentity::Bench { bench } => format!("{timestamp}-bench-{bench}-{pid}"),
        RecordIdentity::Image { image } => format!("{timestamp}-image-{image}-{pid}"),
    })
}

/// UTC timestamp in the record-identifier shape (`YYYY-MM-DDTHH-MM-SS.mmmZ`).
///
/// Scratchpad journal entries reuse this exact shape so entries and record
/// identifiers order on one time axis ([[RFC-0005:C-SCRATCHPAD-JOURNAL]]).
pub(crate) fn utc_timestamp(unix_ms: u64) -> Result<String, InferlabError> {
    let unix_nanos = i128::from(unix_ms) * 1_000_000;
    let timestamp = OffsetDateTime::from_unix_timestamp_nanos(unix_nanos).map_err(|error| {
        InferlabError::ServerLifecycle {
            message: format!("record timestamp is outside the supported range: {error}"),
        }
    })?;
    Ok(format!(
        "{:04}-{:02}-{:02}T{:02}-{:02}-{:02}.{:03}Z",
        timestamp.year(),
        u8::from(timestamp.month()),
        timestamp.day(),
        timestamp.hour(),
        timestamp.minute(),
        timestamp.second(),
        timestamp.millisecond(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{CACHE_DIR, RUNTIME_DIR};

    /// CONFIRMATION_CACHE_DIR and OBSERVATIONS_DIR spell their own concat!
    /// segments (concat! takes only literals, so they cannot derive from the
    /// parent constants); the workspace source-identity exclusion set is built
    /// from CACHE_DIR and RUNTIME_DIR, so this pin keeps the parentage
    /// load-bearing instead of coincidental.
    #[test]
    fn second_level_state_dirs_stay_under_their_owned_parents() {
        assert!(
            crate::environment::CONFIRMATION_CACHE_DIR.starts_with(&format!("{CACHE_DIR}/")),
            "CONFIRMATION_CACHE_DIR left the CACHE_DIR subtree"
        );
        assert!(
            crate::operation::OBSERVATIONS_DIR.starts_with(&format!("{RUNTIME_DIR}/")),
            "OBSERVATIONS_DIR left the RUNTIME_DIR subtree"
        );
    }
}
