//! The InferLab context injected into ad-hoc local execution
//! ([[RFC-0002:C-ADHOC-EXECUTION]]): facts InferLab is authoritative for,
//! offered to the command under one `INFERLAB_` namespace. The command may
//! ignore every variable; nothing here verifies use.

use crate::InferlabError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// The context contract version carried by [`CONTEXT`].
pub(super) const CONTEXT_VERSION: &str = "1";

const NAMESPACE: &str = "INFERLAB_";
const CONTEXT: &str = "INFERLAB_CONTEXT";
const WORKSPACE_ROOT: &str = "INFERLAB_WORKSPACE_ROOT";
pub(super) const RECORD_ID: &str = "INFERLAB_RECORD_ID";
pub(super) const RECORD_ARTIFACTS: &str = "INFERLAB_RECORD_ARTIFACTS";
const SERVE_RECORD: &str = "INFERLAB_SERVE_RECORD";
const SERVE_BASE_URL: &str = "INFERLAB_SERVE_BASE_URL";
const SERVE_MODEL: &str = "INFERLAB_SERVE_MODEL";
const MODEL_ID: &str = "INFERLAB_MODEL_ID";
const MODEL_PATH: &str = "INFERLAB_MODEL_PATH";

/// Every variable the context defines; an inherited one without the
/// [`CONTEXT`] marker is operator misuse.
const DEFINED: [&str; 9] = [
    CONTEXT,
    WORKSPACE_ROOT,
    RECORD_ID,
    RECORD_ARTIFACTS,
    SERVE_RECORD,
    SERVE_BASE_URL,
    SERVE_MODEL,
    MODEL_ID,
    MODEL_PATH,
];

/// The inherited environment's relation to an outer InferLab context.
pub(super) struct Inherited {
    /// Every inherited `INFERLAB_*` name a nested run replaces.
    pub strip: Vec<String>,
    /// The outer run's record, when the outer run was recorded.
    pub parent_record: Option<String>,
}

/// Classify the inherited environment: with the marker the command runs
/// nested and every inherited `INFERLAB_*` variable is replaced; without it,
/// any context-defined variable fails before execution.
pub(super) fn inherited() -> Result<Inherited, InferlabError> {
    let names = std::env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| name.starts_with(NAMESPACE))
        .collect::<Vec<_>>();
    if std::env::var_os(CONTEXT).is_some() {
        return Ok(Inherited {
            strip: names,
            parent_record: std::env::var(RECORD_ID).ok(),
        });
    }
    let mut misplaced = names
        .into_iter()
        .filter(|name| DEFINED.contains(&name.as_str()))
        .collect::<Vec<_>>();
    if misplaced.is_empty() {
        return Ok(Inherited {
            strip: Vec::new(),
            parent_record: None,
        });
    }
    misplaced.sort();
    Err(InferlabError::AdHocRun {
        message: format!(
            "the environment sets InferLab context variables {misplaced:?} without \
             {CONTEXT}; InferLab provides them, so unset them and rerun"
        ),
    })
}

/// A running server's public endpoint and served model, as `--serve` links
/// them.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct ServeContext {
    pub record_id: String,
    pub base_url: String,
    pub model: String,
}

/// The `model_weights` binding `--model` selects, resolved for this machine.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct ModelContext {
    pub id: String,
    pub locator: String,
}

/// The variables provided to one command, in name order.
pub(super) fn variables(
    root: &Path,
    record: Option<(&str, &Path)>,
    serve: Option<&ServeContext>,
    model: Option<&ModelContext>,
) -> BTreeMap<String, String> {
    let mut variables = BTreeMap::new();
    let mut set = |name: &str, value: String| {
        variables.insert(name.to_owned(), value);
    };
    set(CONTEXT, CONTEXT_VERSION.to_owned());
    set(WORKSPACE_ROOT, root.display().to_string());
    if let Some((id, artifacts)) = record {
        set(RECORD_ID, id.to_owned());
        set(RECORD_ARTIFACTS, artifacts.display().to_string());
    }
    if let Some(serve) = serve {
        set(SERVE_RECORD, serve.record_id.clone());
        set(SERVE_BASE_URL, serve.base_url.clone());
        set(SERVE_MODEL, serve.model.clone());
    }
    if let Some(model) = model {
        set(MODEL_ID, model.id.clone());
        set(MODEL_PATH, model.locator.clone());
    }
    variables
}

/// Resolve `--serve`: the record must be running with every process observed
/// alive, and its public endpoint becomes a base URL without a path.
pub(super) fn serve(root: &Path, record_id: &str) -> Result<ServeContext, InferlabError> {
    let report = crate::server::status(root, record_id)?;
    crate::server::require_running(&report)?;
    let server = &report.record.resolved.server;
    let scheme = match server.endpoint.protocol {
        inferlab_protocol::EndpointProtocol::Http => "http",
    };
    let host = &server.endpoint.host;
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.clone()
    };
    Ok(ServeContext {
        record_id: record_id.to_owned(),
        base_url: format!("{scheme}://{host}:{}", server.endpoint.port),
        model: server.model.served_name.clone(),
    })
}

/// Resolve `--model` against the strictly loaded local bindings for the
/// locally launching machine ([[RFC-0002:C-LOCAL-PLACEMENT]]).
pub(super) fn model(root: &Path, id: &str) -> Result<ModelContext, InferlabError> {
    let workspace = crate::workspace::load_workspace(root.to_path_buf(), None)?;
    let locator = workspace.local.local_model_locator(id)?;
    Ok(ModelContext {
        id: id.to_owned(),
        locator,
    })
}
