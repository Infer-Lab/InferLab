//! Image-backed launch substitution ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
//!
//! `serve start` and `recipe run` accept an image build record selection:
//! the record's host-platform assembled image substitutes for the locally
//! installed serving environment through the same containerized substitution
//! image validation uses. Selection compatibility is validated here at
//! resolution, before any external effect, against the facts the image
//! record already owns.

use crate::InferlabError;
use crate::execution::ResolvedExecution;
use crate::image::record::{AssemblyOutcome, ImageRecord};
use crate::workspace::LoadedWorkspace;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// The image substitution facts preserved in the execution plan and every
/// record embedding it ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): the qualifying
/// image build record, the immutable image identity launched, and the
/// workspace revision the image was built from. The invoking revision lives
/// in the execution's own workspace snapshot, so drift between the revision
/// that qualified the image and the revision that launches it stays
/// observable.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ImageLaunchPlan {
    pub record_id: String,
    pub image_id: String,
    pub platform: String,
    pub workspace_revision: String,
}

/// A selected external serving image ([[RFC-0003:C-RUNTIME-WORKFLOWS]]):
/// a digest-pinned image this workspace did not build, launched as an
/// explicitly not-qualified realization. The observed framework version is
/// the only qualification-adjacent evidence available and is filled during
/// resolution.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ExternalImagePlan {
    pub id: String,
    pub reference: String,
    pub digest: String,
    pub integration: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub framework_version: Option<String>,
    #[serde(skip)]
    pub framework_probe_timing: Option<inferlab_runtime::operation_bound::OperationTimingEvidence>,
}

/// Validate an image selection against the recipe's workspace facts and the
/// stored record, before resolution begins: every fact these rejections need
/// is already known, so an incompatible selection never reaches an
/// integration invocation ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub(crate) fn select(
    workspace: &LoadedWorkspace,
    server_id: &str,
    record_id: &str,
) -> Result<ImageLaunchPlan, InferlabError> {
    let Some(server) = workspace.config.servers.get(server_id) else {
        return Err(InferlabError::InvalidConfig {
            message: format!("unknown server {server_id:?}"),
        });
    };
    let record = load_record(&workspace.root, record_id)?;
    if record.resolved.image.stack != server.stack {
        return Err(reject(format!(
            "image build record {record_id} built stack {:?} but server {server_id:?} \
             selects {:?}; an image-backed launch must run the serving stack the image contains",
            record.resolved.image.stack, server.stack
        )));
    }
    let (image_id, platform) = host_assembly(&record, record_id)?;
    Ok(ImageLaunchPlan {
        record_id: record_id.to_owned(),
        image_id,
        platform,
        workspace_revision: record.resolved.workspace.revision,
    })
}

/// The record's successful assembly for the launching host's platform, the
/// shared precondition of every image-backed execution
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
fn host_assembly(record: &ImageRecord, record_id: &str) -> Result<(String, String), InferlabError> {
    let platform = host_platform();
    let Some(assembly) = record
        .assemblies
        .iter()
        .find(|assembly| assembly.platform == platform)
    else {
        let assembled: Vec<&str> = record
            .assemblies
            .iter()
            .map(|assembly| assembly.platform.as_str())
            .collect();
        return Err(reject(format!(
            "image build record {record_id} holds no assembly for host platform {platform:?} \
             (assemblies: {assembled:?})"
        )));
    };
    let image_id = match &assembly.outcome {
        AssemblyOutcome::Assembled { image_id, .. } => image_id.clone(),
        AssemblyOutcome::Pending => {
            return Err(reject(format!(
                "the {platform} assembly of image build record {record_id} did not succeed \
                 (never assembled)"
            )));
        }
        AssemblyOutcome::Failed { message } => {
            return Err(reject(format!(
                "the {platform} assembly of image build record {record_id} did not succeed: \
                 {message}"
            )));
        }
    };
    Ok((image_id, platform))
}

/// Select an image build record for ad-hoc execution
/// ([[RFC-0002:C-ADHOC-EXECUTION]]): no recipe participates, so the record's
/// stored facts and a successful host-platform assembly are the only
/// preconditions, and the executed environment is structurally the one the
/// record's image realizes.
pub(crate) fn select_for_adhoc(root: &Path, record_id: &str) -> Result<String, InferlabError> {
    let record = load_record(root, record_id)?;
    let (image_id, _) = host_assembly(&record, record_id)?;
    Ok(image_id)
}

/// Validate an external-image selection against workspace facts, before
/// resolution begins ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): the declaration
/// must exist, its integration claim must match the server stack's integration,
/// and the image must already be present locally — Inferlab never pulls.
pub(crate) fn select_external(
    workspace: &LoadedWorkspace,
    server_id: &str,
    external_id: &str,
) -> Result<ExternalImagePlan, InferlabError> {
    let Some(server) = workspace.config.servers.get(server_id) else {
        return Err(InferlabError::InvalidConfig {
            message: format!("unknown server {server_id:?}"),
        });
    };
    let Some(declaration) = workspace.config.external_images.get(external_id) else {
        return Err(reject(format!(
            "unknown external image {external_id:?}; declare it under [external_images] in the \
             workspace"
        )));
    };
    let Some(stack) = workspace.config.stacks.get(&server.stack) else {
        return Err(InferlabError::InvalidConfig {
            message: format!("unknown stack {:?}", server.stack),
        });
    };
    if stack.integration != declaration.integration {
        return Err(reject(format!(
            "external image {external_id:?} claims integration {:?} but server {server_id:?} \
             serves through integration {:?}; an external image must answer the serving stack \
             the server expects",
            declaration.integration, stack.integration
        )));
    }
    probe_local_presence(external_id, &declaration.reference)?;
    let digest = declaration
        .reference
        .rsplit_once('@')
        .map(|(_, digest)| digest.to_owned())
        .ok_or_else(|| InferlabError::InvalidConfig {
            message: format!(
                "external image {external_id:?} reference lost its digest after validation"
            ),
        })?;
    Ok(ExternalImagePlan {
        id: external_id.to_owned(),
        reference: declaration.reference.clone(),
        digest,
        integration: declaration.integration.clone(),
        framework_version: None,
        framework_probe_timing: None,
    })
}

/// The placement gate for an image selection, executed as soon as processes
/// are planned — before network resolution or remote-machine preflight
/// consumes a placement the substitution cannot serve
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub(crate) fn gate_placement(
    processes: &[crate::execution::ProcessPlan],
) -> Result<(), InferlabError> {
    if let Err(reason) = super::single_host_local(processes) {
        return Err(reject(format!(
            "image-backed launch requires the single-host local placement the containerized \
             substitution supports: {reason}; a built image exists only in its builder's \
             storage and Inferlab defines no image distribution between machines — declare \
             a pulled external image to serve a distributed placement"
        )));
    }
    Ok(())
}

/// A host-side profiler would wrap the container client and observe no
/// server process, so the capture would claim a profile it never took;
/// rejected until an in-container profiler contract exists
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
fn gate_capture<'a>(
    processes: impl IntoIterator<Item = &'a crate::execution::ProcessPlan>,
) -> Result<(), InferlabError> {
    if let Some(process) = processes
        .into_iter()
        .find(|process| process.capture_target.is_some())
    {
        return Err(reject(format!(
            "server process {:?} is prepared as a profiling capture target, but profiling an \
             image-backed launch requires an in-container profiler, which Inferlab does not \
             define yet; launch without capture or profile a launch from the locally \
             installed serving environment",
            process.id
        )));
    }
    Ok(())
}

/// Apply a validated external-image selection: gate the plan shapes the
/// substitution cannot serve, observe the framework version inside the
/// image, rewrite the server commands through an explicit command override,
/// and preserve the not-qualified evidence
/// ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub(crate) fn apply_external(
    execution: &mut ResolvedExecution,
    external: &ExternalImagePlan,
    machines: &std::collections::BTreeMap<String, crate::workspace::MachineBinding>,
    adapter: &crate::workspace::AdapterBinding,
) -> Result<(), InferlabError> {
    gate_capture(execution.server.processes())?;
    let framework = execution.server.integration.framework.clone();
    let probe = crate::adapter::probe_external_framework(
        &external.reference,
        adapter.image_device,
        adapter.image_timeout(),
        &framework,
    )?;
    containerize(execution, &external.reference, machines, true)?;
    execution.server.external_image = Some(ExternalImagePlan {
        framework_version: Some(probe.version),
        framework_probe_timing: Some(probe.timing),
        ..external.clone()
    });
    Ok(())
}

/// Apply the validated selection to the resolved execution: gate the plan
/// shapes the substitution cannot serve, rewrite the server commands, and
/// preserve the selection as evidence ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
pub(crate) fn apply(
    execution: &mut ResolvedExecution,
    image: &ImageLaunchPlan,
    machines: &std::collections::BTreeMap<String, crate::workspace::MachineBinding>,
) -> Result<(), InferlabError> {
    gate_capture(execution.server.processes())?;
    containerize(execution, &image.image_id, machines, false)?;
    execution.server.image = Some(image.clone());
    Ok(())
}

/// A read-only presence probe: a missing image is the operator's pull to
/// make, never Inferlab's.
fn probe_local_presence(external_id: &str, reference: &str) -> Result<(), InferlabError> {
    let inspect = std::process::Command::new("docker")
        .args(["image", "inspect", "--format", "{{.Id}}", reference])
        .output()
        .map_err(|source| InferlabError::ImageToolLaunch {
            operation: format!("docker image inspect {reference}"),
            source,
        })?;
    if !inspect.status.success() {
        return Err(reject(format!(
            "external image {external_id:?} ({reference}) is not present in local builder \
             storage; run: docker pull {reference}"
        )));
    }
    Ok(())
}

/// Select a declared external serving image for ad-hoc execution
/// ([[RFC-0002:C-ADHOC-EXECUTION]]): the declaration must exist and the
/// image must already be present locally. No recipe participates, so the
/// launch surface's integration agreement has no counterpart to check
/// against; the execution is not qualified by this workspace either way.
pub(crate) fn select_external_for_adhoc(
    config: &crate::workspace::WorkspaceConfig,
    external_id: &str,
) -> Result<String, InferlabError> {
    let Some(declaration) = config.external_images.get(external_id) else {
        return Err(reject(format!(
            "unknown external image {external_id:?}; declare it under [external_images] in the \
             workspace"
        )));
    };
    probe_local_presence(external_id, &declaration.reference)?;
    Ok(declaration.reference.clone())
}

fn reject(message: String) -> InferlabError {
    InferlabError::ImageSelection { message }
}

fn load_record(root: &Path, record_id: &str) -> Result<ImageRecord, InferlabError> {
    // The shared record discipline: the same id grammar as every record
    // family, and the schema-version gate before the strict parse so a
    // record from before a hard cut surfaces the version error rather than a
    // bare serde variant error ([[RFC-0005:C-EVIDENCE]]).
    // The shared grammar, remapped into the image-selection error class
    // (E4003) the call site owns — the same remap the scratchpad applies
    // ([[RFC-0001:C-ERROR-CODES]]).
    crate::record::validate_record_id("image build record", record_id)
        .map_err(|error| reject(error.to_string()))?;
    let path = root
        .join(super::record::RECORDS_DIR)
        .join(record_id)
        .join(super::record::RECORD_FILE);
    let bytes = std::fs::read(&path).map_err(|source| InferlabError::ImageSelection {
        message: format!(
            "image build record {record_id} is not readable at {}: {source}",
            path.display()
        ),
    })?;
    let header: crate::record::SchemaVersionHeader =
        serde_json::from_slice(&bytes).map_err(|source| InferlabError::RecordDecode {
            path: path.clone(),
            source,
        })?;
    if header.schema_version != ImageRecord::SCHEMA_VERSION {
        return Err(InferlabError::InvalidConfig {
            message: format!(
                "image build record {record_id:?} has unsupported schema version {}; expected {}",
                header.schema_version,
                ImageRecord::SCHEMA_VERSION
            ),
        });
    }
    serde_json::from_slice(&bytes).map_err(|source| InferlabError::RecordDecode { path, source })
}

/// The launching host's OCI platform. The container runs on the invoking
/// host, so the process architecture is the architecture an assembly must
/// declare to run here. This is the selection question — which recorded
/// assembly runs here — answered without invoking a builder; the build-side
/// key those assemblies are recorded under is the daemon-reported platform
/// observed by `crate::image::tool::DockerBuilderTool::host_platform`.
fn host_platform() -> String {
    let arch = match std::env::consts::ARCH {
        "x86_64" => "amd64",
        "aarch64" => "arm64",
        other => other,
    };
    format!("{}/{arch}", std::env::consts::OS)
}

/// The weight locators a containerized launch exposes at their host paths,
/// in mount order: the served model first, then each auxiliary artifact,
/// deduplicated, absolute host paths only
/// ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]).
fn weight_mount_locators<'a>(
    model_locator: Option<&'a str>,
    auxiliary: impl IntoIterator<Item = &'a str>,
) -> Vec<&'a str> {
    let mut locators: Vec<&str> = Vec::new();
    for locator in model_locator.into_iter().chain(auxiliary) {
        if locator.starts_with('/') && !locators.contains(&locator) {
            locators.push(locator);
        }
    }
    locators
}

/// The pixi-wrap separator an integration-rendered command must carry for the
/// containerized substitution: everything after `--` is the container's
/// command. A process missing the seam would launch on the host while the
/// record claims an image-backed launch, so it is an image-selection
/// rejection, not a silent skip (E4003: an image-substitution precondition,
/// not a response validation, which is the E2001 family).
fn containerization_seam<'a>(
    process_id: &str,
    argv: &'a [String],
) -> Result<&'a [String], InferlabError> {
    let Some(separator) = argv.iter().position(|arg| arg == "--") else {
        return Err(reject(format!(
            "server process {process_id:?} rendered a command without the containerization \
             seam (no `--` separator) the image-backed launch requires"
        )));
    };
    let inner = &argv[separator + 1..];
    if inner.is_empty() {
        return Err(reject(format!(
            "server process {process_id:?} rendered an empty command behind the \
             containerization seam"
        )));
    }
    Ok(inner)
}

/// Substitute the built image for the locally installed serving environment:
/// every integration-rendered server process runs inside `docker run` with host
/// networking, its allocated devices, and its model weights and runtime cache
/// mounted at their host paths. Inferlab-owned processes (the built-in proxy)
/// keep their host command.
pub(crate) fn containerize(
    execution: &mut ResolvedExecution,
    image_id: &str,
    machines: &std::collections::BTreeMap<String, crate::workspace::MachineBinding>,
    explicit_entrypoint: bool,
) -> Result<(), InferlabError> {
    let remote = execution.server.placement.remote_containers.clone();
    // One nonce per resolution: the container name is the cleanup handle —
    // a container is a daemon-owned object the process-group kill never
    // reaches — and must not collide with any earlier invocation's leftover
    // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
    let nonce = format!(
        "{:x}-{:x}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis())
            .unwrap_or_default()
    );
    for process in execution.server.processes_mut() {
        if process.command_source != crate::execution::ProcessCommandSource::Integration {
            continue;
        }
        let argv = &process.command.argv;
        // Machine-scoped launch facts for a remote launch: the container
        // user identity and pass-through observations come from that
        // machine's preflight, not from controller state
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        let remote_facts = match &process.launch {
            inferlab_runtime::plan::LaunchPlan::Ssh { .. } => remote.get(&process.machine),
            inferlab_runtime::plan::LaunchPlan::Local => None,
        };
        let inner = containerization_seam(&process.id, argv)?.to_vec();
        let container_name = format!("inferlab-{}-{nonce}", process.id);
        let mut container = vec![
            "docker".to_owned(),
            "run".to_owned(),
            "--rm".to_owned(),
            "--init".to_owned(),
            "--name".to_owned(),
            container_name.clone(),
            "--network".to_owned(),
            "host".to_owned(),
            // Host-process semantics extend to IPC: multi-rank frameworks
            // bootstrap NCCL over shared memory, and docker's default 64MB
            // /dev/shm kills the first multi-device launch (verified with
            // SGLang DP attention on real hardware).
            "--ipc".to_owned(),
            "host".to_owned(),
        ];
        if !process.allocation.devices.is_empty() {
            let devices = process
                .allocation
                .devices
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",");
            container.extend(inferlab_runtime::container::docker_device_args(&devices));
        }
        let binding = machines
            .get(&process.machine)
            .and_then(|machine| machine.container.as_ref());
        // Declared container hardware facts, lowered exactly as declared —
        // host device nodes, the pinned-memory ulimit, and capabilities RDMA
        // transports need; never --privileged
        // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
        if let Some(binding) = binding {
            for device in &binding.devices {
                container.push("--device".to_owned());
                container.push(device.display().to_string());
            }
            if binding.memlock_unlimited {
                container.push("--ulimit".to_owned());
                container.push("memlock=-1".to_owned());
            }
            for capability in &binding.capabilities {
                container.push("--cap-add".to_owned());
                container.push(capability.clone());
            }
        }
        // Auxiliary weights resolve through the same machine-local bindings
        // as the served model, so the container exposes them at the same
        // host paths the rendered command references
        // ([[RFC-0003:C-SERVE-AUXILIARY-MODELS]]).
        for locator in weight_mount_locators(
            process.allocation.model_locator.as_deref(),
            process
                .allocation
                .auxiliary_model_locators
                .iter()
                .map(|entry| entry.locator.as_str()),
        ) {
            container.extend(inferlab_runtime::container::docker_bind_mount_readonly(
                locator, locator,
            ));
        }
        let cache = process.allocation.runtime_cache.path.display().to_string();
        container.push("--volume".to_owned());
        container.push(format!("{cache}:{cache}"));
        // Match host-process semantics: the server runs as the operator, not
        // as container root (which NFS-backed workspaces squash to nobody),
        // and its writable home is the allocated per-process cache. A remote
        // launch takes the identity that machine's preflight observed;
        // controller filesystem metadata says nothing about a remote host.
        if let Some(facts) = remote_facts {
            container.push("--user".to_owned());
            container.push(format!("{}:{}", facts.uid, facts.gid));
        } else if let Ok(metadata) = std::fs::metadata(&process.command.cwd) {
            use std::os::unix::fs::MetadataExt;
            container.push("--user".to_owned());
            container.push(format!("{}:{}", metadata.uid(), metadata.gid()));
        }
        container.push("--env".to_owned());
        container.push(format!("HOME={cache}"));
        // The operator uid has no passwd entry inside the image;
        // `getpass.getuser()` and friends need USER/LOGNAME from the
        // environment.
        let user = remote_facts.map_or_else(
            || std::env::var("USER").unwrap_or_else(|_| "inferlab".to_owned()),
            |facts| facts.user.clone(),
        );
        container.push("--env".to_owned());
        container.push(format!("USER={user}"));
        container.push("--env".to_owned());
        container.push(format!("LOGNAME={user}"));
        // Runtime credentials and other operator-selected variables enter
        // the container by name reference only: docker reads the value from
        // its own environment, so it never appears in the launch argv or the
        // image content ([[RFC-0007:C-IMAGE-BUILD]]).
        if let Some(binding) = binding {
            for name in &binding.pass_env {
                // An absent declaration is a silent no-op in docker and worth
                // surfacing — reported against the launching machine's
                // environment: the local process environment for a local
                // launch, the preflight observation for a remote one.
                let absent = match remote_facts {
                    Some(facts) => !facts.present_pass_env.contains(name),
                    None => std::env::var_os(name).is_none(),
                };
                if absent {
                    eprintln!(
                        "warning: declared pass-through env {name} is not set in the \
                         launching environment of machine {:?}",
                        process.machine
                    );
                }
                container.push("--env".to_owned());
                container.push(name.clone());
                // The spawn layer resolves the value from the launching
                // machine's environment; the plan carries only the names
                // observed present there — an absent declaration stays
                // absent in the container rather than becoming set-but-empty
                // through an expanded shell reference
                // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
                if !absent {
                    process.command.pass_env.push(name.clone());
                }
            }
        }
        // The container receives exactly the resolver- and integration-set
        // variables ([[RFC-0003:C-RUNTIME-WORKFLOWS]]): the explicit set is
        // recorded provenance, never inferred by comparing values against
        // the ambient environment, so an explicit setting that coincides
        // with the host still reaches the image.
        for name in &process.command.explicit_env {
            if name == "CUDA_VISIBLE_DEVICES" {
                // `--gpus device=...` already selects and renumbers the
                // devices inside the container.
                let renumbered = (0..process.allocation.devices.len())
                    .map(|index| index.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                container.push("--env".to_owned());
                container.push(format!("CUDA_VISIBLE_DEVICES={renumbered}"));
                continue;
            }
            if let Some(value) = process.command.env.get(name) {
                container.push("--env".to_owned());
                container.push(format!("{name}={value}"));
            }
        }
        if explicit_entrypoint {
            // An external image fixes its own entrypoint; nothing about it
            // is assumed, so the rendered command replaces it outright
            // ([[RFC-0003:C-RUNTIME-WORKFLOWS]]).
            container.push("--entrypoint".to_owned());
            container.push(inner[0].clone());
            container.push(image_id.to_owned());
            container.extend(inner[1..].to_vec());
        } else {
            container.push(image_id.to_owned());
            container.extend(inner);
        }
        process.command.argv = container;
        process.container = Some(crate::execution::ContainerPlan {
            name: container_name,
            image: image_id.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::weight_mount_locators;

    #[test]
    fn containerize_requires_the_containerization_seam() -> Result<(), Box<dyn std::error::Error>> {
        for argv in [
            vec!["python".to_owned(), "-m".to_owned(), "serve".to_owned()],
            vec!["pixi".to_owned(), "run".to_owned(), "--".to_owned()],
        ] {
            let error = super::containerization_seam("server-0", &argv)
                .err()
                .ok_or("argv without the containerization seam must be rejected")?;
            assert_eq!(error.code(), "E4003", "{error}");
            assert!(error.to_string().contains("\"server-0\""), "{error}");
        }
        let wrapped = [
            "pixi".to_owned(),
            "run".to_owned(),
            "--".to_owned(),
            "serve".to_owned(),
        ];
        let inner = super::containerization_seam("server-0", &wrapped)?;
        assert_eq!(inner, ["serve"]);
        Ok(())
    }

    #[test]
    fn weight_mounts_cover_the_model_and_each_auxiliary_once() {
        let locators = weight_mount_locators(
            Some("/models/served"),
            [
                "/models/draft",
                // A duplicate of the served locator mounts once.
                "/models/served",
                // A relative locator is not a host path the container can
                // expose at itself; only absolute locators mount.
                "relative/draft",
            ],
        );
        assert_eq!(locators, ["/models/served", "/models/draft"]);
    }

    #[test]
    fn weight_mounts_tolerate_a_missing_model_locator() {
        let locators = weight_mount_locators(None, ["/models/draft"]);
        assert_eq!(locators, ["/models/draft"]);
    }

    #[test]
    fn image_record_read_rejects_a_malformed_id() -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let error = super::load_record(temp.path(), "../escape")
            .err()
            .ok_or("a malformed id must be rejected")?;
        assert!(
            error
                .to_string()
                .contains("invalid image build record id \"../escape\""),
            "{error}"
        );
        // The error class, not just the text: consumers may branch on it
        // ([[RFC-0001:C-ERROR-CODES]]).
        assert_eq!(error.code(), "E4003");
        Ok(())
    }

    #[test]
    fn image_record_read_gates_the_schema_version_before_strict_decode()
    -> Result<(), Box<dyn std::error::Error>> {
        let temp = tempfile::tempdir()?;
        let dir = temp
            .path()
            .join(crate::record::RECORDS_DIR)
            .join("legacy-record");
        std::fs::create_dir_all(&dir)?;
        // A record from before the current schema: body content no longer
        // matches the strict shape, and the version gate must fire first.
        std::fs::write(
            dir.join(crate::record::RECORD_FILE),
            "{\"schema_version\":1,\"legacy\":true}\n",
        )?;

        let error = super::load_record(temp.path(), "legacy-record")
            .err()
            .map(|error| error.to_string());
        assert!(
            error.as_deref().is_some_and(|error| error.contains(
                "image build record \"legacy-record\" has unsupported schema version 1; expected 2"
            )),
            "{error:?}"
        );
        Ok(())
    }
}
