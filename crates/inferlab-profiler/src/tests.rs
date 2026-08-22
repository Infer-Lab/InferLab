use crate::error::ProfilerError;
use crate::plan::{
    CaptureDeadlines, CaptureWindowActionPlan, CaptureWindowControlEndpointPlan,
    CaptureWindowHttpMethodPlan, NsysEscapes, PreparedProcess, ProcessCapturePlan,
    ProcessPreparation, ProfilerControl, ProfilerFinalization, WindowControlKind, compile_plan,
    prepare_process,
};
use crate::record::CaptureActionRecord;
use crate::transport;
use inferlab_protocol::CaptureMechanism;
use inferlab_runtime::operation_bound::OperationBound;
use inferlab_runtime::plan::{CommandPlan, LaunchPlan, ProcessEndpointPlan};
use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Duration;

struct ProcessFixture {
    command: CommandPlan,
    launch: LaunchPlan,
    capture: ProcessCapturePlan,
    endpoint: ProcessEndpointPlan,
}

fn process() -> ProcessFixture {
    ProcessFixture {
        command: CommandPlan {
            argv: vec!["pixi".to_owned(), "run".to_owned(), "vllm".to_owned()],
            env: BTreeMap::new(),
            explicit_env: Vec::new(),
            pass_env: Vec::new(),
            cwd: PathBuf::from("/workspace/.inferlab"),
        },
        launch: LaunchPlan::Local,
        endpoint: ProcessEndpointPlan {
            host: "127.0.0.1".to_owned(),
            port: 8000,
        },
        capture: ProcessCapturePlan {
            mechanism: CaptureMechanism::ManagedCollection,
            capture_storage: None,
            window_control_endpoint: CaptureWindowControlEndpointPlan::ReplicaEntry,
            control_process_id: "prefill-0".to_owned(),
            device_count: 1,
            start: CaptureWindowActionPlan {
                method: CaptureWindowHttpMethodPlan::Post,
                path: "/start_profile".to_owned(),
                body: None,
                effective_url: "http://127.0.0.1:8000/start_profile".to_owned(),
            },
            stop: CaptureWindowActionPlan {
                method: CaptureWindowHttpMethodPlan::Post,
                path: "/stop_profile".to_owned(),
                body: None,
                effective_url: "http://127.0.0.1:8000/stop_profile".to_owned(),
            },
            escapes: NsysEscapes::default(),
        },
    }
}

fn prepare_fixture(
    record_id: &str,
    process: &ProcessFixture,
) -> Result<PreparedProcess, ProfilerError> {
    prepare_process(ProcessPreparation {
        record_id,
        role_id: "prefill",
        replica_id: "prefill",
        replica_index: 0,
        process_id: "prefill-0",
        rank: Some(0),
        rank_count: Some(1),
        command: &process.command,
        launch: &process.launch,
        capture: Some(&process.capture),
        control_endpoint: Some(&process.endpoint),
    })
}

#[test]
fn prepares_profiled_process_without_changing_the_serving_command() -> Result<(), Box<dyn Error>> {
    let process = process();
    let prepared = prepare_fixture("20260701-120000-serve", &process)?;
    let target = prepared.target.ok_or("missing profiler target")?;
    assert_eq!(target.role_id, "prefill");
    assert_eq!(target.finalization, ProfilerFinalization::NsysStop);
    assert_eq!(prepared.command.argv[..2], ["nsys", "launch"]);
    assert_eq!(
        prepared.command.argv[prepared.command.argv.len() - 3..],
        ["pixi", "run", "vllm"]
    );
    assert_eq!(
        target.runtime_root,
        PathBuf::from("/workspace/.inferlab/runtime/20260701-120000-serve/prefill-0/profiles")
    );
    let ProfilerControl::Http {
        window_control_endpoint,
        process_id,
        start,
        stop,
        ..
    } = &target.control;
    assert_eq!(
        *window_control_endpoint,
        CaptureWindowControlEndpointPlan::ReplicaEntry
    );
    assert_eq!(process_id, "prefill-0");
    assert_eq!(start.method, CaptureWindowHttpMethodPlan::Post);
    assert_eq!(start.path, "/start_profile");
    assert_eq!(start.effective_url, "http://127.0.0.1:8000/start_profile");
    assert_eq!(stop.method, CaptureWindowHttpMethodPlan::Post);
    assert_eq!(stop.path, "/stop_profile");
    assert_eq!(stop.effective_url, "http://127.0.0.1:8000/stop_profile");
    Ok(())
}

fn escapes() -> NsysEscapes {
    NsysEscapes {
        executable: Some("nsys-custom".to_owned()),
        launch_options: vec!["--cuda-graph-trace=node".to_owned()],
        start_options: vec!["--nic-metrics=true".to_owned()],
        trace: vec!["cuda".to_owned(), "nvtx".to_owned()],
        sampling: Some("cpu".to_owned()),
        context_switch: Some("process-tree".to_owned()),
        env: BTreeMap::from([("NSYS_FIXTURE".to_owned(), "a b".to_owned())]),
    }
}

#[test]
fn escapes_splice_ahead_of_the_managed_launch_tail() -> Result<(), Box<dyn Error>> {
    let mut process = process();
    process.capture.escapes = escapes();
    let target = prepare_fixture("serve", &process)?
        .target
        .ok_or("missing profiler target")?;
    assert_eq!(
        target.launch_prefix,
        [
            "env",
            "--",
            "NSYS_FIXTURE=a b",
            "nsys-custom",
            "launch",
            "--cuda-graph-trace=node",
            "--session-new",
            "inferlab-serve-prefill-0",
            "--trace=cuda,nvtx",
            "--wait=all",
        ]
    );
    assert_eq!(target.executable, "nsys-custom");
    assert_eq!(target.escapes, escapes());
    Ok(())
}

#[test]
fn escapes_splice_ahead_of_the_managed_start_tail() -> Result<(), Box<dyn Error>> {
    let mut process = process();
    process.capture.escapes = escapes();
    let target = prepare_fixture("serve", &process)?
        .target
        .ok_or("missing profiler target")?;
    let bound = OperationBound::finite(Duration::from_secs(1));
    let action = transport::arm_range_collection(&target, Path::new("/profiles/trace"), 2, &bound);
    let CaptureActionRecord::Command { argv, .. } = action else {
        return Err("profiler start fixture returned non-command evidence".into());
    };
    assert_eq!(
        argv,
        [
            "env",
            "--",
            "NSYS_FIXTURE=a b",
            "nsys-custom",
            "start",
            "--nic-metrics=true",
            "--session=inferlab-serve-prefill-0",
            "--sample=cpu",
            "--cpuctxsw=process-tree",
            "--force-overwrite=true",
            "--export=none",
            "--output=/profiles/trace",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=repeat:2:async",
        ]
    );
    Ok(())
}

#[test]
fn static_range_plan_maps_windows_to_one_based_reports() -> Result<(), Box<dyn Error>> {
    let process = process();
    let target = prepare_fixture("serve", &process)?
        .target
        .ok_or("missing profiler target")?;
    let plan = compile_plan(
        "serve",
        "bench-c8k1k",
        &["c1".to_owned(), "c32".to_owned()],
        &[target],
        CaptureDeadlines {
            capture_arm_deadline_seconds: 60,
            capture_control_deadline_seconds: 60,
            capture_finalization_deadline_seconds: 300,
        },
    )?;
    assert_eq!(plan.control, WindowControlKind::FrameworkRange);
    assert_eq!(plan.windows[0].range_index, Some(1));
    assert_eq!(plan.windows[1].range_index, Some(2));
    assert_eq!(plan.targets[0].expected_range_count, Some(2));
    assert!(plan.targets[0].reports[1].ends_with("trace.2.nsys-rep"));
    Ok(())
}

#[test]
fn engine_trace_preparation_leaves_the_serving_command_unwrapped() -> Result<(), Box<dyn Error>> {
    let mut process = process();
    process.capture.mechanism = CaptureMechanism::EngineTrace;
    process.capture.capture_storage = Some(PathBuf::from(
        "/workspace/.inferlab/runtime/engine-trace/serve/prefill",
    ));
    let prepared = prepare_fixture("20260701-120000-serve", &process)?;
    let target = prepared.target.ok_or("missing profiler target")?;

    assert_eq!(prepared.command.argv, ["pixi", "run", "vllm"]);
    assert_eq!(target.mechanism, CaptureMechanism::EngineTrace);
    assert_eq!(target.finalization, ProfilerFinalization::EngineTraceFlush);
    assert_eq!(
        target.trace_storage.as_deref(),
        Some(Path::new(
            "/workspace/.inferlab/runtime/engine-trace/serve/prefill"
        ))
    );
    assert!(target.launch_prefix.is_empty());
    let plan = compile_plan(
        "serve",
        "bench-c8k1k",
        &["c1".to_owned()],
        std::slice::from_ref(&target),
        CaptureDeadlines {
            capture_arm_deadline_seconds: 60,
            capture_control_deadline_seconds: 60,
            capture_finalization_deadline_seconds: 300,
        },
    )?;
    assert_eq!(plan.windows[0].range_index, None);
    assert_eq!(plan.targets[0].expected_range_count, None);
    assert!(plan.targets[0].reports.is_empty());
    assert_eq!(
        plan.targets[0].output_base,
        PathBuf::from("/workspace/.inferlab/runtime/engine-trace/serve/prefill")
    );
    Ok(())
}

#[test]
fn engine_trace_coverage_requires_one_new_artifact_per_device() -> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let trace_dir = temp.path().join("trace");
    std::fs::create_dir_all(&trace_dir)?;
    std::fs::write(trace_dir.join("stale.trace.json"), b"old")?;
    let baseline = crate::finalization::snapshot_trace_files(&trace_dir)?;
    let bound = OperationBound::finite(Duration::from_millis(50));

    let missing = crate::finalization::verify_engine_trace_coverage(
        "serve", &trace_dir, 2, &baseline, &bound,
    );
    assert!(!missing.verified);
    assert!(missing.new_files.is_empty());
    assert_eq!(missing.expected_artifacts, 2);

    std::fs::write(trace_dir.join("rank-0.trace.json"), b"0")?;
    std::fs::write(trace_dir.join("rank-1.trace.json"), b"1")?;
    let covered = crate::finalization::verify_engine_trace_coverage(
        "serve", &trace_dir, 2, &baseline, &bound,
    );
    assert!(covered.verified);
    assert_eq!(
        covered.new_files,
        [
            PathBuf::from("rank-0.trace.json"),
            PathBuf::from("rank-1.trace.json")
        ]
    );
    Ok(())
}

#[test]
fn engine_trace_coverage_polls_past_an_early_frontend_artifact() -> Result<(), Box<dyn Error>> {
    // The vLLM TP2 shape: the replica has one entry process (rank_count 1)
    // but two devices, and the frontend async_llm trace lands before the
    // worker traces. The device-count baseline must reject the frontend-only
    // delta and keep polling until the second artifact lands.
    let temp = tempfile::tempdir()?;
    let trace_dir = temp.path().join("trace");
    std::fs::create_dir_all(&trace_dir)?;
    let baseline = crate::finalization::snapshot_trace_files(&trace_dir)?;
    std::fs::write(trace_dir.join("async_llm.trace.json"), b"frontend")?;
    let worker_dir = trace_dir.clone();
    let flush = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        std::fs::write(worker_dir.join("worker-rank-1.trace.json"), b"1")
    });

    let bound = OperationBound::finite(Duration::from_secs(5));
    let coverage = crate::finalization::verify_engine_trace_coverage(
        "serve", &trace_dir, 2, &baseline, &bound,
    );
    flush.join().map_err(|_| "worker flush thread panicked")??;

    assert!(coverage.verified);
    assert_eq!(coverage.expected_artifacts, 2);
    assert_eq!(
        coverage.new_files,
        [
            PathBuf::from("async_llm.trace.json"),
            PathBuf::from("worker-rank-1.trace.json")
        ]
    );

    // The same shape without the second artifact exhausts the budget
    // unverified: a frontend trace alone must not pass a 2-device replica.
    let temp = tempfile::tempdir()?;
    let trace_dir = temp.path().join("trace");
    std::fs::create_dir_all(&trace_dir)?;
    let baseline = crate::finalization::snapshot_trace_files(&trace_dir)?;
    std::fs::write(trace_dir.join("async_llm.trace.json"), b"frontend")?;
    let bound = OperationBound::finite(Duration::from_millis(200));
    let started = std::time::Instant::now();

    let coverage = crate::finalization::verify_engine_trace_coverage(
        "serve", &trace_dir, 2, &baseline, &bound,
    );

    assert!(!coverage.verified);
    assert_eq!(coverage.new_files, [PathBuf::from("async_llm.trace.json")]);
    assert!(
        started.elapsed() >= Duration::from_millis(200),
        "an insufficient delta must consume the polling budget"
    );
    Ok(())
}

#[test]
fn engine_trace_coverage_stops_polling_on_a_snapshot_failure() -> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let missing_dir = temp.path().join("gone");
    // A generous budget: if the snapshot error were swallowed into "zero new
    // files", the check would burn the whole budget before answering.
    let bound = OperationBound::finite(Duration::from_secs(60));
    let started = std::time::Instant::now();

    let coverage = crate::finalization::verify_engine_trace_coverage(
        "serve",
        &missing_dir,
        1,
        &Default::default(),
        &bound,
    );

    assert!(!coverage.verified);
    assert!(coverage.new_files.is_empty());
    let error = coverage.error.as_deref().ok_or("missing snapshot error")?;
    assert!(error.contains("failed to list trace directory"), "{error}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "snapshot failure must end polling immediately"
    );
    Ok(())
}

#[test]
fn trace_snapshot_does_not_descend_through_symlinks() -> Result<(), Box<dyn Error>> {
    let temp = tempfile::tempdir()?;
    let trace_dir = temp.path().join("trace");
    let outside = temp.path().join("outside");
    std::fs::create_dir_all(&trace_dir)?;
    std::fs::create_dir_all(&outside)?;
    std::fs::write(outside.join("foreign.trace.json"), b"foreign")?;
    std::os::unix::fs::symlink(&outside, trace_dir.join("linked"))?;
    std::fs::write(trace_dir.join("rank-0.trace.json"), b"0")?;

    let files = crate::finalization::snapshot_trace_files(&trace_dir)?;

    assert_eq!(
        files.iter().cloned().collect::<Vec<_>>(),
        [PathBuf::from("linked"), PathBuf::from("rank-0.trace.json")]
    );
    Ok(())
}
