use crate::error::ProfilerError;
use crate::plan::{
    CaptureDeadlines, CaptureWindowActionPlan, CaptureWindowControlEndpointPlan,
    CaptureWindowHttpMethodPlan, NsysEscapes, PreparedProcess, ProcessCapturePlan,
    ProcessPreparation, ProfilerControl, ProfilerFinalization, WindowControlKind, compile_plan,
    prepare_process,
};
use crate::record::CaptureActionRecord;
use crate::transport;
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
            window_control_endpoint: CaptureWindowControlEndpointPlan::ReplicaEntry,
            control_process_id: "prefill-0".to_owned(),
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
