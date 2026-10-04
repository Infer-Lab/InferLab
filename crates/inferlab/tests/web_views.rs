//! The web console's workspace views ([[RFC-0012:C-VIEWS]]): the TUI's
//! views over HTTP, record pages with metrics and paged logs, and live
//! updates, exercised against the real binary.

mod web_support;

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use web_support::{Console, identity};

const PARENT: &str = "2026-01-02T10-00-00.000Z-recipe-sweep-11";
const SERVER: &str = "2026-01-02T10-00-00.000Z-serve-qwen-11";
const BENCH: &str = "2026-01-02T10-00-00.000Z-recipe-sweep-11-bench-000-random";
const SOLO: &str = "2026-01-01T09-00-00.000Z-bench-latency-7";

fn record(root: &Path, id: &str, json: &serde_json::Value) -> Result<(), Box<dyn Error>> {
    let directory = root.join(".inferlab/records").join(id);
    fs::create_dir_all(&directory)?;
    fs::write(directory.join("record.json"), serde_json::to_vec(json)?)?;
    Ok(())
}

/// A workspace with a failed recipe whose server and Bench children are
/// recorded, the server's log, and a standalone Bench.
fn fixture(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(path.join(".inferlab"))?;
    fs::write(
        path.join(".inferlab/workspace.toml"),
        "schema_version = 2\n",
    )?;
    let day = 1_767_348_000_000_u64; // 2026-01-02 10:00 UTC
    record(
        path,
        PARENT,
        &serde_json::json!({
            "id": PARENT, "status": "failed", "error": "bench case failed",
            "started_unix_ms": day, "finished_unix_ms": day + 600_000,
            "server": {"id": SERVER}, "evals": [], "benches": [{"id": BENCH}],
        }),
    )?;
    let log = format!(".inferlab/records/{SERVER}/stdout.log");
    record(
        path,
        SERVER,
        &serde_json::json!({
            "id": SERVER, "status": "stopped",
            "started_unix_ms": day, "finished_unix_ms": day + 600_000,
            "process_evidence": {"server": {"stdout": log}},
        }),
    )?;
    let lines = (1..=4000)
        .map(|line| format!("server log line {line:04}"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path.join(&log), lines)?;
    record(
        path,
        BENCH,
        &serde_json::json!({
            "id": BENCH, "kind": "bench", "status": "succeeded",
            "started_unix_ms": day, "finished_unix_ms": day + 300_000,
            "cases": [
                {"id": "c1", "status": "succeeded", "metrics": {"request_throughput": 1.25, "p99_ttft_ms": 40.0}},
                {"id": "c8", "status": "succeeded", "metrics": {"request_throughput": 7.5, "p99_ttft_ms": 90.0}},
            ],
        }),
    )?;
    record(
        path,
        SOLO,
        &serde_json::json!({
            "id": SOLO, "kind": "bench", "status": "succeeded",
            "started_unix_ms": day - 86_400_000, "finished_unix_ms": day - 86_000_000,
            "cases": [],
        }),
    )?;
    Ok(path.canonicalize()?)
}

/// A request that carries one case's typed load, as `bench` records it.
fn case_request(
    root: &Path,
    record: &str,
    case: &str,
    load_shape: &str,
) -> Result<String, Box<dyn Error>> {
    let directory = format!(".inferlab/records/{record}/cases/{case}");
    fs::create_dir_all(root.join(&directory))?;
    let artifacts = serde_json::to_string(&root.join(&directory).join("artifacts"))?;
    fs::write(
        root.join(&directory).join("request.json"),
        format!(
            r#"{{"protocol_version":"11","endpoint":{{"protocol":"http","host":"127.0.0.1","port":8000,"completions_path":"/v1/completions","chat_completions_path":"/v1/chat/completions","server_metrics":null}},"model":{{"locator":"/models/test","served_name":"test"}},"definition":{{"request_source":{{"kind":"random","input_tokens":8,"output_tokens":1,"prefix_sharing":null,"shared_system_content":null}},"prompt":{{"kind":"server_chat","request_representation":"structured_messages","route":"chat_completions","rendering_authority":"server"}},"server_metrics":false,"seed":7,"request_body":{{}},"request_slo":null,"timeout_seconds":120,"cache_start":"uncontrolled"}},"case":{{"load_shape":{load_shape},"request_count":4,"warmup_request_count":0}},"case_budget_seconds":120.0,"artifact_dir":{artifacts}}}"#
        ),
    )?;
    Ok(format!("{directory}/request.json"))
}

/// A Bench record whose cases carry typed loads: `(case, load shape, metrics)`.
fn load_bench(
    root: &Path,
    id: &str,
    cases: &[(&str, &str, serde_json::Value)],
) -> Result<(), Box<dyn Error>> {
    let mut recorded = Vec::new();
    for (case, load_shape, metrics) in cases {
        let request = case_request(root, id, case, load_shape)?;
        recorded.push(serde_json::json!({
            "id": case, "status": "succeeded", "request": request, "metrics": metrics,
        }));
    }
    record(
        root,
        id,
        &serde_json::json!({
            "id": id, "kind": "bench", "status": "succeeded",
            "started_unix_ms": 1_767_348_000_000_u64, "cases": recorded,
        }),
    )
}

const ALPHA_SWEEP: &str = "2026-01-02T11-00-00.000Z-bench-alpha-sweep";
const BETA_SWEEP: &str = "2026-01-02T12-00-00.000Z-bench-beta-sweep";
const BETA_RATE: &str = "2026-01-02T13-00-00.000Z-bench-beta-rate";

fn concurrency(value: u32) -> String {
    format!(r#"{{"kind":"concurrency_limited","concurrency":{value}}}"#)
}

/// Two workspaces whose Bench records sweep concurrency and request rate;
/// case identifiers deliberately disagree with the loads they carry.
fn comparison_fixture(path: &Path) -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
    let alpha = fixture(&path.join("alpha"))?;
    let (c8, c1) = (concurrency(8), concurrency(1));
    load_bench(
        &alpha,
        ALPHA_SWEEP,
        &[
            ("first", &c8, serde_json::json!({"request_throughput": 7.5})),
            (
                "second",
                &c1,
                serde_json::json!({"request_throughput": 1.25}),
            ),
        ],
    )?;
    let beta = fixture(&path.join("beta"))?;
    let c4 = concurrency(4);
    load_bench(
        &beta,
        BETA_SWEEP,
        &[
            ("x", &c1, serde_json::json!({"request_throughput": 2.0})),
            ("y", &c4, serde_json::json!({"p99_ttft_ms": 50.0})),
            ("z", &c8, serde_json::json!({"request_throughput": 9.0})),
        ],
    )?;
    load_bench(
        &beta,
        BETA_RATE,
        &[(
            "r",
            r#"{"kind":"request_rate_limited","request_rate":3.5,"burstiness":null}"#,
            serde_json::json!({"request_throughput": 3.0}),
        )],
    )?;
    Ok((alpha, beta))
}

/// The table row whose load cell reads `load`.
fn table_row<'a>(page: &'a str, load: &str) -> Option<&'a str> {
    let cell = page.find(&format!(">{load}</th>"))?;
    let start = page[..cell].rfind("<tr")?;
    let end = page[cell..].find("</tr>")? + cell;
    Some(&page[start..end])
}

#[tokio::test]
async fn views_present_the_tui_rows_filter_and_record_tree() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let workspace = fixture(&root.path().join("ws"))?;
    let console = Console::start(&workspace, state.path()).await?;
    let base = console.workspace_path().await?;

    let overview = console
        .settled(&format!("{base}/overview"), "sweep-11")
        .await?;
    assert!(overview.contains("ATTENTION"), "{overview}");
    assert!(overview.contains("REC"), "authority badges stay on rows");
    assert!(overview.contains("failed"));

    let records = console.page(&format!("{base}/records")).await?;
    assert!(records.contains("latency-7") && records.contains("sweep-11"));
    assert!(records.contains("Jan 2"), "records group by day");
    assert!(
        !records.contains(">bench-000-random</a>"),
        "children start collapsed"
    );

    let expanded = console
        .page(&format!("{base}/records?open={PARENT}"))
        .await?;
    assert!(
        expanded.contains(">bench-000-random</a>") && expanded.contains("serve-qwen-11"),
        "an opened parent lists its children: {expanded}"
    );

    let issues = console
        .page(&format!("{base}/records?status=attention"))
        .await?;
    assert!(
        issues.contains("sweep-11") && !issues.contains("latency-7"),
        "{issues}"
    );

    let selected = console
        .page(&format!("{base}/records?open={PARENT}&selected={BENCH}"))
        .await?;
    assert!(selected.contains("METRICS"), "{selected}");
    assert!(
        selected.contains("1.25") && selected.contains("7.5"),
        "the metrics table holds every case"
    );
    assert!(selected.contains("c1") && selected.contains("c8"));

    let refreshed = console
        .client
        .post(format!("{}{base}/refresh", console.origin))
        .header(reqwest::header::COOKIE, &console.cookie)
        .header(reqwest::header::ORIGIN, &console.origin)
        .form(&[("back", format!("{base}/records"))])
        .send()
        .await?;
    assert_eq!(refreshed.status(), 303);
    Ok(())
}

#[tokio::test]
async fn record_logs_page_a_bounded_tail_and_pages_stream_updates() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let workspace = fixture(&root.path().join("ws"))?;
    let console = Console::start(&workspace, state.path()).await?;
    let base = console.workspace_path().await?;
    console
        .settled(&format!("{base}/records"), "sweep-11")
        .await?;

    let reference = format!(".inferlab/records/{SERVER}/stdout.log");
    let tail = console.page(&format!("{base}/log?ref={reference}")).await?;
    assert!(
        tail.contains("server log line 4000"),
        "the tail ends at the newest line"
    );
    assert!(
        !tail.contains("server log line 0001"),
        "the tail is bounded"
    );
    assert!(tail.contains("Earlier"), "earlier lines are a page away");
    let earlier = console
        .page(&format!("{base}/log?ref={reference}&page=1"))
        .await?;
    assert!(earlier.contains("server log line 0001"), "{earlier}");

    let mut events = console
        .client
        .get(format!("{}{base}/events", console.origin))
        .header(reqwest::header::COOKIE, &console.cookie)
        .send()
        .await?;
    assert_eq!(
        events
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), events.chunk())
        .await??
        .ok_or("the stream ended")?;
    assert!(
        String::from_utf8_lossy(&chunk).contains("event: generation"),
        "a published generation reaches the page"
    );
    Ok(())
}

#[tokio::test]
async fn comparison_lists_bench_records_across_workspaces() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let (alpha, beta) = comparison_fixture(root.path())?;
    let console = Console::start(&alpha, state.path()).await?;
    console.register(&beta).await?;
    let (alpha_id, beta_id) = (
        console.workspace_id("alpha").await?,
        console.workspace_id("beta").await?,
    );

    let picker = console.settled("/compare", BETA_RATE).await?;
    for (workspace, record) in [
        (&alpha_id, ALPHA_SWEEP),
        (&beta_id, BETA_SWEEP),
        (&beta_id, BETA_RATE),
        (&alpha_id, BENCH),
    ] {
        assert!(
            picker.contains(&format!("value=\"{workspace}:{record}\"")),
            "{record} is selectable: {picker}"
        );
    }
    assert!(
        !picker.contains(&format!(":{PARENT}\"")) && !picker.contains(&format!(":{SERVER}\"")),
        "only Bench records are compared"
    );

    let detail = console
        .settled(
            &format!("/w/{alpha_id}/records?selected={ALPHA_SWEEP}"),
            "METRICS",
        )
        .await?;
    assert!(
        detail.contains(&format!("href=\"/compare?r={alpha_id}:{ALPHA_SWEEP}\"")),
        "a Bench record's detail starts a comparison: {detail}"
    );
    Ok(())
}

#[tokio::test]
async fn comparison_places_series_by_recorded_load() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let (alpha, beta) = comparison_fixture(root.path())?;
    let console = Console::start(&alpha, state.path()).await?;
    console.register(&beta).await?;
    let (alpha_id, beta_id) = (
        console.workspace_id("alpha").await?,
        console.workspace_id("beta").await?,
    );
    let selection = format!("r={alpha_id}:{ALPHA_SWEEP}&r={beta_id}:{BETA_SWEEP}");
    console.settled("/compare", BETA_RATE).await?;

    let page = console
        .page(&format!("/compare?{selection}&metric=request_throughput"))
        .await?;
    assert_eq!(page.matches("<svg").count(), 1, "one load kind, one chart");
    assert!(page.contains("alpha · alpha-sweep") && page.contains("beta · beta-sweep"));
    let row = |load| table_row(&page, load).ok_or(format!("no {load} row: {page}"));
    let (c1, c4, c8) = (row("c1")?, row("c4")?, row("c8")?);
    assert!(c1.contains(">1.25<") && c1.contains(">2<"), "{c1}");
    assert!(c8.contains(">7.5<") && c8.contains(">9<"), "{c8}");
    assert_eq!(
        c4.matches(">—<").count(),
        1,
        "beta's case without the metric reads as missing: {c4}"
    );
    assert!(c4.contains("<td></td>"), "alpha has no case at c4");
    assert!(!c4.contains(">0<"), "missing is never zero");
    assert!(
        page.find(">c1</th>") < page.find(">c8</th>"),
        "rows follow the load, not the case order"
    );

    let mixed = console
        .page(&format!(
            "/compare?r={alpha_id}:{ALPHA_SWEEP}&r={beta_id}:{BETA_RATE}&metric=request_throughput"
        ))
        .await?;
    assert_eq!(
        mixed.matches("<svg").count(),
        2,
        "concurrency and request rate never share an axis"
    );
    assert!(mixed.contains("CONCURRENCY") && mixed.contains("REQUEST RATE"));

    let unplaced = console
        .page(&format!(
            "/compare?r={alpha_id}:{BENCH}&metric=request_throughput"
        ))
        .await?;
    assert!(
        !unplaced.contains("<svg"),
        "loads without a value are not charted"
    );
    assert!(
        unplaced.contains(">1.25<") && unplaced.contains(">7.5<"),
        "they stay in the table: {unplaced}"
    );
    Ok(())
}

#[tokio::test]
async fn an_operation_log_opens_from_its_reference() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let workspace = fixture(&root.path().join("ws"))?;
    // A live operation whose phase log is referenced by absolute path, as
    // workload commands publish it.
    let log = workspace.join(".inferlab/runtime/phase.log");
    fs::create_dir_all(workspace.join(".inferlab/runtime/observations"))?;
    fs::write(&log, "phase log line\n")?;
    let observation = serde_json::json!({
        "schema_version": 1, "inferlab_version": "0.0.0",
        "producer": identity(std::process::id())?,
        "command": "bench", "started_unix_ms": 1, "updated_unix_ms": 1,
        "progress": {"phase": "measuring", "log_ref": log.display().to_string()},
    });
    fs::write(
        workspace.join(".inferlab/runtime/observations/operation.json"),
        serde_json::to_vec(&observation)?,
    )?;
    let console = Console::start(&workspace, state.path()).await?;
    let base = console.workspace_path().await?;
    console
        .settled(&format!("{base}/operations"), "measuring")
        .await?;

    let page = console
        .page(&format!(
            "{base}/log?ref={}",
            log.display().to_string().replace('/', "%2F")
        ))
        .await?;
    assert!(page.contains("phase log line"), "{page}");
    Ok(())
}

#[tokio::test]
async fn workspaces_with_one_directory_name_stay_distinguishable() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let first = fixture(&root.path().join("first/lab"))?;
    let second = fixture(&root.path().join("second/lab"))?;
    let console = Console::start(&first, state.path()).await?;
    console.register(&second).await?;

    let picker = console.settled("/compare", "second/lab").await?;
    assert!(picker.contains("first/lab"), "{picker}");
    Ok(())
}

#[tokio::test]
async fn a_workspace_whose_definitions_fail_shows_the_failure() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let workspace = fixture(&root.path().join("ws"))?;
    fs::write(
        workspace.join(".inferlab/workspace.toml"),
        "schema_version = [\n",
    )?;
    let console = Console::start(&workspace, state.path()).await?;

    let index = console.settled("/", "definitions do not load").await?;
    assert!(
        index.contains("definitions unavailable") && index.contains("href=\"/w/"),
        "the failure is shown and the workspace still opens: {index}"
    );
    Ok(())
}
