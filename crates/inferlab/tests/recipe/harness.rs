use crate::support;

use serde_json::Value;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

pub(crate) const WORKSPACE: &str = include_str!("../fixtures/dsv4-workspace.toml");

pub(crate) fn resolved_ranks(
    server: &Value,
) -> Result<Vec<support::ResolvedProcessProjection>, Box<dyn Error>> {
    support::resolved_processes(server)
}

pub(crate) fn process_evidence<'a>(
    record: &'a Value,
    id: &str,
) -> Result<&'a Value, Box<dyn Error>> {
    record["process_evidence"]
        .get(id)
        .ok_or_else(|| format!("missing process evidence {id:?}").into())
}

pub(crate) struct TestWorkspace {
    // Declared before `root` so fixture process groups are reaped before the
    // workspace directory they run in is removed.
    reaper: support::ServeReaper,
    root: TempDir,
    bin: PathBuf,
    data_home: PathBuf,
    bench_marker: PathBuf,
    eval_marker: PathBuf,
    capture_events: PathBuf,
}

impl TestWorkspace {
    pub(crate) fn new() -> Result<Self, Box<dyn Error>> {
        let root = tempfile::tempdir()?;
        let reaper = support::ServeReaper::for_workspace(root.path());
        let ports = support::reserve_local_ports(1)?;
        let port = ports.get(0);
        let inferlab = root.path().join(".inferlab");
        let bin = root.path().join("bin");
        fs::create_dir_all(&inferlab)?;
        fs::create_dir_all(&bin)?;
        fs::create_dir_all(root.path().join("vendor/vllm"))?;
        fs::create_dir_all(root.path().join("vendor/flashinfer"))?;
        fs::write(inferlab.join("workspace.toml"), WORKSPACE)?;
        fs::write(root.path().join("vendor/vllm/source.txt"), "baseline\n")?;
        fs::write(
            root.path().join("vendor/flashinfer/source.txt"),
            "baseline\n",
        )?;
        fs::write(
            root.path().join("pixi.toml"),
            "[workspace]\n\
             channels = [\"conda-forge\"]\n\
             platforms = [\"linux-64\"]\n\
             \n\
             [environments]\n\
             vllm = []\n\
             \n\
             [pypi-dependencies]\n\
             inferlab-integration-vllm = \"==0.1.0\"\n",
        )?;
        fs::write(
            root.path().join("pixi.lock"),
            "version: 6\nenvironments:\n  vllm: {}\n",
        )?;
        // ensure_usable checks this prefix exists on disk before shelling
        // out to pixi at all.
        fs::create_dir_all(root.path().join(".pixi/envs/vllm"))?;
        fs::write(root.path().join(".gitignore"), ".inferlab/local.toml\n")?;
        fs::write(
            inferlab.join("local.toml"),
            format!(
                "default_placement = \"local\"\n\
                 \n\
                 [model_weights.dsv4]\n\
                 locator = \"/models/dsv4\"\n\
                 \n\
                 [machines.local]\n\
                 host = \"127.0.0.1\"\n\
                 ports = [{port}]\n\
                 devices = [0, 1, 2, 3]\n\
                 \n\
                 [placements.local]\n\
                 machines = [\"local\"]\n"
            ),
        )?;
        ports.release();
        write_executable(&bin.join("pixi"), PIXI)?;
        write_executable(&bin.join("inferlab-adapter-vllm"), ADAPTER)?;
        write_executable(&bin.join("fixture-server"), FIXTURE_SERVER)?;
        write_executable(&bin.join("nsys"), NSYS)?;
        write_executable(&bin.join("fixture-eval-client"), EVAL_CLIENT)?;
        write_executable(&bin.join("fixture-bench-client"), BENCH_CLIENT)?;
        write_executable(&bin.join("nvidia-smi"), NVIDIA_SMI)?;
        let data_home = root.path().join("data");
        let mut path = OsString::from(&bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let install = Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .current_dir(root.path())
            .env("PATH", path)
            .env("XDG_DATA_HOME", &data_home)
            .args(["toolchain", "install"])
            .output()?;
        if !install.status.success() {
            return Err(format!(
                "toolchain fixture install failed: {}",
                String::from_utf8_lossy(&install.stderr)
            )
            .into());
        }
        git(root.path(), &["init", "-q"])?;
        git(root.path(), &["config", "user.email", "test@example.com"])?;
        git(root.path(), &["config", "user.name", "Inferlab Test"])?;
        git(root.path(), &["add", "."])?;
        git(root.path(), &["commit", "-qm", "fixture"])?;
        let bench_marker = root.path().join("bench-ran");
        let eval_marker = root.path().join("eval-ran");
        let capture_events = root.path().join("capture-events");
        Ok(Self {
            reaper,
            root,
            bin,
            data_home,
            bench_marker,
            eval_marker,
            capture_events,
        })
    }

    pub(crate) fn command(&self) -> Command {
        let mut path = OsString::from(&self.bin);
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut command = Command::new(env!("CARGO_BIN_EXE_inferlab"));
        command
            .current_dir(self.root.path().join("vendor/vllm"))
            .env("PATH", path)
            .env("XDG_DATA_HOME", &self.data_home)
            .env("FIXTURE_BENCH_MARKER", &self.bench_marker)
            .env("FIXTURE_EVAL_MARKER", &self.eval_marker)
            .env("FIXTURE_CAPTURE_EVENTS", &self.capture_events)
            .env(
                "FIXTURE_NSYS_STATE",
                self.root.path().join(".inferlab/nsys-state"),
            );
        for (key, value) in self.reaper.env() {
            command.env(key, value);
        }
        command
    }

    pub(crate) fn run(&self) -> Result<Output, Box<dyn Error>> {
        Ok(self
            .command()
            .args(["recipe", "run", "dsv4-qualify"])
            .output()?)
    }

    pub(crate) fn load_record(&self, id: &str) -> Result<Value, Box<dyn Error>> {
        Ok(serde_json::from_slice(&fs::read(
            self.root
                .path()
                .join(".inferlab/records")
                .join(id)
                .join("record.json"),
        )?)?)
    }

    pub(crate) fn configure_pd(&self, transport: &str) -> Result<(), Box<dyn Error>> {
        let config = WORKSPACE
            .replacen(
                "topology = \"single\"",
                &format!(
                    "topology = \"prefill_decode\"\ngateway_backend = \"builtin\"\npd_router_backend = \"builtin\"\nkv_transfer = {transport:?}"
                ),
                1,
            )
            .replacen(
                "[servers.dsv4-qualify.roles.serve.parallelism.attention]\n",
                "[servers.dsv4-qualify.roles.prefill]\nreplicas = 2\n\n[servers.dsv4-qualify.roles.prefill.parallelism.attention]\n",
                1,
            )
            .replacen(
                "[servers.dsv4-qualify.roles.serve.settings]\n",
                "[servers.dsv4-qualify.roles.prefill.settings]\n",
                1,
            )
            .replace(
                "[servers.dsv4-qualify.cases.tp2.parallelism.outer]",
                "[servers.dsv4-qualify.roles.decode]\nreplicas = 2\n\n[servers.dsv4-qualify.cases.tp2.parallelism.outer]",
            )
            .replace("reset_prefix_cache = true", "reset_prefix_cache = false");
        fs::write(self.root.path().join(".inferlab/workspace.toml"), config)?;
        let ports = support::reserve_local_ports(9)?;
        fs::write(
            self.root.path().join(".inferlab/local.toml"),
            format!(
                "default_placement = \"local\"\n\n[model_weights.dsv4]\nlocator = \"/models/dsv4\"\n\n[machines.local]\nhost = \"127.0.0.1\"\nports = [{}, {}, {}, {}, {}, {}, {}, {}, {}]\ndevices = [0, 1, 2, 3, 4, 5, 6, 7]\n\n[placements.local]\nmachines = [\"local\"]\n",
                ports.get(0),
                ports.get(1),
                ports.get(2),
                ports.get(3),
                ports.get(4),
                ports.get(5),
                ports.get(6),
                ports.get(7),
                ports.get(8)
            ),
        )?;
        ports.release();
        Ok(())
    }

    pub(crate) fn append_manifest(&self, block: &str) -> Result<(), Box<dyn Error>> {
        let manifest = self.root.path().join(".inferlab/workspace.toml");
        let mut text = fs::read_to_string(&manifest)?;
        text.push_str(block);
        fs::write(manifest, text)?;
        Ok(())
    }

    pub(crate) fn root(&self) -> &Path {
        self.root.path()
    }

    pub(crate) fn bin(&self) -> &Path {
        &self.bin
    }

    pub(crate) fn bench_marker(&self) -> &Path {
        &self.bench_marker
    }

    pub(crate) fn eval_marker(&self) -> &Path {
        &self.eval_marker
    }

    pub(crate) fn capture_events(&self) -> &Path {
        &self.capture_events
    }
}

pub(crate) fn write_executable(path: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, content)?;
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

pub(crate) fn wait_for_path(path: &Path, timeout: Duration) -> Result<(), Box<dyn Error>> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if path.exists() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!("{} was not created within {timeout:?}", path.display()).into())
}

fn git(root: &Path, args: &[&str]) -> Result<(), Box<dyn Error>> {
    let output = Command::new("git").current_dir(root).args(args).output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into())
    }
}

const PIXI: &str = include_str!("../fixtures/bin/pixi.sh");

const ADAPTER: &str = include_str!("../fixtures/bin/recipe-adapter.py");

const FIXTURE_SERVER: &str = include_str!("../fixtures/bin/recipe-server.py");

const NSYS: &str = include_str!("../fixtures/bin/recipe-nsys.sh");

const EVAL_CLIENT: &str = include_str!("../fixtures/bin/recipe-eval-client.py");

const BENCH_CLIENT: &str = include_str!("../fixtures/bin/bench-client.py");

/// Fixture GPU inventory in nvidia-smi's `csv,noheader,nounits` row shape.
const NVIDIA_SMI: &str = include_str!("../fixtures/bin/recipe-nvidia-smi.sh");
