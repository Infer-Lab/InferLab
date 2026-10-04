//! The typed actions the console launches ([[RFC-0012:C-ACTIONS]]). Each
//! form becomes an argument vector for the service's own executable with the
//! canonical workspace; option values use the `--name=value` form and the
//! positional value follows `--`, so no field is ever read as an option.

use serde::Deserialize;
use std::path::Path;

/// The submitted fields of every action form; unused fields stay empty.
#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct ActionForm {
    pub(super) action: String,
    #[serde(default)]
    pub(super) target: String,
    #[serde(default)]
    pub(super) case: String,
    #[serde(default)]
    pub(super) placement: String,
    #[serde(default)]
    pub(super) serve: String,
    #[serde(default)]
    pub(super) text: String,
    #[serde(default)]
    pub(super) topic: String,
    #[serde(default)]
    pub(super) records: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ActionKind {
    ServeStart,
    ServeStop,
    RecipeRun,
    Bench,
    Note,
}

impl ActionKind {
    pub(super) const ALL: [Self; 5] = [
        Self::ServeStart,
        Self::ServeStop,
        Self::RecipeRun,
        Self::Bench,
        Self::Note,
    ];

    /// The form value and job-name segment.
    pub(super) const fn key(self) -> &'static str {
        match self {
            Self::ServeStart => "serve-start",
            Self::ServeStop => "serve-stop",
            Self::RecipeRun => "recipe-run",
            Self::Bench => "bench",
            Self::Note => "scratchpad-note",
        }
    }

    pub(super) const fn title(self) -> &'static str {
        match self {
            Self::ServeStart => "Serve start",
            Self::ServeStop => "Serve stop",
            Self::RecipeRun => "Recipe run",
            Self::Bench => "Bench",
            Self::Note => "Scratchpad note",
        }
    }

    pub(super) fn parse(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.key() == key)
    }

    /// Whether the launch must follow a successful `--dry-run` preview.
    pub(super) const fn previewed(self) -> bool {
        matches!(self, Self::ServeStart | Self::RecipeRun | Self::Bench)
    }
}

/// A validated action: its kind and the vector it launches.
pub(super) struct Action {
    pub(super) kind: ActionKind,
    pub(super) form: ActionForm,
    launch: Vec<String>,
}

impl Action {
    pub(super) fn from_form(
        form: ActionForm,
        executable: &Path,
        root: &Path,
    ) -> Result<Self, String> {
        let kind = ActionKind::parse(&form.action)
            .ok_or_else(|| format!("unknown action {:?}", form.action))?;
        let required = |value: &str, name: &str| {
            let value = value.trim();
            if value.is_empty() {
                Err(format!("{} needs {name}", kind.title()))
            } else {
                Ok(value.to_owned())
            }
        };
        let optional = |name: &str, value: &str| {
            let value = value.trim();
            (!value.is_empty()).then(|| format!("--{name}={value}"))
        };
        let mut argv = vec![
            executable.display().to_string(),
            format!("--workspace={}", root.display()),
        ];
        let positional = match kind {
            ActionKind::ServeStart | ActionKind::RecipeRun => {
                let target = required(&form.target, "a definition")?;
                argv.extend(
                    if kind == ActionKind::ServeStart {
                        ["serve", "start"]
                    } else {
                        ["recipe", "run"]
                    }
                    .map(str::to_owned),
                );
                argv.extend(optional("case", &form.case));
                argv.extend(optional("placement", &form.placement));
                target
            }
            ActionKind::ServeStop => {
                let record = required(&form.target, "a server record")?;
                argv.extend(["serve", "stop"].map(str::to_owned));
                record
            }
            ActionKind::Bench => {
                let bench = required(&form.target, "a Bench definition")?;
                let serve = required(&form.serve, "a running server record")?;
                argv.push("bench".to_owned());
                argv.push(format!("--serve={serve}"));
                bench
            }
            ActionKind::Note => {
                // The text is the operator's own; only blank text is rejected.
                if form.text.trim().is_empty() {
                    return Err("Scratchpad note needs text".to_owned());
                }
                argv.extend(["scratchpad", "note"].map(str::to_owned));
                argv.extend(optional("topic", &form.topic));
                argv.extend(
                    form.records
                        .split(|character: char| character == ',' || character.is_whitespace())
                        .filter(|record| !record.is_empty())
                        .map(|record| format!("--record={record}")),
                );
                form.text.clone()
            }
        };
        argv.push("--".to_owned());
        argv.push(positional);
        Ok(Self {
            kind,
            form,
            launch: argv,
        })
    }

    /// The vector a launch runs.
    pub(super) fn launch_argv(&self) -> &[String] {
        &self.launch
    }

    /// The previewed vector: the launch vector with `--dry-run` before `--`.
    pub(super) fn preview_argv(&self) -> Vec<String> {
        let mut argv = self.launch.clone();
        let separator = argv.len().saturating_sub(2);
        argv.insert(separator, "--dry-run".to_owned());
        argv
    }
}

/// The captured outcome of a dry run.
pub(super) struct Preview {
    pub(super) success: bool,
    pub(super) status: String,
    pub(super) stdout: String,
    pub(super) stderr: String,
}

/// The most of each stream a preview keeps, from its end.
const PREVIEW_BYTES: usize = 256 * 1024;

/// Run the dry run in its own process group. An abandoned request drops
/// this future; the dry run then receives the interrupt a terminal Ctrl+C
/// delivers, so the CLI cleans up as it would in a terminal.
pub(super) async fn preview(argv: &[String], root: &Path) -> Preview {
    let failed = |status: String| Preview {
        success: false,
        status,
        stdout: String::new(),
        stderr: String::new(),
    };
    let Some((executable, arguments)) = argv.split_first() else {
        return failed("no executable".to_owned());
    };
    let child = tokio::process::Command::new(executable)
        .args(arguments)
        .current_dir(root)
        .process_group(0)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let child = match child {
        Ok(child) => child,
        Err(error) => return failed(format!("could not start the dry run: {error}")),
    };
    let mut abandoned = InterruptOnDrop(child.id());
    let output = child.wait_with_output().await;
    abandoned.0 = None;
    match output {
        Ok(output) => Preview {
            success: output.status.success(),
            status: output.status.to_string(),
            stdout: tail(&output.stdout),
            stderr: tail(&output.stderr),
        },
        Err(error) => failed(format!("could not wait for the dry run: {error}")),
    }
}

/// Interrupts a still-running dry run's process group when dropped.
struct InterruptOnDrop(Option<u32>);

impl Drop for InterruptOnDrop {
    fn drop(&mut self) {
        if let Some(group) = self
            .0
            .and_then(|pid| i32::try_from(pid).ok())
            .and_then(rustix::process::Pid::from_raw)
        {
            let _ = rustix::process::kill_process_group(group, rustix::process::Signal::INT);
        }
    }
}

fn tail(bytes: &[u8]) -> String {
    let start = bytes.len().saturating_sub(PREVIEW_BYTES);
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::{Action, ActionForm};
    use std::path::Path;

    #[test]
    fn values_reach_the_cli_as_values() -> Result<(), String> {
        let action = Action::from_form(
            ActionForm {
                action: "serve-start".to_owned(),
                target: "--help".to_owned(),
                case: "-x".to_owned(),
                ..ActionForm::default()
            },
            Path::new("/bin/inferlab"),
            Path::new("/ws"),
        )?;
        assert_eq!(
            action.launch_argv(),
            [
                "/bin/inferlab",
                "--workspace=/ws",
                "serve",
                "start",
                "--case=-x",
                "--",
                "--help"
            ]
        );
        assert_eq!(
            action.preview_argv()[4..],
            ["--case=-x", "--dry-run", "--", "--help"]
        );
        Ok(())
    }
}
