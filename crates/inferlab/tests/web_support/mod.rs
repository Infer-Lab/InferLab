//! A client of a real `inferlab web` process for the web console tests:
//! token entry, cookie, same-origin forms, and workspace lookup.
#![allow(dead_code)]

use std::error::Error;
use std::ffi::OsString;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};

pub(crate) struct Console {
    pub(crate) child: Child,
    pub(crate) origin: String,
    pub(crate) cookie: String,
    pub(crate) client: reqwest::Client,
}

impl Console {
    pub(crate) async fn start(cwd: &Path, state: &Path) -> Result<Self, Box<dyn Error>> {
        Self::start_with(cwd, state, &[]).await
    }

    /// Start with extra environment, such as a fixture `PATH`.
    pub(crate) async fn start_with(
        cwd: &Path,
        state: &Path,
        env: &[(&str, OsString)],
    ) -> Result<Self, Box<dyn Error>> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .args(["web", "--refresh-interval", "200ms"])
            .current_dir(cwd)
            .env("XDG_STATE_HOME", state)
            .envs(env.iter().map(|(key, value)| (key, value)))
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut line = String::new();
        BufReader::new(child.stdout.take().ok_or("stdout")?).read_line(&mut line)?;
        let url = line.trim().to_owned();
        let origin = url
            .strip_prefix("http://")
            .and_then(|rest| {
                rest.find('/')
                    .map(|end| url[.."http://".len() + end].to_owned())
            })
            .ok_or("url")?;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        let entry = client.get(&url).send().await?;
        let cookie = entry
            .headers()
            .get(reqwest::header::SET_COOKIE)
            .ok_or("cookie")?
            .to_str()?
            .split(';')
            .next()
            .unwrap_or_default()
            .to_owned();
        Ok(Self {
            child,
            origin,
            cookie,
            client,
        })
    }

    pub(crate) async fn page(&self, path: &str) -> Result<String, Box<dyn Error>> {
        let response = self
            .client
            .get(format!("{}{path}", self.origin))
            .header(reqwest::header::COOKIE, &self.cookie)
            .send()
            .await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(format!("GET {path} -> {status}: {text}").into());
        }
        Ok(text)
    }

    /// Submit a same-origin form; the response is returned as is.
    pub(crate) async fn post(
        &self,
        path: &str,
        form: &[(&str, &str)],
    ) -> Result<reqwest::Response, Box<dyn Error>> {
        Ok(self
            .client
            .post(format!("{}{path}", self.origin))
            .header(reqwest::header::COOKIE, &self.cookie)
            .header(reqwest::header::ORIGIN, &self.origin)
            .form(form)
            .send()
            .await?)
    }

    /// Register another workspace the way the browser's Add button does.
    pub(crate) async fn register(&self, root: &Path) -> Result<(), Box<dyn Error>> {
        let response = self
            .client
            .post(format!("{}/workspaces", self.origin))
            .header(reqwest::header::COOKIE, &self.cookie)
            .header(reqwest::header::ORIGIN, &self.origin)
            .form(&[("path", root.display().to_string())])
            .send()
            .await?;
        if response.status() != 303 {
            return Err(format!("register -> {}", response.status()).into());
        }
        Ok(())
    }

    /// The console id of a registered workspace, read from the workspace
    /// list where its name links to its views.
    pub(crate) async fn workspace_id(&self, name: &str) -> Result<String, Box<dyn Error>> {
        let list = self.page("/").await?;
        let anchor = list
            .find(&format!("\">{name}</a>"))
            .ok_or("workspace not listed")?;
        let start = list[..anchor].rfind("/w/").ok_or("no workspace link")? + "/w/".len();
        Ok(list[start..anchor].to_owned())
    }

    /// The workspace's console path, taken from the workspace list.
    pub(crate) async fn workspace_path(&self) -> Result<String, Box<dyn Error>> {
        let list = self.page("/").await?;
        let start = list.find("href=\"/w/").ok_or("no workspace link")? + "href=\"".len();
        let end = list[start..].find('"').ok_or("unterminated link")? + start;
        Ok(list[start..end].to_owned())
    }

    /// A page once the workspace's first generation has been published.
    pub(crate) async fn settled(&self, path: &str, needle: &str) -> Result<String, Box<dyn Error>> {
        let deadline = std::time::Instant::now() + crate::support::FIXTURE_HANG_GUARD;
        while std::time::Instant::now() < deadline {
            let page = self.page(path).await?;
            if page.contains(needle) {
                return Ok(page);
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        Err(format!("{path} never showed {needle:?}").into())
    }
}

impl Drop for Console {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Field `index` (1-based, as in proc(5)) of a process's stat line.
pub(crate) fn stat_field(pid: &str, index: usize) -> Result<String, Box<dyn Error>> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let tail = stat.rsplit_once(')').ok_or("stat")?.1;
    Ok(tail
        .split_whitespace()
        .nth(index - 3)
        .ok_or("field")?
        .to_owned())
}

/// The producer identity InferLab records for a live process.
pub(crate) fn identity(pid: u32) -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(serde_json::json!({
        "host": std::fs::read_to_string("/proc/sys/kernel/hostname")?.trim(),
        "boot_id": std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?.trim(),
        "pid": pid,
        "process_start_ticks": stat_field(&pid.to_string(), 22)?.parse::<u64>()?,
    }))
}
