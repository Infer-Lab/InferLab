//! The web console's access, registry, and start-up contract
//! ([[RFC-0012:C-SCOPE]], [[RFC-0012:C-ACCESS]], [[RFC-0012:C-WORKSPACES]]),
//! exercised against the real binary over HTTP.

use std::error::Error;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// A workspace root: a directory that holds `.inferlab/workspace.toml`.
fn workspace(path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    fs::create_dir_all(path.join(".inferlab"))?;
    fs::write(
        path.join(".inferlab/workspace.toml"),
        "schema_version = 2\n",
    )?;
    Ok(path.canonicalize()?)
}

struct Console {
    child: Child,
    url: String,
    origin: String,
}

impl Console {
    fn start(cwd: &Path, state: &Path, extra: &[&str]) -> Result<Self, Box<dyn Error>> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_inferlab"))
            .arg("web")
            .args(extra)
            .current_dir(cwd)
            .env("XDG_STATE_HOME", state)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().ok_or("stdout")?;
        let mut line = String::new();
        BufReader::new(stdout).read_line(&mut line)?;
        let url = line.trim().to_owned();
        // The origin is everything before the path: `http://host:port`.
        let origin = url
            .strip_prefix("http://")
            .and_then(|rest| {
                rest.find('/')
                    .map(|end| url[.."http://".len() + end].to_owned())
            })
            .ok_or_else(|| format!("unexpected access URL {url:?}"))?;
        Ok(Self { child, url, origin })
    }

    fn client() -> Result<reqwest::Client, Box<dyn Error>> {
        Ok(reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()?)
    }

    /// The cookie the access URL sets.
    async fn cookie(&self) -> Result<String, Box<dyn Error>> {
        let response = Self::client()?.get(&self.url).send().await?;
        let cookie = response
            .headers()
            .get(reqwest::header::SET_COOKIE)
            .ok_or("no cookie")?
            .to_str()?
            .to_owned();
        Ok(cookie.split(';').next().unwrap_or_default().to_owned())
    }

    async fn get(&self, path: &str, cookie: &str) -> Result<reqwest::Response, Box<dyn Error>> {
        Ok(Self::client()?
            .get(format!("{}{path}", self.origin))
            .header(reqwest::header::COOKIE, cookie)
            .send()
            .await?)
    }

    async fn post(
        &self,
        path: &str,
        cookie: &str,
        origin: &str,
        form: &[(&str, &str)],
    ) -> Result<reqwest::Response, Box<dyn Error>> {
        Ok(Self::client()?
            .post(format!("{}{path}", self.origin))
            .header(reqwest::header::COOKIE, cookie)
            .header(reqwest::header::ORIGIN, origin)
            .form(form)
            .send()
            .await?)
    }
}

impl Drop for Console {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn registry(state: &Path) -> Result<serde_json::Value, Box<dyn Error>> {
    Ok(serde_json::from_slice(&fs::read(
        state.join("inferlab/web/workspaces.json"),
    )?)?)
}

#[tokio::test]
async fn access_needs_the_per_start_token_and_a_matching_origin() -> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let home = workspace(&root.path().join("ws"))?;
    let console = Console::start(&home, state.path(), &[])?;
    assert!(
        console.url.starts_with("http://127.0.0.1:"),
        "{}",
        console.url
    );

    let anonymous = Console::client()?.get(&console.origin).send().await?;
    assert_eq!(anonymous.status(), 401);
    assert!(
        !anonymous
            .text()
            .await?
            .contains(home.to_string_lossy().as_ref())
    );

    let entry = Console::client()?.get(&console.url).send().await?;
    assert_eq!(entry.status(), 303);
    let printed_path = console
        .url
        .strip_prefix(&console.origin)
        .and_then(|path| path.split_once('?'))
        .map(|(path, _)| path)
        .ok_or("url")?;
    assert_eq!(
        entry
            .headers()
            .get(reqwest::header::LOCATION)
            .map(|value| value.as_bytes()),
        Some(printed_path.as_bytes()),
        "the token leaves the address bar"
    );
    let set_cookie = entry
        .headers()
        .get(reqwest::header::SET_COOKIE)
        .ok_or("no cookie")?
        .to_str()?;
    assert!(set_cookie.starts_with("inferlab_web_"), "{set_cookie}");
    assert!(set_cookie.contains("HttpOnly") && set_cookie.contains("SameSite=Strict"));

    let cookie = console.cookie().await?;
    let page = console.get("/", &cookie).await?;
    assert_eq!(page.status(), 200);
    assert_eq!(
        page.headers()
            .get("referrer-policy")
            .map(|value| value.as_bytes()),
        Some(&b"no-referrer"[..])
    );
    assert!(page.text().await?.contains(home.to_string_lossy().as_ref()));

    let forged = console
        .post(
            "/workspaces/remove",
            &cookie,
            "http://evil.example",
            &[("path", home.to_string_lossy().as_ref())],
        )
        .await?;
    assert_eq!(
        forged.status(),
        403,
        "a change needs an Origin matching its Host"
    );
    assert_eq!(
        registry(state.path())?["workspaces"][0],
        home.to_string_lossy().as_ref()
    );
    Ok(())
}

#[tokio::test]
async fn workspaces_register_from_the_directory_browser_and_persist() -> Result<(), Box<dyn Error>>
{
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let first = workspace(&root.path().join("first"))?;
    let second = workspace(&root.path().join("second"))?;
    fs::create_dir_all(root.path().join("plain"))?;
    fs::create_dir_all(root.path().join(".hidden"))?;
    fs::write(root.path().join("notes.txt"), "not a directory")?;
    std::os::unix::fs::symlink(&second, root.path().join("alias"))?;
    let console = Console::start(&first, state.path(), &[])?;
    let cookie = console.cookie().await?;
    let parent = root.path().canonicalize()?;

    let listing = console
        .get(&format!("/browse?path={}", parent.display()), &cookie)
        .await?
        .text()
        .await?;
    assert!(
        listing.contains("second") && listing.contains("plain"),
        "{listing}"
    );
    assert!(
        !listing.contains("notes.txt"),
        "only directories are listed"
    );
    assert!(listing.contains("data-workspace=\"second\""), "{listing}");
    assert!(!listing.contains("data-workspace=\"plain\""), "{listing}");
    assert!(
        !listing.contains(".hidden"),
        "hidden directories wait for the toggle"
    );
    assert!(
        listing.find("second").unwrap_or(usize::MAX) < listing.find("plain").unwrap_or(0),
        "workspaces lead the listing"
    );
    let hidden = console
        .get(
            &format!("/browse?path={}&hidden=1", parent.display()),
            &cookie,
        )
        .await?
        .text()
        .await?;
    assert!(hidden.contains(".hidden"), "{hidden}");

    let rejected = console
        .post(
            "/workspaces",
            &cookie,
            &console.origin,
            &[("path", parent.join("plain").to_string_lossy().as_ref())],
        )
        .await?;
    assert_eq!(rejected.status(), 400, "only a workspace root registers");

    for path in [second.clone(), parent.join("alias"), second.clone()] {
        let added = console
            .post(
                "/workspaces",
                &cookie,
                &console.origin,
                &[("path", path.to_string_lossy().as_ref())],
            )
            .await?;
        assert_eq!(added.status(), 303);
    }
    assert_eq!(
        registry(state.path())?["workspaces"],
        serde_json::json!([first.to_string_lossy(), second.to_string_lossy()]),
        "registration stores the canonical root once"
    );

    let removed = console
        .post(
            "/workspaces/remove",
            &cookie,
            &console.origin,
            &[("path", first.to_string_lossy().as_ref())],
        )
        .await?;
    assert_eq!(removed.status(), 303);
    assert_eq!(
        registry(state.path())?["workspaces"],
        serde_json::json!([second.to_string_lossy()])
    );
    assert!(
        first.join(".inferlab/workspace.toml").exists(),
        "removal only unregisters"
    );
    Ok(())
}

#[tokio::test]
async fn an_unloadable_registered_root_stays_registered_as_unavailable()
-> Result<(), Box<dyn Error>> {
    let root = tempfile::tempdir()?;
    let state = tempfile::tempdir()?;
    let gone = root.path().join("gone");
    fs::create_dir_all(state.path().join("inferlab/web"))?;
    fs::write(
        state.path().join("inferlab/web/workspaces.json"),
        serde_json::to_vec(&serde_json::json!({ "workspaces": [gone] }))?,
    )?;
    let console = Console::start(root.path(), state.path(), &[])?;
    let cookie = console.cookie().await?;

    let page = console.get("/", &cookie).await?.text().await?;
    assert!(page.contains(gone.to_string_lossy().as_ref()), "{page}");
    assert!(page.contains("unavailable"), "{page}");
    assert_eq!(
        registry(state.path())?["workspaces"][0],
        gone.to_string_lossy().as_ref()
    );
    Ok(())
}

fn failed_start(state: &Path, extra: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_inferlab"))
        .arg("web")
        .args(extra)
        .current_dir(state)
        .env("XDG_STATE_HOME", state)
        .output()?;
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "no URL is printed before a failure"
    );
    Ok(String::from_utf8(output.stderr)?)
}

#[test]
fn a_malformed_registry_or_a_taken_port_fails_the_start_with_e9002() -> Result<(), Box<dyn Error>> {
    let state = tempfile::tempdir()?;
    let file = state.path().join("inferlab/web/workspaces.json");
    fs::create_dir_all(file.parent().ok_or("parent")?)?;
    fs::write(&file, "{not json")?;
    let stderr = failed_start(state.path(), &[])?;
    assert!(
        stderr.contains("error[E9002]") && stderr.contains("workspaces.json"),
        "{stderr}"
    );
    assert_eq!(
        fs::read_to_string(&file)?,
        "{not json",
        "a malformed registry is not overwritten"
    );

    fs::remove_file(&file)?;
    let taken = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = taken.local_addr()?.port().to_string();
    let stderr = failed_start(state.path(), &["--port", &port])?;
    assert!(stderr.contains("error[E9002]"), "{stderr}");
    Ok(())
}

#[test]
fn a_non_loopback_bind_warns_that_traffic_is_not_encrypted() -> Result<(), Box<dyn Error>> {
    let state = tempfile::tempdir()?;
    let mut console = Console::start(state.path(), state.path(), &["--bind", "0.0.0.0"])?;
    let mut stderr = console.child.stderr.take().ok_or("stderr")?;
    let _ = console.child.kill();
    let mut text = String::new();
    stderr.read_to_string(&mut text)?;
    assert!(text.contains("not encrypted"), "{text}");
    Ok(())
}

#[tokio::test]
async fn an_ipv6_bind_prints_an_openable_url() -> Result<(), Box<dyn Error>> {
    let state = tempfile::tempdir()?;
    let console = Console::start(state.path(), state.path(), &["--bind", "::1"])?;
    assert!(console.url.starts_with("http://[::1]:"), "{}", console.url);
    let entry = Console::client()?.get(&console.url).send().await?;
    assert_eq!(entry.status(), 303);
    Ok(())
}

#[tokio::test]
async fn the_url_opens_the_workspace_it_was_started_in() -> Result<(), Box<dyn Error>> {
    let state = tempfile::tempdir()?;
    let root = tempfile::tempdir()?;
    std::fs::create_dir_all(root.path().join(".inferlab"))?;
    std::fs::write(
        root.path().join(".inferlab/workspace.toml"),
        "schema_version = 2\n",
    )?;
    let console = Console::start(root.path(), state.path(), &[])?;
    let path = console
        .url
        .strip_prefix(&console.origin)
        .ok_or("url outside its origin")?;
    assert!(
        path.starts_with("/w/") && path.contains("/overview?token="),
        "{}",
        console.url
    );
    let entry = Console::client()?.get(&console.url).send().await?;
    let location = entry
        .headers()
        .get(reqwest::header::LOCATION)
        .ok_or("no redirect")?
        .to_str()?;
    assert!(
        location.starts_with("/w/") && location.ends_with("/overview"),
        "{location}"
    );

    let elsewhere = tempfile::tempdir()?;
    let unfocused = Console::start(elsewhere.path(), state.path(), &["--port", "0"])?;
    assert!(
        unfocused.url.ends_with(&format!(
            "/?token={}",
            unfocused
                .url
                .rsplit_once('=')
                .map_or("", |(_, token)| token)
        )),
        "outside a workspace the URL opens the workspace list: {}",
        unfocused.url
    );
    Ok(())
}
