//! `refx connect`: a stdio MCP server that forwards to the daemon.
//!
//! A client that starts stdio servers (Claude Code's `.mcp.json` with a `command`) gets the
//! daemon's warm engine without racing a hook: `connect` starts the daemon if it is not
//! running, then relays each JSON-RPC line from stdin to `POST /mcp/<project>` and each message
//! of the reply (JSON or SSE) to stdout. Requests run concurrently. If the daemon restarts, the
//! session it knew is gone (404): `connect` starts it again if needed, replays the client's
//! `initialize` to open a new session and retries the request, so the client never notices.

use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use serde_json::Value;

struct Link {
    addr: String,
    url: String,
    agent: ureq::Agent,
    /// Session id and protocol version of the current daemon session.
    session: Mutex<Option<(String, String)>>,
    /// The client's `initialize` request, replayed to reopen a session.
    init: Mutex<Option<String>>,
    out: Mutex<std::io::Stdout>,
}

enum Reply {
    /// Messages to hand to the client.
    Messages(Vec<String>),
    /// The daemon does not know the session (it restarted).
    NoSession,
}

impl Link {
    fn emit(&self, msg: &str) {
        let mut o = self.out.lock().unwrap_or_else(|p| p.into_inner());
        let _ = writeln!(o, "{msg}");
        let _ = o.flush();
    }

    /// One POST; collects the reply's messages.
    fn post(&self, body: &str, capture_session: bool) -> Result<Reply> {
        let session = self.session.lock().unwrap().clone();
        let mut req = self.agent.post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");
        if let Some((id, version)) = &session {
            req = req.header("Mcp-Session-Id", id).header("MCP-Protocol-Version", version);
        }
        let mut resp = req.send(body)?;
        let status = resp.status().as_u16();
        if status == 404 && session.is_some() {
            return Ok(Reply::NoSession);
        }
        anyhow::ensure!(status < 300, "daemon answered {status}");
        if capture_session && let Some(id) = resp.headers().get("mcp-session-id") {
            let id = id.to_str()?.to_string();
            *self.session.lock().unwrap() = Some((id, String::new()));
        }
        let sse = resp.headers().get("content-type")
            .is_some_and(|c| c.to_str().unwrap_or("").starts_with("text/event-stream"));
        let mut msgs = vec![];
        let reader = BufReader::new(resp.body_mut().as_reader());
        if sse {
            for line in reader.lines() {
                let line = line?;
                if let Some(d) = line.strip_prefix("data:") {
                    let d = d.trim();
                    if d.starts_with('{') {
                        msgs.push(d.to_string());
                    }
                }
            }
        } else {
            let text: String = reader.lines().collect::<std::io::Result<Vec<_>>>()?.join("\n");
            if !text.trim().is_empty() {
                msgs.push(text);
            }
        }
        Ok(Reply::Messages(msgs))
    }

    /// Open a session with the client's `initialize`; returns the daemon's messages.
    fn open(&self, init: &str) -> Result<Vec<String>> {
        *self.session.lock().unwrap() = None;
        let Reply::Messages(msgs) = self.post(init, true)? else { unreachable!() };
        let version = msgs.iter()
            .filter_map(|m| serde_json::from_str::<Value>(m).ok())
            .find_map(|v| v["result"]["protocolVersion"].as_str().map(str::to_string))
            .unwrap_or_default();
        if let Some(s) = self.session.lock().unwrap().as_mut() {
            s.1 = version;
        }
        let ready = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        if self.session.lock().unwrap().is_some() {
            self.post(ready, false)?;
        }
        Ok(msgs)
    }

    /// Reopen after the daemon lost the session or went away.
    fn reopen(&self) -> Result<()> {
        crate::mcp::ensure_daemon(&self.addr)?;
        let init = self.init.lock().unwrap().clone().ok_or_else(|| anyhow!("no initialize to replay"))?;
        self.open(&init)?;
        Ok(())
    }

    fn forward(&self, line: &str) -> Result<Vec<String>> {
        match self.post(line, false) {
            Ok(Reply::Messages(m)) => return Ok(m),
            Ok(Reply::NoSession) => {}
            Err(e) => eprintln!("refx connect: {e:#}; reconnecting"),
        }
        self.reopen()?;
        match self.post(line, false)? {
            Reply::Messages(m) => Ok(m),
            Reply::NoSession => Err(anyhow!("daemon refused the new session")),
        }
    }
}

pub fn connect(addr: &str) -> Result<()> {
    let url = crate::mcp::ensure_daemon(addr)?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();
    let link = Arc::new(Link {
        addr: addr.to_string(),
        url,
        agent,
        session: Mutex::new(None),
        init: Mutex::new(None),
        out: Mutex::new(std::io::stdout()),
    });
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
        match msg["method"].as_str() {
            // The handshake runs in order: everything after it needs the session.
            Some("initialize") => {
                *link.init.lock().unwrap() = Some(line.clone());
                match link.open(&line) {
                    Ok(msgs) => msgs.iter().for_each(|m| link.emit(m)),
                    Err(e) => link.emit(&error_reply(&msg, &e)),
                }
            }
            // `open` already sent it to the daemon.
            Some("notifications/initialized") => {}
            _ => {
                let link = link.clone();
                std::thread::spawn(move || match link.forward(&line) {
                    Ok(msgs) => msgs.iter().for_each(|m| link.emit(m)),
                    Err(e) if !msg["id"].is_null() => link.emit(&error_reply(&msg, &e)),
                    Err(e) => eprintln!("refx connect: {e:#}"),
                });
            }
        }
    }
    Ok(())
}

/// A JSON-RPC error for a request the daemon could not answer, so the client does not wait.
fn error_reply(req: &Value, e: &anyhow::Error) -> String {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": req["id"],
        "error": { "code": -32603, "message": format!("refx daemon: {e:#}") },
    })
    .to_string()
}
