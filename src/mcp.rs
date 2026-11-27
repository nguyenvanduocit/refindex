//! MCP server over the index, for coding agents: stdio (`refx mcp`, one process per client) or
//! one HTTP daemon for every project on the machine (`refx serve`, started by `refx up`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::config::{self, Project, Scope};

use anyhow::Result;
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{ServerCapabilities, ServerConfig};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::search::{Engine, Filter, fmt_hit, fmt_list};

/// Handed to agents on connect unless `refx.toml` sets `mcp_instructions`.
const INSTRUCTIONS: &str = "Peer research: hybrid search (BM25 + code embeddings + symbol graph) \
over the peer repos and this project's own docs (sources named @<dir>), so it finds code by \
meaning as well as by name. Workflow: ref_peers for how two or three peers solve a contract \
(restrict with the project's work areas); ref_failure_class N for peer tests and bug-linked code \
for a failure class; evidence=true to keep only tests, bug-citing code and definitions named after \
failures; ref_related from a hit for what it uses, who uses it, its tests, cited bugs, and which of \
the project's docs already cite it. Hits print as `<peers>/<repo>/<path>:a-b`, ready to cite. \
ref_def / ref_refs for exact symbols, ref_read to read. Use grep only for exact strings.";

/// The MCP server of one project. Clones share the project's engine: every client of that
/// project (stdio: one; daemon: all sessions) uses one engine, so vectors and caches load
/// once; the models are shared by all projects of the process (`search::warm_models`).
#[derive(Clone)]
pub struct Server {
    project: Arc<Project>,
    engine: Arc<Mutex<Option<Opened>>>,
    tool_router: ToolRouter<Self>,
}

/// An open engine and the identity of the `graph.db` it reads. A rebuilt index is a new file,
/// which a long-lived daemon would otherwise keep reading through the deleted one; vectors
/// reload by themselves when `vectors.bin` grows.
struct Opened {
    engine: Engine,
    db_id: Option<(u64, u64)>,
}

fn db_id() -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let m = std::fs::metadata(crate::paths::Paths::new().db).ok()?;
    Some((m.dev(), m.ino()))
}

/// The project's engine, with the project's config selected on this thread while held.
pub struct EngineGuard<'a> {
    g: MutexGuard<'a, Option<Opened>>,
    _scope: Scope,
}

impl std::ops::Deref for EngineGuard<'_> {
    type Target = Engine;
    fn deref(&self) -> &Engine {
        &self.g.as_ref().unwrap().engine
    }
}

impl std::ops::DerefMut for EngineGuard<'_> {
    fn deref_mut(&mut self) -> &mut Engine {
        &mut self.g.as_mut().unwrap().engine
    }
}

impl Server {
    pub fn new(project: Arc<Project>) -> Self {
        Server { project, engine: Arc::new(Mutex::new(None)), tool_router: Server::tool_router() }
    }

    /// The engine: opened on first use, reopened if the index was rebuilt. A panic in an
    /// earlier call does not poison it for later ones: the engine holds no half-written state.
    fn engine(&self) -> Result<EngineGuard<'_>> {
        let scope = config::enter(self.project.clone());
        let mut g = self.engine.lock().unwrap_or_else(|p| p.into_inner());
        let now = db_id();
        if g.as_ref().is_none_or(|o| now.is_some() && now != o.db_id) {
            *g = Some(Opened { engine: Engine::open()?, db_id: now });
        }
        Ok(EngineGuard { g, _scope: scope })
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// Natural language or identifiers.
    pub query: String,
    /// Restrict to repos, e.g. ["chromium","servo"] or ["@docs"].
    pub repos: Option<Vec<String>>,
    /// rust|cpp|c|objc|kotlin|java|typescript|tsx|javascript|python|go|swift|zig|doc|plain
    pub lang: Option<String>,
    /// Glob or substring of the file path.
    pub path: Option<String>,
    /// true: only tests; false: no tests.
    pub tests: Option<bool>,
    /// Work areas: names from `[areas]` in refx.toml, or `ours` for the project's own docs.
    pub areas: Option<Vec<String>>,
    /// Only failure evidence: tests, code citing bugs, definitions named after failures.
    pub evidence: Option<bool>,
    /// Number of results (default 10).
    pub k: Option<usize>,
    /// Lines of each hit to show (default 12).
    pub show_lines: Option<usize>,
    /// hybrid (default) | fts | vec
    pub mode: Option<String>,
    /// Cross-encoder rerank of the top 50 (default true; false is faster).
    pub rerank: Option<bool>,
}

impl SearchArgs {
    fn filter(&self) -> Filter {
        Filter {
            repos: self.repos.clone().unwrap_or_default(),
            lang: self.lang.clone(),
            path: self.path.clone(),
            tests: self.tests,
            areas: self.areas.clone().unwrap_or_default(),
            evidence: self.evidence.unwrap_or(false),
        }
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct ClassArgs {
    /// Class number in the project's failure-class list; omit to list classes.
    pub number: Option<usize>,
    /// Extra words to narrow the query.
    pub extra: Option<String>,
    pub repos: Option<Vec<String>>,
    pub areas: Option<Vec<String>>,
    /// Number of peer repos (default 5).
    pub peer_repos: Option<usize>,
    pub show_lines: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct SymbolArgs {
    /// Symbol name; qualified names (a::B::c) are reduced to the last segment.
    pub name: String,
    pub repos: Option<Vec<String>>,
    /// true: only tests; false: no tests.
    pub tests: Option<bool>,
    pub limit: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ChunkArgs {
    /// `#id` from a result, or `repo/path:line`.
    pub chunk: String,
    /// Extra lines of context around the chunk (ref_read only).
    pub context: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct EntityArgs {
    /// ADR-0012, T42, class 3, D012, "Leptos #4457" (ref_about) or words (ref_concepts).
    pub entity: String,
    pub show_lines: Option<usize>,
}

#[derive(Deserialize, JsonSchema)]
pub struct BugArgs {
    /// crbug:123, webkit:123, mozilla:123, gh:owner/repo#12, or a bare number.
    pub bug: String,
}

fn err(e: anyhow::Error) -> String {
    format!("error: {e:#}")
}

#[tool_router(router = tool_router)]
impl Server {
    #[tool(description = "Hybrid search (BM25 + embeddings + symbol graph) over the peer repos and the project's docs. Each hit shows #id and why it matched.")]
    async fn ref_search(&self, Parameters(a): Parameters<SearchArgs>) -> String {
        let mut e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        let f = a.filter();
        e.rerank = a.rerank.unwrap_or(true);
        match e.search(&a.query, a.k.unwrap_or(10), &f, a.mode.as_deref().unwrap_or("hybrid")) {
            Ok(h) => {
                let exp = if e.last_expansion.is_empty() { String::new() }
                          else { format!("expanded: {}\n", e.last_expansion.join(", ")) };
                exp + &fmt_list(&e, &h, a.show_lines.unwrap_or(12))
            }
            Err(x) => err(x),
        }
    }

    #[tool(description = "How the peers do it: the best hits of each peer repo for one contract, printed as `<peers>/<repo>/<path>:a-b` citations ready for a research note. Use areas to stay in one work area and evidence=true for tests and bug-linked code.")]
    async fn ref_peers(&self, Parameters(a): Parameters<SearchArgs>) -> String {
        let mut e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        let f = a.filter();
        e.rerank = a.rerank.unwrap_or(true);
        match e.peers(&a.query, &f, a.k.unwrap_or(4), 2) {
            Ok(h) => fmt_list(&e, &h, a.show_lines.unwrap_or(12)),
            Err(x) => err(x),
        }
    }

    #[tool(description = "Peer evidence for a failure class of the project's failure-class list: tests and bug-linked code in the peers for that class. number omitted: list the classes. extra narrows the query (e.g. 'mailbox worker').")]
    async fn ref_failure_class(&self, Parameters(a): Parameters<ClassArgs>) -> String {
        let run = || -> Result<String> {
            let mut e = self.engine()?; // first: it also selects the project's config
            let classes = crate::search::failure_classes()?;
            let Some(n) = a.number else {
                return Ok(classes.iter().map(|c| format!("{:3}. {} ({})\n", c.number, c.title, c.section)).collect());
            };
            let c = classes.iter().find(|c| c.number == n).ok_or_else(|| anyhow::anyhow!("no class {n}"))?;
            let f = Filter { repos: a.repos.unwrap_or_default(), areas: a.areas.unwrap_or_default(),
                             evidence: true, ..Default::default() };
            let q = format!("{} {}", c.query(), a.extra.unwrap_or_default());
            let hits = e.peers(&q, &f, a.peer_repos.unwrap_or(5), 2)?;
            Ok(format!("class {}: {} ({})\n{}", c.number, c.title, c.section, fmt_list(&e, &hits, a.show_lines.unwrap_or(12))))
        };
        run().unwrap_or_else(err)
    }

    #[tool(description = "A project entity and everything that mentions it, our docs first: ADR-0012, a ledger task (T42), a failure class ('class 3'), a decision (D012), or a peer bug our docs cite ('Leptos #4457'). Shows status, supersession, and peer code/tests tied to it.")]
    async fn ref_about(&self, Parameters(a): Parameters<EntityArgs>) -> String {
        let e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        crate::about_text(&e, &a.entity, a.show_lines.unwrap_or(0)).unwrap_or_else(err)
    }

    #[tool(description = "What a word of ours expands to in BM25: [concepts] groups of refx.toml (our vocabulary <-> peer names) and links learned from citations in our docs.")]
    async fn ref_concepts(&self, Parameters(a): Parameters<EntityArgs>) -> String {
        let e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        match e.expand(&a.entity) {
            Ok((terms, added)) => format!("expansions: {}\nbm25 terms: {}\n", added.join(", "),
                terms.iter().map(|(t, w)| format!("{t}^{w}")).collect::<Vec<_>>().join(" ")),
            Err(x) => err(x),
        }
    }

    #[tool(description = "Where a symbol (function, type, trait, class, method) is defined.")]
    async fn ref_def(&self, Parameters(a): Parameters<SymbolArgs>) -> String {
        let e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        let f = Filter { repos: a.repos.unwrap_or_default(), tests: a.tests, ..Default::default() };
        e.definitions(&a.name, &f, a.limit.unwrap_or(30)).map(|h| fmt_list(&e, &h, 0)).unwrap_or_else(err)
    }

    #[tool(description = "Chunks that mention a symbol: callers, implementors, tests, docs.")]
    async fn ref_refs(&self, Parameters(a): Parameters<SymbolArgs>) -> String {
        let e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        let f = Filter { repos: a.repos.unwrap_or_default(), tests: a.tests, ..Default::default() };
        e.references(&a.name, &f, a.limit.unwrap_or(50)).map(|h| fmt_list(&e, &h, 0)).unwrap_or_else(err)
    }

    #[tool(description = "Graph neighbours of a chunk: what it uses, what uses it, tests touching it, nearby chunks in the file, bugs it cites.")]
    async fn ref_related(&self, Parameters(a): Parameters<ChunkArgs>) -> String {
        let e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        let run = || -> Result<String> {
            let (me, groups, bugs) = e.related(e.locate(&a.chunk)?.id, 15)?;
            let mut s = fmt_hit(&me, None) + "\n";
            for (k, hits) in groups {
                s += &format!("== {k}\n{}", fmt_list(&e, &hits, 0));
            }
            if !bugs.is_empty() {
                s += &format!("== bugs: {}\n", bugs.join(", "));
            }
            Ok(s)
        };
        run().unwrap_or_else(err)
    }

    #[tool(description = "Read a chunk with line numbers, plus optional context lines.")]
    async fn ref_read(&self, Parameters(a): Parameters<ChunkArgs>) -> String {
        let e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        e.locate(&a.chunk)
            .map(|h| format!("{}\n{}", fmt_hit(&h, None), e.read(&h, a.context.unwrap_or(0))))
            .unwrap_or_else(err)
    }

    #[tool(description = "Code, tests and docs that cite a bug id.")]
    async fn ref_bug(&self, Parameters(a): Parameters<BugArgs>) -> String {
        let e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        e.bug(&a.bug, 100).map(|h| fmt_list(&e, &h, 0)).unwrap_or_else(err)
    }

    #[tool(description = "Indexed repositories and sources with commit and file counts, and index stats.")]
    async fn ref_repos(&self) -> String {
        let mut e = match self.engine() { Ok(e) => e, Err(x) => return err(x) };
        let mut run = || -> Result<String> {
            let mut s = String::new();
            for (n, h, c) in e.repos()? {
                s += &format!("{n:28} {:14} {c}\n", &h[..h.len().min(12)]);
            }
            for (k, v) in e.stats()? {
                s += &format!("{k}: {v}\n");
            }
            Ok(s)
        };
        run().unwrap_or_else(err)
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Server {
    fn get_info(&self) -> ServerConfig {
        let mut info = ServerConfig::new(ServerCapabilities::builder().enable_tools().build());
        info.instructions =
            Some(self.project.cfg.mcp_instructions.clone().unwrap_or_else(|| INSTRUCTIONS.into()));
        info
    }
}

pub fn serve() -> Result<()> {
    let server = Server::new(config::project());
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let svc = server.serve(rmcp::transport::stdio()).await?;
        svc.waiting().await?;
        anyhow::Ok(())
    })
}

pub const DEFAULT_ADDR: &str = "127.0.0.1:7311";

/// Register the current project with the daemon and start the daemon if it is not running;
/// prints the project's MCP URL. Meant for a session-start hook: quick when the daemon runs,
/// and the daemon it starts inherits the caller's file access (no launchd or privacy grants).
pub fn up(addr: &str) -> Result<()> {
    use std::net::{SocketAddr, TcpStream};
    use std::os::unix::process::CommandExt;
    use std::time::{Duration, Instant};
    let file = config::find_config()?
        .ok_or_else(|| anyhow::anyhow!("no {} here or in a parent directory", config::CONFIG_FILE))?;
    let name = Project::load(&file)?.name();
    if config::register(&name, &file)? {
        eprintln!("refx: registered {name} -> {}", file.display());
    }
    let sock: SocketAddr = addr.parse()?;
    let alive = || TcpStream::connect_timeout(&sock, Duration::from_millis(300)).is_ok();
    if !alive() {
        let log_path = config::registry_path().with_file_name("serve.log");
        std::fs::create_dir_all(log_path.parent().unwrap())?;
        let log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path)?;
        std::process::Command::new(std::env::current_exe()?)
            .args(["serve", "--addr", addr])
            // Not the starting project's directory: the daemon serves every project, and a
            // lookup that missed its project scope must fail, not read this one's config.
            .current_dir("/")
            .stdin(std::process::Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log)
            .process_group(0) // not killed with the hook's process group
            .spawn()?;
        let t0 = Instant::now();
        while !alive() {
            anyhow::ensure!(t0.elapsed() < Duration::from_secs(20),
                "refx serve did not start; see {}", log_path.display());
            std::thread::sleep(Duration::from_millis(100));
        }
        eprintln!("refx: daemon started at {addr} (log {})", log_path.display());
    }
    println!("http://{addr}/mcp/{name}");
    Ok(())
}

type HttpService = rmcp::transport::StreamableHttpService<
    Server,
    rmcp::transport::streamable_http_server::session::local::LocalSessionManager,
>;

/// The daemon's projects, by name, opened on their first request from the registry.
struct Daemon {
    services: Mutex<HashMap<String, HttpService>>,
}

impl Daemon {
    fn service(&self, name: &str) -> Result<HttpService> {
        use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
        let mut s = self.services.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(svc) = s.get(name) {
            return Ok(svc.clone());
        }
        // Read on every miss, so a project registered after the daemon started just works.
        let file = config::registry()?.remove(name).ok_or_else(|| anyhow::anyhow!(
            "no project {name} in {}; run `refx up` in the project", config::registry_path().display()))?;
        let server = Server::new(Project::load(&file)?);
        let svc = HttpService::new(
            move || Ok(server.clone()),
            Arc::new(LocalSessionManager::default()),
            rmcp::transport::StreamableHttpServerConfig::default(),
        );
        eprintln!("refx: serving {name} ({})", file.display());
        s.insert(name.to_string(), svc.clone());
        Ok(svc)
    }
}

async fn route(
    axum::extract::State(d): axum::extract::State<Arc<Daemon>>,
    axum::extract::Path(name): axum::extract::Path<String>,
    req: axum::extract::Request,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    use tower::ServiceExt;
    match d.service(&name) {
        Ok(svc) => match svc.oneshot(req).await {
            Ok(r) => r.into_response(),
            Err(e) => match e {},
        },
        Err(e) => (axum::http::StatusCode::NOT_FOUND, format!("{e:#}\n")).into_response(),
    }
}

/// MCP over streamable HTTP for every registered project: `http://<addr>/mcp/<project>`.
/// One long-lived process, so the models load once for all projects and each project's index
/// once for all its sessions. Binds where told (default loopback); rmcp also checks the Host
/// header against loopback names.
pub fn serve_http(addr: &str) -> Result<()> {
    let daemon = Arc::new(Daemon { services: Mutex::new(HashMap::new()) });
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let app = axum::Router::new()
            .route("/mcp/{name}", axum::routing::any(route))
            .with_state(daemon);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        eprintln!("refx: MCP at http://{}/mcp/<project>", listener.local_addr()?);
        // Listening first, then warming the models, so `refx up` returns at once.
        tokio::task::spawn_blocking(crate::search::warm_models);
        // Ctrl-C in a terminal, SIGTERM from launchd/systemd.
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        axum::serve(listener, app).with_graceful_shutdown(async move {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
        }).await?;
        anyhow::Ok(())
    })
}
