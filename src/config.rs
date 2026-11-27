//! Paths, file selection and model settings.
//!
//! The project is described by `refx.toml`, found at `$REFX_CONFIG` or in the current directory
//! or the nearest parent. Paths in it are relative to the file's directory (the project root).
//! Without a config file the project root is the current directory and only `references/` is
//! indexed. `REFX_HOME` and `REFX_REFS` override `home` and `peers`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

pub const MODEL: &str = "ibm-granite/granite-embedding-small-english-r2";
pub const MODEL_FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];
pub const EMBED_TOKENS: usize = 256;
pub const EMBED_CHARS: usize = 2000;

pub const MAX_FILE_BYTES: usize = 400_000;
pub const MAX_AVG_LINE: usize = 220;
pub const MAX_CHUNK_LINES: usize = 120;
pub const WINDOW_LINES: usize = 80;
pub const WINDOW_OVERLAP: usize = 10;
pub const GROUP_LINES: usize = 60;

pub const EXCLUDE_DIRS: &[&str] = &[
    "node_modules", "third_party", "third-party", "vendor", "vendored", "external", "deps",
    "web-platform-tests", "wpt", "LayoutTests", "JSTests", "PerformanceTests", "testdata",
    "test-data", "fixtures", "__snapshots__", "snapshots", "dist", "build", "out", "target", "gen",
    "generated", ".git", "golden", "goldens", "expected", "expectations",
];
/// `.txt` only where it carries design text.
pub const TXT_ONLY_UNDER: &[&str] = &["docs/", "doc/", "design", "README", "NOTES"];

pub const CONFIG_FILE: &str = "refx.toml";

/// `refx.toml`. Every key is optional.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Directory holding one checkout per peer repo. Default `references`.
    pub peers: Option<String>,
    /// Index home: sqlite graph db, tantivy dir, vectors file, model cache. Default `.refx`.
    pub home: Option<String>,
    /// The project's own document trees, indexed as `@<dir>` (e.g. `docs` -> `@docs`).
    pub own: Vec<String>,
    /// Peer repos whose `.xml` files are design documents (protocol specs).
    pub xml_repos: Vec<String>,
    /// Project vocabulary that becomes graph entities. Paths must lie inside an `own` tree.
    pub entities: Entities,
    /// Text the MCP server hands to agents on connect; a generic text when unset.
    pub mcp_instructions: Option<String>,
    /// Work areas mapped onto peer files: `[areas.<name>] about, globs`.
    pub areas: BTreeMap<String, Area>,
    /// Project vocabulary <-> peer names: `[concepts.<name>] terms`.
    pub concepts: BTreeMap<String, Concept>,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Entities {
    /// Directory of numbered ADRs (`0026-title.md`).
    pub adr_dir: Option<String>,
    /// Markdown list of failure classes (`N. **Title.** question`).
    pub failure_classes: Option<String>,
    /// TSV of decisions (`D012\t...`).
    pub decisions: Option<String>,
    /// JSON task ledger: `{"tasks": {"<id>": {"title", "status"}}}`.
    pub ledger: Option<String>,
}

struct Loaded {
    root: PathBuf,
    cfg: Config,
}

static LOADED: LazyLock<Loaded> = LazyLock::new(|| match load() {
    Ok(l) => l,
    Err(e) => {
        eprintln!("refx: {e:#}");
        std::process::exit(2);
    }
});

fn load() -> anyhow::Result<Loaded> {
    use anyhow::Context;
    let file = match std::env::var_os("REFX_CONFIG") {
        Some(p) => Some(std::path::absolute(PathBuf::from(p))?),
        None => {
            let cwd = std::env::current_dir()?;
            cwd.ancestors().map(|d| d.join(CONFIG_FILE)).find(|p| p.is_file())
        }
    };
    let Some(file) = file else {
        return Ok(Loaded { root: std::env::current_dir()?, cfg: Config::default() });
    };
    let text = std::fs::read_to_string(&file).with_context(|| format!("reading {}", file.display()))?;
    let cfg = toml::from_str(&text).with_context(|| format!("parsing {}", file.display()))?;
    Ok(Loaded { root: file.parent().unwrap_or(Path::new(".")).to_path_buf(), cfg })
}

pub fn config() -> &'static Config {
    &LOADED.cfg
}

/// The project root: the directory of `refx.toml`.
pub fn root() -> PathBuf {
    LOADED.root.clone()
}

/// How the project's docs spell the peers directory in citations (`references`).
pub fn peers_prefix() -> &'static str {
    config().peers.as_deref().unwrap_or("references").trim_end_matches('/')
}

pub fn refs_dir() -> PathBuf {
    std::env::var_os("REFX_REFS").map(PathBuf::from).unwrap_or_else(|| root().join(peers_prefix()))
}

/// Index home: sqlite graph db, tantivy dir, vectors file, model cache.
pub fn home() -> PathBuf {
    std::env::var_os("REFX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join(config().home.as_deref().unwrap_or(".refx")))
}

/// A tree of files to index: a peer repo under the peers directory, or one of the project's
/// own document trees (named `@<dir>`).
pub struct Source {
    pub name: String,
    pub root: PathBuf,
}

/// A work area of the project mapped onto peer files (`[areas.<name>]`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Area {
    pub about: String,
    pub globs: Vec<String>,
}

pub fn areas() -> anyhow::Result<BTreeMap<String, Area>> {
    Ok(config().areas.clone())
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Concept {
    pub terms: Vec<String>,
}

/// Project vocabulary <-> peer names (`[concepts.<name>]`).
pub fn concepts() -> anyhow::Result<BTreeMap<String, Concept>> {
    Ok(config().concepts.clone())
}

pub fn own_dirs() -> &'static [String] {
    &config().own
}

/// A configured project path split into (own source `@<dir>`, path inside it).
fn in_own(path: &str) -> Option<(String, String)> {
    let path = path.trim_matches('/');
    own_dirs().iter().find_map(|d| {
        let rest = path.strip_prefix(d.trim_matches('/'))?.strip_prefix('/')?;
        Some((format!("@{}", d.trim_matches('/')), rest.to_string()))
    })
}

/// ADR directory as (source, dir inside it), e.g. (`@docs`, `adr`).
pub fn adr_dir() -> Option<(String, String)> {
    in_own(config().entities.adr_dir.as_deref()?)
}

/// Failure-class list as (source, path inside it).
pub fn failure_classes_doc() -> Option<(String, String)> {
    in_own(config().entities.failure_classes.as_deref()?)
}

/// Decisions TSV as (source, path inside it).
pub fn decisions_doc() -> Option<(String, String)> {
    in_own(config().entities.decisions.as_deref()?)
}

pub fn failure_classes_path() -> anyhow::Result<PathBuf> {
    let p = config().entities.failure_classes.as_deref()
        .ok_or_else(|| anyhow::anyhow!("no entities.failure_classes in {CONFIG_FILE}"))?;
    Ok(root().join(p))
}

pub fn ledger_path() -> Option<PathBuf> {
    config().entities.ledger.as_deref().map(|p| root().join(p))
}

/// Where a hit lives, as this project cites it: `references/<repo>/<path>` or `docs/<path>`.
pub fn cite_path(repo: &str, path: &str) -> String {
    match repo.strip_prefix('@') {
        Some(d) => format!("{d}/{path}"),
        None => format!("{}/{repo}/{path}", peers_prefix()),
    }
}

pub fn source_root(name: &str) -> PathBuf {
    match name.strip_prefix('@') {
        Some(d) => root().join(d),
        None => refs_dir().join(name),
    }
}

pub fn sources() -> Vec<Source> {
    let mut out: Vec<Source> = std::fs::read_dir(refs_dir())
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| !n.starts_with('.'))
                .map(|n| Source { root: refs_dir().join(&n), name: n })
                .collect()
        })
        .unwrap_or_default();
    if std::env::var_os("REFX_REFS").is_none() {
        for d in own_dirs() {
            let d = d.trim_matches('/');
            let root = root().join(d);
            if root.is_dir() {
                out.push(Source { name: format!("@{d}"), root });
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Language of a path: a tree-sitter grammar name, `doc` (chunk by headings) or `plain` (code
/// without a grammar here, chunked by line windows). `None` = not indexed.
pub fn lang_of(repo: &str, path: &str) -> Option<&'static str> {
    let mut parts = path.split('/').collect::<Vec<_>>();
    let file = parts.pop()?;
    let own = repo.starts_with('@');
    if !own && parts.iter().any(|p| EXCLUDE_DIRS.contains(p)) {
        return None;
    }
    let ext = file.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase())?;
    Some(match ext.as_str() {
        "rs" => "rust",
        "c" => "c",
        "h" | "cc" | "cpp" | "cxx" | "hpp" | "hh" | "inl" => "cpp",
        "m" | "mm" => "objc",
        "kt" | "kts" => "kotlin",
        "java" => "java",
        "ts" | "mts" => "typescript",
        "tsx" => "tsx",
        "js" | "mjs" | "cjs" | "jsx" => "javascript",
        "py" => "python",
        "go" => "go",
        "zig" => "zig",
        "swift" => "swift",
        "md" | "mdx" | "rst" | "adoc" => "doc",
        // Code with no grammar linked in: shaders, IDLs, DSLs, styles, other languages.
        "vue" | "svelte" | "astro" | "dart" | "cs" | "fs" | "lua" | "gd" | "scala" | "rb"
        | "ex" | "hs" | "ml" | "wgsl" | "wesl" | "glsl" | "frag" | "vert" | "comp" | "hlsl"
        | "hlsli" | "metal" | "sksl" | "idl" | "webidl" | "aidl" | "mojom" | "proto" | "qml"
        | "slint" | "rsx" | "splash" | "css" | "scss" | "less" => "plain",
        "txt" if own || TXT_ONLY_UNDER.iter().any(|m| path.contains(m)) => "doc",
        "xml" if config().xml_repos.iter().any(|r| r == repo) => "doc",
        "tsv" if own => "tsv",
        "sh" | "html" | "log" if own => "plain",
        _ => return None,
    })
}

pub fn ts_language(lang: &str) -> Option<tree_sitter::Language> {
    Some(match lang {
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        "c" => tree_sitter_c::LANGUAGE.into(),
        "cpp" => tree_sitter_cpp::LANGUAGE.into(),
        "objc" => tree_sitter_objc::LANGUAGE.into(),
        "kotlin" => tree_sitter_kotlin_ng::LANGUAGE.into(),
        "java" => tree_sitter_java::LANGUAGE.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "javascript" => tree_sitter_javascript::LANGUAGE.into(),
        "python" => tree_sitter_python::LANGUAGE.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "zig" => tree_sitter_zig::LANGUAGE.into(),
        "swift" => tree_sitter_swift::LANGUAGE.into(),
        _ => return None,
    })
}

pub fn is_test_path(path: &str) -> bool {
    const M: &[&str] = &[
        "/test/", "/tests/", "/testing/", "_test.", "_tests.", "_unittest.", "/test_", "/spec/",
        ".spec.", ".test.", "/unittests/", "/browsertests/",
    ];
    let p = format!("/{}", path.to_ascii_lowercase());
    M.iter().any(|m| p.contains(m))
}
