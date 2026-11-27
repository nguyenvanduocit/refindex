# refx: hybrid search over peer repos and a project's own docs

`grep` misses code that does the thing under another name. `refx` combines three indexes and
fuses their rankings:

| Index | Store | Finds |
|---|---|---|
| BM25 full text | tantivy (`fts/`) | exact identifiers, words, paths; identifiers also split into camel/snake subwords |
| Embeddings | `vectors.bin` (f16, exact scan) | code and docs by meaning: "how are flex items broken into lines" |
| Symbol graph | SQLite (`graph.db`): names, defs, refs, bug links | definitions a hit uses, callers, tests, cited crbug/webkit/github bugs |

Chunks are definitions (functions, types, impls, classes, with their doc comment) cut by
tree-sitter for Rust, C, C++, Objective-C, Kotlin, Java, TypeScript/TSX, JavaScript, Python,
Go, Swift and Zig; heading sections for Markdown/AsciiDoc/RST; line windows for code without a
grammar here (shaders, IDL, Dart, C#, QML, Slint, CSS...). The syntax tree also decides:

- where a long function is cut: between the statements of its body, never inside one; each
  continuation piece repeats the signature line in its embedding text;
- what is not a chunk of its own: body-less one-line declarations (`mod x;`, `struct Unit;`,
  forward declarations, prototypes) join the neighbouring items and keep their names for `def`;
- the graph edges: a chunk uses the identifier leaves of its code, not words in its comments
  or strings (files without a grammar fall back to identifier-shaped words).

Sources: every repo checked out under the peers directory (`references/` by default), plus the
project's own document trees listed in `own` (indexed as `@docs`, `@notes`, ...).

Ranking: reciprocal-rank fusion of BM25 and vector lists; a cross-encoder
(`granite-embedding-reranker-english-r2`, ModernBERT, our own SiLU head on candle's backbone)
reads (query, chunk) for the fused top 32 and adds its order as a steep list (k = 10, not 60)
scaled by its relevance, so its confident top decides the head while poor scores leave the
fused order alone (`rr#n(relevance)` in the output; `--no-rerank` or MCP `rerank=false` to
skip; about 2 s on Metal, plus a one-off 2 s model load in the CLI); then one step of
personalised PageRank over the hit subgraph. Every hit spreads its score over the definitions it uses,
weighted by how specific each name is (few definitions, few users). A definition that two or
more hits use but neither list found is pulled in from the users' repos.

## Setup

```sh
cargo build --release                # macOS: embeddings run on Metal
cp refx.example.toml <project>/refx.toml   # then edit it
cd <project> && refx index && refx embed
```

`refx` reads `refx.toml` from `$REFX_CONFIG`, or from the current directory or its nearest
parent. Paths in it are relative to the file, whose directory is the project root. Without a
config file the project root is the current directory and only `references/` is indexed. See
[refx.example.toml](refx.example.toml) for every key:

| Key | Meaning |
|---|---|
| `peers` | directory with one checkout per peer repo (default `references`); also the citation prefix |
| `home` | index home: graph db, tantivy dir, vectors, model cache (default `.refx`) |
| `own` | the project's document trees, indexed as `@<dir>` |
| `xml_repos` | peer repos whose `.xml` files are design documents |
| `[entities]` | `adr_dir`, `failure_classes`, `decisions`, `ledger`: project vocabulary that becomes graph nodes |
| `mcp_instructions` | text the MCP server gives agents on connect |
| `[areas.<name>]` | `about`, `globs`: a work area mapped onto peer paths (`--area`) |
| `[concepts.<name>]` | `terms`: project vocabulary <-> the names peers use (BM25 expansion) |

`REFX_CONFIG`, `REFX_HOME` and `REFX_REFS` override the file. Areas and concepts take effect
without a reindex. `REFX_DEVICE=cpu` keeps the models off Metal; `REFX_TIMING=1` prints
rerank batch times. An index built by an older chunker (`graph.db` format) is refused with a
note to rebuild it; `vectors.bin` is keyed by text and reused.

## Use

```sh
refx index                 # incremental: git blob ids per repo, mtimes for own docs
refx embed                 # vectors for new chunk texts; resumable
refx s how does chromium coalesce scroll updates -r chromium -k 8 -s 15
refx def collect_flex_lines
refx refs LayoutBox -r chromium --tests
refx related '#12345'      # or repo/path:line
refx show references/taffy/src/compute/flexbox.rs:1272-1373 -C 5
refx bug crbug:1234567
refx stats
```

Peer research:

```sh
refx areas                                        # work areas of refx.toml
refx peers keyed list diffing with moves -a list  # best hits per peer, as citations
refx class                                        # failure classes (entities.failure_classes)
refx class 3 mailbox worker                       # peer tests / bug-linked code for class 3
refx s scroll anchoring after relayout -a layout --evidence
refx related references/skia/src/gpu/graphite/Device.cpp:2279   # incl. which docs cite it
```

- Hits print as `<peers>/<repo>/<path>:a-b` (or `<own dir>/...`), ready to cite in a note.
- `--evidence` keeps tests, code citing bugs (crbug, webkit, mozilla, github) and definitions
  named after failures (regress, crash, leak, race...).
- Citations in the own docs form graph edges: a peer hit shows `cited by: docs/adr/...md:N`,
  and `related` on a doc chunk lists the peer code it cites.

Project knowledge in the graph and the ranking:

```sh
refx about ADR-0012          # status, supersedes/superseded by, every doc and peer chunk mentioning it
refx about T42               # a ledger task (entities.ledger)
refx about class 3           # a failure class, its examples, peer code for its cited bugs
refx about "Leptos #1234"    # a peer bug the docs cite -> peer tests/code naming that bug
refx concepts invalidation   # what BM25 also searches for this word
```

- Entities: ADRs (title, status, supersession), ledger tasks, failure classes, decisions
  (rows of the decisions TSV), peer bugs (`Leptos #1234` in the docs, `issue_1234` /
  `fixes #1234` / github issue links in peer code). Chunks mention entities; `related` lists them.
- Concept expansion (BM25 only): `[concepts]` maps project vocabulary to the names peers use
  (invalidation -> markNeedsLayout, SetNeedsLayout, queue_resize...), plus links learned at
  index time from citations: the words around `references/<repo>/<path>:a-b` in the docs
  paired with the symbols the cited peer chunk defines. A link needs two citations that agree,
  and words around a tenth of all citations teach nothing. Search prints `expanded: ...`.
- Ranking after RRF and the symbol-graph step: peer code the docs cite (`+cited`) and code
  tied to a peer bug the docs mention (`+known-bug`) move up; sections of superseded ADRs
  (`-superseded`) move down so the decision in force ranks first.

Agents use the MCP server `refx mcp` (stdio), with tools `ref_search`, `ref_peers`,
`ref_failure_class`, `ref_about`, `ref_concepts`, `ref_def`, `ref_refs`, `ref_related`,
`ref_read`, `ref_bug`, `ref_repos`. Register it with the project as the working directory, or
set `REFX_CONFIG`:

```json
{ "mcpServers": { "refindex": { "command": "refx", "args": ["mcp"] } } }
```

The embedding model `ibm-granite/granite-embedding-small-english-r2` (ModernBERT, 384 dims,
CLS pooling) is read from the Hugging Face cache or downloaded into the index home.

## Checks

- `cargo test`: chunker contracts (doc comments attach, large containers split into members,
  C++ qualified names, subwords, bug links, doc sections).
- `uv run referee/embed_referee.py <refx>`: an independent referee that computes the same
  embeddings with sentence-transformers and requires cosine > 0.995 per text. `--broken` shows
  that it fails when the vectors differ.
- `uv run referee/rerank_referee.py <refx>`: the same for the cross-encoder against
  sentence-transformers' CrossEncoder (scores within 0.01, same order); `--broken` must fail.
