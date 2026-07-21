#!/usr/bin/env bash
#
# HADES end-to-end pipeline smoke test (#103).
#
# Where scripts/cli_audit.sh asks "does every command run" against a dirty
# bident_burn, this asks "is the graph RIGHT" against a dedicated fresh
# database (bident_burn_smoke): full pipelines with assertions on the
# invariants that past regressions actually violated —
#
#   #165/#171  chunks + embeddings carry their parent/file foreign keys
#   #166       input canonicalization (same file via ./relative == absolute)
#              and the honest batch envelope (partial failure → exit 1)
#   #169       --force re-ingest survives (the declared-but-unused bind family)
#   #159       --force does not leak stale chunks (counts stable)
#   #158/#157  drift → retire → prune-orphans → validate close the loop
#
# Prerequisites: ArangoDB up (hades user), embedder service on :8087.
# The target database is created if missing and its collections truncated
# per run — bident_burn itself is never touched.
#
# Usage:
#   ./scripts/bident_burn_smoke.sh                # deployed binary
#   HADES_BIN=target/release/hades ./scripts/bident_burn_smoke.sh   # candidate
#
# Exit code: 0 all assertions pass, 1 otherwise.

set -u
shopt -s lastpipe

HADES_BIN=${HADES_BIN:-hades}
DB=bident_burn_smoke
EMBEDDER_URL=${EMBEDDER_URL:-http://localhost:8087}

PASS=0
FAIL=0
FAILURES=()
# NOT under /tmp: the extraction service runs with PrivateTmp, so files
# there are invisible to it and doc ingest fails with "File not found"
# even though the file exists locally. A home-scoped dir is visible to
# both this process and the services.
mkdir -p "$HOME/.cache"
WORK=$(mktemp -d -p "$HOME/.cache" hades_smoke_XXXXXX)
# mktemp creates 0700; the extraction service runs as its own user and
# must traverse into the fixture dir, so open it up for the run.
chmod 755 "$WORK"
trap 'rm -rf "$WORK"' EXIT

if [[ -t 1 ]]; then G=$'\033[32m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'; else G="" R="" D="" N=""; fi

h() { "$HADES_BIN" --db "$DB" "$@" 2>>"$WORK/stderr.log"; }

# assert "<name>" "<shell command producing output>" "<jq-style predicate on $OUT, bash -c evaluated>"
step() {
    local name=$1; shift
    if "$@" >/dev/null 2>>"$WORK/stderr.log"; then
        PASS=$((PASS+1)); echo "${G}✓${N} $name"
    else
        FAIL=$((FAIL+1)); FAILURES+=("$name"); echo "${R}✗${N} $name"
    fi
}

# check "<name>" <actual> <expected>
check() {
    local name=$1 actual=$2 expected=$3
    if [[ "$actual" == "$expected" ]]; then
        PASS=$((PASS+1)); echo "${G}✓${N} $name ${D}($actual)${N}"
    else
        FAIL=$((FAIL+1)); FAILURES+=("$name: expected '$expected', got '$actual'")
        echo "${R}✗${N} $name ${D}expected '$expected', got '$actual'${N}"
    fi
}

# check_min "<name>" <actual> <minimum>
check_min() {
    local name=$1 actual=$2 min=$3
    if [[ "$actual" =~ ^[0-9]+$ ]] && (( actual >= min )); then
        PASS=$((PASS+1)); echo "${G}✓${N} $name ${D}($actual ≥ $min)${N}"
    else
        FAIL=$((FAIL+1)); FAILURES+=("$name: expected ≥$min, got '$actual'")
        echo "${R}✗${N} $name ${D}expected ≥$min, got '$actual'${N}"
    fi
}

count() { h db count "$1" | jq -r '.data.count // 0'; }

echo "── HADES pipeline smoke ── binary: $HADES_BIN → db: $DB"
"$HADES_BIN" --version || { echo "hades binary not runnable"; exit 1; }

# ── 0. Service preflight ────────────────────────────────────────────────
curl -sf "$EMBEDDER_URL/v1/models" >/dev/null \
    || { echo "${R}embedder not reachable at $EMBEDDER_URL — the embedding assertions are the point; aborting${N}"; exit 1; }
echo "${G}✓${N} embedder reachable"

# ── 1. Database + collections (rerunnable) ──────────────────────────────
if ! h db collections >/dev/null; then
    step "create database $DB" h db create-database "$DB"
fi
for col in documents chunks embeddings persephone_tasks persephone_logs persephone_handoffs; do
    h db count "$col" >/dev/null || h db create "$col" >/dev/null
    h db truncate "$col" --yes >/dev/null 2>>"$WORK/stderr.log" || h db truncate "$col" >/dev/null
done
h db count persephone_edges >/dev/null || h db create persephone_edges --edge >/dev/null
step "collections provisioned" h db collections
# Codebase collections are auto-created by ingest; truncate if present from
# a prior run so counts start clean.
for col in codebase_files codebase_chunks codebase_embeddings codebase_symbols codebase_calls_edges codebase_implements_edges codebase_imports_edges codebase_defines_edges; do
    h db count "$col" >/dev/null 2>&1 && { h db truncate "$col" --yes >/dev/null 2>&1 || h db truncate "$col" >/dev/null 2>&1; }
done

# ── 2. Task lifecycle ───────────────────────────────────────────────────
TASK_KEY=$(h task create "smoke: pipeline check" --description "created by bident_burn_smoke.sh" | jq -r '.data.task._key // .data._key // empty')
check_min "task create returns a key" "${#TASK_KEY}" 1
step "task show"    h task show "$TASK_KEY"
step "task start"   h task start "$TASK_KEY"
step "task update"  h task update "$TASK_KEY" --priority high
step "task close"   h task close "$TASK_KEY"
STATUS=$(h task show "$TASK_KEY" | jq -r '.data.task.status // .data.status // empty')
check "task reached closed" "$STATUS" "closed"

# ── 3. Document ingest: canonicalization + honest envelope ──────────────
DOCS_DIR="$WORK/docs"; mkdir -p "$DOCS_DIR"
cat > "$DOCS_DIR/smoke_note.md" <<'MD'
# Smoke Note

The greeting subsystem emits a friendly hello to every caller.
It is exercised by the pipeline smoke test to verify chunking and embedding.
MD

step "doc ingest (absolute path)" h ingest "$DOCS_DIR/smoke_note.md"
DOCS_1=$(count documents)
check_min "documents written" "$DOCS_1" 1
check_min "chunks written" "$(count chunks)" 1
check_min "embeddings written" "$(count embeddings)" 1

# Same file through a ./relative path must canonicalize to the same doc (#166).
( cd "$DOCS_DIR" && "$HADES_BIN" --db "$DB" ingest "./smoke_note.md" --force >/dev/null 2>&1 )
check "relative path canonicalizes to same doc" "$(count documents)" "$DOCS_1"

# Batch with one missing file: envelope success=false AND exit code 1 (#166).
BATCH_OUT=$("$HADES_BIN" --db "$DB" ingest "$DOCS_DIR/smoke_note.md" "$DOCS_DIR/does_not_exist.md" --batch --force 2>>"$WORK/stderr.log")
BATCH_RC=$?
check "partial batch exits nonzero" "$BATCH_RC" "1"
check "partial batch envelope success=false" "$(echo "$BATCH_OUT" | jq -r '.success')" "false"

# ── 4. Codebase ingest: FK invariants (#165/#171) ───────────────────────
CRATE="$WORK/smoke_crate"; mkdir -p "$CRATE/src"
cat > "$CRATE/Cargo.toml" <<'TOML'
[package]
name = "smoke_crate"
version = "0.1.0"
edition = "2021"
TOML
cat > "$CRATE/src/lib.rs" <<'RS'
/// Greets a caller by name.
pub fn greet(name: &str) -> String {
    format!("hello, {name}")
}

/// Greets the whole world by delegating to `greet`.
pub fn greet_world() -> String {
    greet("world")
}
RS

step "codebase ingest fixture crate" h codebase ingest "$CRATE"
FILES_1=$(count codebase_files); CHUNKS_1=$(count codebase_chunks); EMB_1=$(count codebase_embeddings)
check_min "codebase_files written" "$FILES_1" 1
check_min "codebase_chunks written" "$CHUNKS_1" 1
check_min "codebase_embeddings written" "$EMB_1" 1

# #165/#171: every chunk and embedding must carry its parent foreign keys.
# Chunks carry file_key (their one FK); embeddings carry file_key + chunk_key.
ORPHAN_CHUNKS=$(h db aql "RETURN LENGTH(FOR c IN codebase_chunks FILTER c.file_key == null RETURN 1)" | jq -r '.data.results[0]')
check "no codebase_chunks missing file_key" "$ORPHAN_CHUNKS" "0"
ORPHAN_EMB=$(h db aql "RETURN LENGTH(FOR e IN codebase_embeddings FILTER e.file_key == null OR e.chunk_key == null RETURN 1)" | jq -r '.data.results[0]')
check "no codebase_embeddings missing file/chunk keys" "$ORPHAN_EMB" "0"

# ── 5. --force re-ingest: bind family (#169) + leak (#159) ──────────────
step "codebase ingest --force re-ingest" h codebase ingest "$CRATE" --force
check "force re-ingest: files stable" "$(count codebase_files)" "$FILES_1"
check "force re-ingest: chunks stable (no leak)" "$(count codebase_chunks)" "$CHUNKS_1"
check "force re-ingest: embeddings stable" "$(count codebase_embeddings)" "$EMB_1"

# ── 6. drift → retire → prune → validate (#158/#157) ────────────────────
DRIFT_CLEAN=$(h codebase drift "$CRATE" | jq -r '.data.stale.count')
check "drift clean after ingest" "$DRIFT_CLEAN" "0"

rm "$CRATE/src/lib.rs"
DRIFT_JSON=$(h codebase drift "$CRATE")
DRIFT_MISSING=$(echo "$DRIFT_JSON" | jq -r '.data.stale.count')
check_min "drift reports deleted file" "$DRIFT_MISSING" 1
# Targets are always explicit: feed drift's stale keys straight into retire —
# the exact drift→retire loop the commands were designed to close (#158).
echo "$DRIFT_JSON" | jq -r '.data.stale.keys[]' > "$WORK/stale_keys"
step "retire drift's stale keys" h codebase retire --from "$WORK/stale_keys" --yes
check "retire removed the file node" "$(count codebase_files)" "0"
step "prune-orphans" h codebase prune-orphans
check "prune left no orphan chunks" "$(count codebase_chunks)" "0"
step "codebase validate" h codebase validate

# ── 7. Semantic search over the doc pipeline ────────────────────────────
h db create-index -c embeddings --dimension 2048 >/dev/null 2>>"$WORK/stderr.log" || true
HITS=$(h db query "friendly greeting subsystem" | jq -r '.data.results // .data.matches // [] | length')
check_min "semantic search returns results" "$HITS" 1

# ── 8. Stats sanity ─────────────────────────────────────────────────────
step "codebase stats" h codebase stats
step "db stats" h db stats

# ── Summary ─────────────────────────────────────────────────────────────
echo
echo "── ${PASS} passed, ${FAIL} failed"
if (( FAIL > 0 )); then
    printf '%s\n' "${FAILURES[@]/#/  ✗ }"
    echo "stderr log: $WORK/stderr.log (removed on exit; rerun with 'bash -x' to keep context)"
    exit 1
fi
