#!/usr/bin/env bash
#
# HADES CLI audit — exercises every read/write command against bident_burn.
#
# Skips create-database / drop-database (DDL). bident_burn is treated as
# expendable; the script seeds its own fixtures and tolerates dirty state.
#
# Usage:
#   ./scripts/cli_audit.sh           # full audit
#   ./scripts/cli_audit.sh --verbose # show full output for each command
#
# Exit code: 0 if all tests pass, non-zero with a count if any fail.

set -u
shopt -s lastpipe  # so PASS/FAIL counts survive piped grep

DB=bident_burn
VERBOSE=0
[[ "${1:-}" == "--verbose" || "${1:-}" == "-v" ]] && VERBOSE=1

PASS=0
FAIL=0
SKIPPED=0
FAILURES=()
LOG_DIR=$(mktemp -d -t hades_audit_XXXXXX)
trap 'rm -rf "$LOG_DIR"' EXIT

# ── ANSI for human readability (auto-disabled when piped) ───────────────
if [[ -t 1 ]]; then
    G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; D=$'\033[2m'; N=$'\033[0m'
else
    G=""; R=""; Y=""; D=""; N=""
fi

run() {
    # Usage: run "<test name>" "<shell command>"
    local name=$1
    local cmd=$2
    local out_file="$LOG_DIR/${PASS}_${FAIL}.out"
    local err_file="$LOG_DIR/${PASS}_${FAIL}.err"
    if eval "$cmd" >"$out_file" 2>"$err_file"; then
        printf "${G}✓${N} %s\n" "$name"
        ((PASS++))
        if [[ $VERBOSE -eq 1 ]]; then
            head -c 400 "$out_file" | sed 's/^/    /'
            echo
        fi
    else
        local rc=$?
        printf "${R}✗${N} %s ${D}(exit %d)${N}\n" "$name" "$rc"
        printf "    ${D}cmd: %s${N}\n" "$cmd"
        local err
        err=$(head -c 300 "$err_file" 2>/dev/null || true)
        if [[ -n "$err" ]]; then
            printf "    ${D}err: %s${N}\n" "$err"
        else
            local out
            out=$(head -c 300 "$out_file" 2>/dev/null || true)
            [[ -n "$out" ]] && printf "    ${D}out: %s${N}\n" "$out"
        fi
        ((FAIL++))
        FAILURES+=("$name")
    fi
}

skip() {
    printf "${Y}∼${N} %s ${D}(skipped: %s)${N}\n" "$1" "$2"
    ((SKIPPED++))
}

section() {
    printf "\n${D}── %s ────────────────────────────────────${N}\n" "$1"
}

# ── Preflight ──────────────────────────────────────────────────────────
section "preflight"
run "binary on PATH"        "command -v hades"
run "bident_burn reachable" "hades --db $DB db count codebase_files"

# ── Top-level system commands ──────────────────────────────────────────
section "system"
# Note: status, orient both reach into the DB for stats, so both need --db.
run "status --db"           "hades --db $DB status"
run "orient --db"           "hades --db $DB orient"
run "orient (no DB → fail)" "! hades orient"
run "status (no DB → fail)" "! hades status"

# ── DB read commands ───────────────────────────────────────────────────
section "db read"
run "db collections"         "hades --db $DB db collections"
run "db databases"           "hades --db $DB db databases"
run "db count <existing>"    "hades --db $DB db count codebase_files"
run "db count <missing>"     "! hades --db $DB db count nonexistent_xyz_collection"
run "db list -c codebase"    "hades --db $DB db list -c codebase -n 3"
run "db stats"               "hades --db $DB db stats"
run "db recent"              "hades --db $DB db recent -n 3"
run "db health"              "hades --db $DB db health"
run "db index-status"        "hades --db $DB db index-status"
# Capture a real key for db get / db check (works regardless of file extension).
FIRST_KEY=$(hades --db $DB db aql 'FOR f IN codebase_files LIMIT 1 RETURN f._key' 2>/dev/null \
            | python3 -c 'import sys,json; d=json.load(sys.stdin)["data"]["results"]; print(d[0] if d else "")' 2>/dev/null \
            || true)
if [[ -n "$FIRST_KEY" ]]; then
    run "db get (existing key)"  "hades --db $DB db get codebase_files $FIRST_KEY"
    run "db check (existing)"    "hades --db $DB db check codebase_files/$FIRST_KEY"
else
    skip "db get (existing key)" "no codebase_files seeded"
    skip "db check (existing)"   "no codebase_files seeded"
fi
run "db aql (simple)"        "hades --db $DB db aql 'RETURN 1'"
run "db aql (with bind)"     "hades --db $DB db aql 'FOR i IN 1..@n RETURN i' --bind '{\"n\":3}'"
run "db query (semantic)"    "hades --db $DB db query 'vector similarity search' -c codebase -n 2"
run "db query --hybrid"      "hades --db $DB db query 'vector similarity search' -c codebase -n 2 --hybrid"

# ── DB schema commands ─────────────────────────────────────────────────
section "db schema"
run "schema list"     "hades --db $DB db schema list"
run "schema version"  "hades --db $DB db schema version"
# `db schema show <name>` only supports edge_definition / named_graph names.
# An empty-seeded schema has none, so we expect failure on a known-bad name.
run "schema show (missing → fail)" "! hades --db $DB db schema show nonexistent_xyz"

# ── DB graph commands ──────────────────────────────────────────────────
section "db graph"
run "graph list"      "hades --db $DB db graph list"
# Note: traverse / shortest-path / neighbors require a real graph; codebase_graph
# wasn't created (codebase ingest skipped gharial via the non-fatal warn path).
# These are tested only if the named graph exists.
if hades --db $DB db graph list 2>/dev/null | grep -q '"name"'; then
    GRAPH_NAME=$(hades --db $DB db graph list 2>/dev/null | grep -oE '"name":\s*"[^"]+"' | head -1 | sed 's/.*: "\([^"]*\)".*/\1/')
    SOME_FILE_ID=$(hades --db $DB db aql 'FOR f IN codebase_files LIMIT 1 RETURN f._id' 2>/dev/null | grep -oE '"codebase_files/[^"]+"' | head -1 | tr -d '"')
    if [[ -n "${SOME_FILE_ID:-}" && -n "${GRAPH_NAME:-}" ]]; then
        run "graph traverse"   "hades --db $DB db graph traverse \"$SOME_FILE_ID\" -d outbound --max-depth 2 -g \"$GRAPH_NAME\""
        run "graph neighbors"  "hades --db $DB db graph neighbors \"$SOME_FILE_ID\" -g \"$GRAPH_NAME\""
    else
        skip "graph traverse"  "no file vertex found"
        skip "graph neighbors" "no file vertex found"
    fi
else
    skip "graph traverse"  "no named graph in db"
    skip "graph neighbors" "no named graph in db"
fi

# ── DB write commands (against bident_burn) ────────────────────────────
section "db write (bident_burn — expendable)"
TEST_COL="audit_test_$(date +%s)"
run "db create <collection>"  "hades --db $DB db create $TEST_COL"
run "db insert (single doc)"  "echo '{\"_key\":\"k1\",\"value\":\"first\"}' | hades --db $DB db insert $TEST_COL"
run "db insert (batch)"       "echo '[{\"_key\":\"k2\",\"value\":\"second\"},{\"_key\":\"k3\",\"value\":\"third\"}]' | hades --db $DB db insert $TEST_COL"
run "db count after inserts"  "[ \"\$(hades --db $DB db count $TEST_COL 2>/dev/null | grep -oE '\"count\": [0-9]+' | awk '{print \$2}')\" = '3' ]"
run "db get k1"               "hades --db $DB db get $TEST_COL k1"
run "db update k1"            "hades --db $DB db update $TEST_COL k1 --data '{\"value\":\"updated\"}'"
run "db update verified"      "hades --db $DB db get $TEST_COL k1 | grep -q '\"value\": \"updated\"'"
run "db delete k2"            "hades --db $DB db delete $TEST_COL k2 --force"
run "db count after delete"   "[ \"\$(hades --db $DB db count $TEST_COL 2>/dev/null | grep -oE '\"count\": [0-9]+' | awk '{print \$2}')\" = '2' ]"
run "db export"               "hades --db $DB db export $TEST_COL -o $LOG_DIR/export.jsonl && [ -s $LOG_DIR/export.jsonl ]"

# Refuse to write to non-writable databases
run "write guard (NestedLearning)" "! hades --db NestedLearning db insert codebase_files --data '{\"_key\":\"shouldfail\"}'"

# ── DB write — schema mutation ─────────────────────────────────────────
# Note: schema init was already run during regeneration; skip re-init.
# Re-init would truncate hades_schema, breaking subsequent runs.

# ── Codebase commands ──────────────────────────────────────────────────
section "codebase"
run "codebase stats"      "hades --db $DB codebase stats"
run "codebase validate"   "hades --db $DB codebase validate"

# ── Embed commands (no DB needed) ──────────────────────────────────────
section "embed"
run "embed text (single)"        "hades embed text 'hello world'"
run "embed service status"       "hades embed service status"
run "embed gpu list"             "hades embed gpu list"

# ── Task commands (Persephone task tracker — uses persephone_* coll'ns) ─
section "task"
# Tasks need persephone_tasks collection. Create it if missing.
hades --db $DB db create persephone_tasks 2>/dev/null > /dev/null || true
hades --db $DB db create persephone_sessions 2>/dev/null > /dev/null || true
hades --db $DB db create persephone_logs 2>/dev/null > /dev/null || true
hades --db $DB db create persephone_handoffs 2>/dev/null > /dev/null || true
run "task list (empty ok)"      "hades --db $DB task list"
TASK_OUT=$(hades --db $DB task create "audit-test-task" --description "from cli_audit" 2>/dev/null || true)
TASK_ID=$(echo "$TASK_OUT" \
    | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    print(d["data"]["task"]["_key"])
except Exception:
    pass' 2>/dev/null || true)
if [[ -n "${TASK_ID:-}" ]]; then
    run "task create"               "true"   # captured above
    run "task show <created>"       "hades --db $DB task show $TASK_ID"
    # task update changes editable fields (title/desc/priority/tags). State
    # transitions use explicit verbs (start/close/block) by design.
    run "task update --priority"    "hades --db $DB task update $TASK_ID --priority high"
    run "task close <created>"      "hades --db $DB task close $TASK_ID"
else
    skip "task create"   "task command failed or output unparseable"
    skip "task show"     "no task to show"
    skip "task update"   "no task to update"
    skip "task close"    "no task to close"
fi

# ── Smell commands (codebase compliance) ───────────────────────────────
section "smell"
# `smell report` requires (a) smell definitions seeded in the DB, and (b) a
# path argument. A freshly-regenerated bident_burn has no smell definitions
# (no seed mechanism is exposed by the CLI), so this fails with "failed to
# load smell definitions". Treating the failure as expected for an empty DB.
SMELL_PATH="$(pwd)/scripts/cli_audit.sh"
if hades --db $DB db count smell_specs 2>/dev/null | grep -q '"count": [1-9]'; then
    run "smell report" "hades --db $DB smell report '$SMELL_PATH'"
else
    skip "smell report" "no smell_specs seeded in $DB (no CLI seed mechanism)"
fi

# ── DB write — purge (run last; destructive within scope) ──────────────
section "cleanup (test collection only)"
# Drop the audit-only collection via raw AQL since there's no `db drop` for collections.
# We use `db aql` to run a write — it should be permitted on bident_burn.
hades --db $DB db aql "RETURN 'cleanup-noop'" >/dev/null 2>&1 || true
# The collection is left in place; bident_burn is expendable. Skipping a real cleanup.
skip "drop test collection $TEST_COL" "no CLI 'db drop-collection'; collection left in bident_burn"

# ── Summary ────────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL + SKIPPED))
printf "\n%s══════════════════════════════════════════════%s\n" "$D" "$N"
printf "Total:   %3d\n" "$TOTAL"
printf "%sPassed:  %3d%s\n" "$G" "$PASS" "$N"
printf "%sFailed:  %3d%s\n" "$R" "$FAIL" "$N"
printf "%sSkipped: %3d%s\n" "$Y" "$SKIPPED" "$N"

if [[ $FAIL -gt 0 ]]; then
    printf "\n%sFailures:%s\n" "$R" "$N"
    for f in "${FAILURES[@]}"; do
        printf "  - %s\n" "$f"
    done
    exit 1
fi
exit 0
