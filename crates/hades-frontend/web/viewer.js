/* hades-viewer — WebGL graph explorer over the /api/graph-data contract.
 *
 * Rendering is sigma.js (WebGL); topology is a graphology MultiDirectedGraph;
 * layout is ForceAtlas2 in a Web Worker. All styling is driven by attributes
 * DISCOVERED from the snapshot (meta.attribute_index) — nothing about any
 * particular graph's domain is hard-coded. That is what makes the viewer work
 * on any HADES graph, and how "grounded vs. asserted" edges become visually
 * distinct without a fixed field name. */

const Graph = window.graphology.MultiDirectedGraph;
const { forceAtlas2, FA2Layout } = window.HadesFA2;

// Categorical palette (colour-blind-aware-ish, high separation on dark bg).
const PALETTE = [
  "#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#e3b341", "#ff7b72",
  "#79c0ff", "#56d364", "#ffa657", "#bc8cff", "#7ee787", "#ffab70",
  "#a5d6ff", "#f0883e", "#d29922", "#db61a2", "#39c5cf", "#8ddb8c",
];
const DIM = "#30363d";
const HILITE = "#f2cc60";

// ---- Live UI/render state read by the sigma reducers. ----
const state = {
  db: null,                // selected database (from the allowlist)
  snapshot: null,
  colorBy: "collection",   // node attribute (or "collection")
  sizeBy: "degree",        // "degree" or a numeric node attribute
  edgeColorBy: "collection", // edge attribute (or "collection") driving persistent edge colour
  edgeAttr: "collection",  // edge attribute for the emphasis focus
  edgeVal: "",             // value to emphasize; "" = emphasize nothing
  hidden: new Set(),       // hidden collections
  selected: null,          // selected node id
  neighbors: new Set(),    // neighbors of the selected node
  colorMap: new Map(),     // node category value -> colour, for current colorBy
  edgeColorMap: new Map(), // edge category value -> colour, for current edgeColorBy
};

let graph = null;
let renderer = null;
let layout = null;

// ---------- deep links ----------
//
// Every view is addressable: ?db=&graph=&node=&expand=1&limit=N. This is the
// agent->human half of the shared-reference channel — an agent can hand back a
// URL that opens this exact database/graph focused on the node it means, and the
// address bar always holds a link to what you're currently looking at.

const deepLink = (() => {
  const p = new URLSearchParams(location.search);
  return {
    db: p.get("db"),
    graph: p.get("graph"),
    node: p.get("node"),
    expand: p.get("expand") === "1",
    limit: p.get("limit"),
  };
})();
// Consumed once on load, so later navigation isn't hijacked by the initial URL.
let pendingLink = { ...deepLink };

/// Build a shareable URL for the current view (optionally focused on a node).
function currentLink(nodeId) {
  const u = new URL(location.origin + location.pathname);
  if (state.db) u.searchParams.set("db", state.db);
  if (state.snapshot) u.searchParams.set("graph", state.snapshot.meta.named_graph);
  const lim = parseInt(document.getElementById("load-limit").value, 10);
  if (Number.isFinite(lim) && lim > 0) u.searchParams.set("limit", lim);
  if (nodeId) {
    u.searchParams.set("node", nodeId);
    u.searchParams.set("expand", "1");
  }
  return u.toString();
}

/// Keep the address bar in sync with the current view (no page reload).
function syncUrl(nodeId) {
  history.replaceState(null, "", currentLink(nodeId));
}

// ---------- data loading ----------

async function loadDatabases() {
  const r = await fetch("/api/databases");
  const { databases = [] } = await r.json();
  const picker = document.getElementById("db-picker");
  picker.innerHTML = databases.map((d) => `<option>${esc(d)}</option>`).join("");
  // Honor ?db= when it names a database this server actually serves.
  const wanted = pendingLink.db && databases.includes(pendingLink.db) ? pendingLink.db : databases[0];
  state.db = wanted || null;
  picker.value = state.db || "";
  await loadGraphList();
}

async function loadGraphList() {
  const r = await fetch("/api/graphs?db=" + encodeURIComponent(state.db));
  const { graphs = [] } = await r.json();
  const picker = document.getElementById("graph-picker");
  picker.innerHTML = graphs.map((g) => `<option>${esc(g)}</option>`).join("");
  if (!graphs.length) { setStats(`no named graphs in ${esc(state.db)}`); return; }
  const wanted = pendingLink.graph && graphs.includes(pendingLink.graph) ? pendingLink.graph : graphs[0];
  picker.value = wanted;
  if (pendingLink.limit) document.getElementById("load-limit").value = pendingLink.limit;
  await loadGraph(wanted);
}

/// After a graph loads, honor ?node= — focus it, and expand if asked. When the
/// node isn't in the loaded slice, pull it in via expansion rather than failing.
async function applyPendingNode() {
  const target = pendingLink.node;
  pendingLink = {}; // one-shot
  if (!target) { syncUrl(); return; }
  if (!graph.hasNode(target)) {
    setOverlay("fetching linked node…");
    await expandNode(target).catch(() => {});
  }
  if (graph.hasNode(target)) {
    if (deepLink.expand) await expandNode(target).catch(() => {});
    focusNode(target);
  } else {
    setOverlay(`linked node not found: ${target}`);
  }
}

/// Center the camera on a node and select it.
function focusNode(id) {
  const dd = renderer.getNodeDisplayData(id);
  if (dd) renderer.getCamera().animate({ x: dd.x, y: dd.y, ratio: 0.3 }, { duration: 400 });
  selectNode(id);
}

async function loadGraph(name) {
  setStats(`loading ${name}…`);
  // Optional server-side cap for large graphs (0/empty = load all).
  const limit = parseInt(document.getElementById("load-limit").value, 10);
  let url = "/api/graph-data?db=" + encodeURIComponent(state.db) + "&graph=" + encodeURIComponent(name);
  if (Number.isFinite(limit) && limit > 0) url += "&limit=" + limit;
  const r = await fetch(url);
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    setStats("error: " + (e.error || r.status));
    return;
  }
  state.snapshot = await r.json();
  buildGraph();
  buildControls();
  render();
  runLayout();
  reportStats();
  await applyPendingNode();
}

// ---------- graph construction ----------

function buildGraph() {
  graph = new Graph();
  const { nodes, edges } = state.snapshot;

  // Seed positions on a circle so FA2 has something to expand from.
  nodes.forEach((n, i) => {
    const a = (2 * Math.PI * i) / nodes.length;
    graph.addNode(n.id, {
      x: Math.cos(a) * 100,
      y: Math.sin(a) * 100,
      size: 3,
      label: n.label,
      collection: n.collection,
      degree: n.degree,
      attrs: n.attrs,
    });
  });

  let skipped = 0;
  for (const e of edges) {
    if (!graph.hasNode(e.source) || !graph.hasNode(e.target)) { skipped++; continue; }
    graph.addEdgeWithKey(e.id, e.source, e.target, {
      collection: e.collection,
      attrs: e.attrs,
      size: 1,
    });
  }
  if (skipped) console.warn(`${skipped} edges skipped (endpoint outside snapshot)`);

  recomputeColors();
  recomputeEdgeColors();
  recomputeSizes();
}

// ---------- attribute helpers ----------

// Flatten a value to a display category string.
function asCategory(v) {
  if (v == null) return "(none)";
  if (Array.isArray(v)) return v.length ? String(v[0]) : "(empty)";
  if (typeof v === "object") return "(object)";
  return String(v);
}

// The category value used for a node under the current colorBy.
function nodeCategory(id) {
  if (state.colorBy === "collection") return graph.getNodeAttribute(id, "collection");
  const attrs = graph.getNodeAttribute(id, "attrs") || {};
  return asCategory(attrs[state.colorBy]);
}

// Edge category under an arbitrary attribute key (or "collection").
function edgeCategoryBy(edge, key) {
  if (key === "collection") return graph.getEdgeAttribute(edge, "collection");
  const attrs = graph.getEdgeAttribute(edge, "attrs") || {};
  return asCategory(attrs[key]);
}
const edgeCategory = (edge) => edgeCategoryBy(edge, state.edgeAttr);        // for emphasis
const edgeColorCategory = (edge) => edgeCategoryBy(edge, state.edgeColorBy); // for colouring

function recomputeColors() {
  const values = new Set();
  graph.forEachNode((id) => values.add(nodeCategory(id)));
  state.colorMap = new Map(
    [...values].sort().map((v, i) => [v, PALETTE[i % PALETTE.length]])
  );
  buildLegend();
}

function recomputeEdgeColors() {
  const values = new Set();
  graph.forEachEdge((edge) => values.add(edgeColorCategory(edge)));
  // Offset into the palette so edge colours don't collide with node colours at a glance.
  state.edgeColorMap = new Map(
    [...values].sort().map((v, i) => [v, PALETTE[(i + 7) % PALETTE.length]])
  );
  buildEdgeLegend();
}

function recomputeSizes() {
  if (state.sizeBy === "degree") {
    graph.forEachNode((id, a) => graph.setNodeAttribute(id, "size", 3 + Math.sqrt(a.degree) * 1.6));
    return;
  }
  // Numeric attribute: min-max normalise across nodes that carry a number.
  let min = Infinity, max = -Infinity;
  graph.forEachNode((id, a) => {
    const v = Number((a.attrs || {})[state.sizeBy]);
    if (Number.isFinite(v)) { min = Math.min(min, v); max = Math.max(max, v); }
  });
  const span = max - min || 1;
  graph.forEachNode((id, a) => {
    const v = Number((a.attrs || {})[state.sizeBy]);
    const t = Number.isFinite(v) ? (v - min) / span : 0;
    graph.setNodeAttribute(id, "size", 3 + t * 12);
  });
}

// ---------- rendering ----------

function render() {
  if (renderer) renderer.kill();
  renderer = new Sigma(graph, document.getElementById("graph"), {
    defaultEdgeColor: DIM,
    renderEdgeLabels: false,
    labelColor: { color: "#c9d1d9" },
    labelDensity: 0.6,
    labelRenderedSizeThreshold: 8,
    zIndex: true,
    nodeReducer,
    edgeReducer,
  });
  wireInteraction();
}

function nodeReducer(id, data) {
  const res = { ...data };
  if (state.hidden.has(data.collection)) { res.hidden = true; return res; }
  res.color = state.colorMap.get(nodeCategory(id)) || "#888";
  if (state.selected) {
    if (id === state.selected) { res.zIndex = 2; res.highlighted = true; }
    else if (state.neighbors.has(id)) { res.zIndex = 1; }
    else { res.color = DIM; res.label = ""; res.zIndex = 0; }
  }
  return res;
}

function edgeReducer(edge, data) {
  const res = { ...data };
  const [s, t] = graph.extremities(edge);
  if (state.hidden.has(graph.getNodeAttribute(s, "collection")) ||
      state.hidden.has(graph.getNodeAttribute(t, "collection"))) {
    res.hidden = true; return res;
  }
  // Selection focus: only edges touching the selected node stay lit.
  if (state.selected && s !== state.selected && t !== state.selected) {
    res.hidden = true; return res;
  }
  // Persistent colour by edge category (collection = relation type by default).
  res.color = state.edgeColorMap.get(edgeColorCategory(edge)) || DIM;
  // Emphasis overlay: matching edges pop, non-matching recede to near-background.
  if (state.edgeVal) {
    if (edgeCategory(edge) === state.edgeVal) {
      res.size = 2.5; res.zIndex = 1;
    } else {
      res.color = "#21262d"; res.size = 0.5;
    }
  }
  return res;
}

// ---------- layout ----------

function runLayout() {
  if (layout) { layout.kill(); layout = null; }
  const settings = forceAtlas2.inferSettings(graph);
  layout = new FA2Layout(graph, { settings });
  layout.start();
  setLayoutButton(true);
  // Auto-settle: stop after a bounded run so the CPU frees itself.
  clearTimeout(runLayout._t);
  runLayout._t = setTimeout(() => stopLayout(), 6000);
}

function stopLayout() { if (layout && layout.isRunning()) { layout.stop(); setLayoutButton(false); } }
function setLayoutButton(running) {
  const b = document.getElementById("layout-toggle");
  b.textContent = running ? "Pause layout" : "Resume layout";
  b.classList.toggle("active", running);
}

// ---------- neighborhood expansion (live/scale) ----------

const EXPAND_LIMIT = 100;

async function expandNode(id) {
  const g = state.snapshot.meta.named_graph;
  setOverlay("expanding…");
  try {
    // A deep link may target a node outside the loaded slice — fetch it first so
    // its edges have an endpoint to attach to.
    if (!graph.hasNode(id)) {
      const nr = await fetch(`/api/node?db=${encodeURIComponent(state.db)}&id=${encodeURIComponent(id)}`);
      if (!nr.ok) { setOverlay("node not found"); return 0; }
      mergeDelta({ nodes: [await nr.json()], edges: [] }, { x: 0, y: 0 });
    }
    const r = await fetch(`/api/expand?db=${encodeURIComponent(state.db)}&graph=${encodeURIComponent(g)}&node=${encodeURIComponent(id)}&limit=${EXPAND_LIMIT}`);
    if (!r.ok) { setOverlay("expand failed"); return 0; }
    const delta = await r.json();
    const anchor = graph.hasNode(id)
      ? { x: graph.getNodeAttribute(id, "x"), y: graph.getNodeAttribute(id, "y") }
      : { x: 0, y: 0 };
    const added = mergeDelta(delta, anchor);
    setOverlay(added ? `+${added} nodes` : "no new nodes");
    if (added) runLayout();
    return added;
  } catch (_) {
    setOverlay("expand error");
    return 0;
  }
}

// Merge an expansion delta into the live graph; returns the count of new nodes.
// New nodes are seeded near their anchor so the layout grows outward from it.
function mergeDelta(delta, anchor) {
  let added = 0;
  for (const n of delta.nodes) {
    if (graph.hasNode(n.id)) continue;
    graph.addNode(n.id, {
      x: anchor.x + (Math.random() - 0.5) * 40,
      y: anchor.y + (Math.random() - 0.5) * 40,
      size: 3,
      label: n.label,
      collection: n.collection,
      degree: 0,
      attrs: n.attrs,
    });
    added++;
  }
  for (const e of delta.edges) {
    if (graph.hasNode(e.source) && graph.hasNode(e.target) && !graph.hasEdge(e.id)) {
      graph.addEdgeWithKey(e.id, e.source, e.target, { collection: e.collection, attrs: e.attrs, size: 1 });
    }
  }
  if (added || delta.edges.length) {
    recomputeDegrees();
    recomputeColors();
    recomputeEdgeColors();
    recomputeSizes();
    refreshLiveStats();
    renderer.refresh();
  }
  return added;
}

// Recompute degree from the live (possibly merged) graph.
function recomputeDegrees() {
  graph.forEachNode((id) => graph.setNodeAttribute(id, "degree", graph.degree(id)));
}

// After a merge the payload no longer matches the snapshot counts; report live.
function refreshLiveStats() {
  const m = state.snapshot.meta;
  setStats(
    `<b>${esc(m.named_graph)}</b> @ ${esc(m.database)}<br>` +
    `<b>${graph.order}</b> nodes · <b>${graph.size}</b> edges (live)` +
    (m.partial ? ' · <span id="partial">seeded from a partial slice</span>' : "")
  );
}

// ---------- shared reference: send-to-agent + deep links ----------
//
// The human->agent half of the channel. Right-click a node to copy a structured
// briefing an agent can act on: identity, attributes, connections, a deep link
// back into this view, and the exact `hades` commands to investigate further.
// Clipboard-based on purpose — it works with ANY agent and needs no server state.

// Total characters a briefing may occupy. Spent where the information is,
// rather than as fixed per-field caps that cut mid-structure.
const PAYLOAD_BUDGET = 12000;
// Above this many connections, list them grouped by relation instead of flat.
const CONN_GROUP_THRESHOLD = 60;
const CONN_SAMPLES_PER_GROUP = 8;

/// Render one attribute value within `budget` chars.
///
/// Key rule: when a value doesn't fit, degrade to a STRUCTURAL SUMMARY rather
/// than slicing the bytes. A JSON array cut mid-token is actively misleading —
/// it looks like data but isn't parseable and hides how much is missing.
/// "[13 objects, 8421 chars — fields: id, name, statement, …]" tells the reader
/// exactly what's there and that it must be fetched to be read.
function briefValue(v, budget) {
  if (v === null || v === undefined) return String(v);
  if (Array.isArray(v) && v.length > 12 && v.every((x) => typeof x === "number")) {
    return `[${v.length} numbers — vector]`;
  }
  if (typeof v === "string") {
    if (v.length <= budget) return v;
    // Cut on a line/word boundary when one is reasonably close to the limit.
    const head = v.slice(0, budget);
    const brk = Math.max(head.lastIndexOf("\n"), head.lastIndexOf(" "));
    const at = brk > budget * 0.6 ? brk : budget;
    return v.slice(0, at) + ` …[+${v.length - at} chars — fetch with the CLI command below]`;
  }
  if (typeof v !== "object") return String(v);

  const json = JSON.stringify(v, null, 1);
  if (json.length <= budget) return json;
  if (Array.isArray(v)) {
    const keys = new Set();
    for (const item of v) {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        Object.keys(item).forEach((k) => keys.add(k));
      }
    }
    return keys.size
      ? `[${v.length} objects, ${json.length} chars — fields: ${[...keys].join(", ")}]`
      : `[${v.length} items, ${json.length} chars]`;
  }
  const keys = Object.keys(v);
  return `{${keys.length} fields, ${json.length} chars — keys: ${keys.join(", ")}}`;
}

/// Max-min fair allocation: each item first gets what it needs up to an even
/// share, and whatever the small items don't use flows to the large ones. Keeps
/// a few huge attributes from starving all the small readable ones.
function allocateBudgets(sizes, total) {
  const out = new Array(sizes.length).fill(0);
  let remaining = Math.max(0, total);
  let left = sizes.length;
  const order = sizes.map((len, i) => ({ len, i })).sort((a, b) => a.len - b.len);
  for (const { len, i } of order) {
    const share = Math.floor(remaining / Math.max(1, left));
    const give = Math.min(len, share);
    out[i] = give;
    remaining -= give;
    left--;
  }
  return out;
}

/// Collect a node's connections from the live graph.
function collectConnections(id) {
  const conns = [];
  graph.forEachEdge(id, (edge, ea, source, target) => {
    const outbound = source === id;
    const other = outbound ? target : source;
    conns.push({
      rel: ea.collection,
      dir: outbound ? "->" : "<-",
      other,
      label: graph.hasNode(other) ? graph.getNodeAttribute(other, "label") : other,
    });
  });
  return conns;
}

/// Render connections. Flat when few; grouped by relation when many, so a
/// high-degree node still communicates its *shape* (which relations, how many)
/// instead of an arbitrary first-N slice.
function renderConnections(conns) {
  const line = (c) => `- \`${c.rel}\` ${c.dir} \`${c.other}\` — ${c.label}`;
  if (conns.length <= CONN_GROUP_THRESHOLD) {
    return [`### Connections (${conns.length})`, ...conns.map(line)];
  }
  const groups = new Map();
  for (const c of conns) {
    const k = `${c.rel} ${c.dir}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(c);
  }
  const out = [
    `### Connections (${conns.length}) — grouped by relation, ` +
      `up to ${CONN_SAMPLES_PER_GROUP} shown per group`,
  ];
  for (const [k, items] of [...groups].sort((a, b) => b[1].length - a[1].length)) {
    out.push("", `**${k}** (${items.length})`);
    for (const c of items.slice(0, CONN_SAMPLES_PER_GROUP)) {
      out.push(`- \`${c.other}\` — ${c.label}`);
    }
    if (items.length > CONN_SAMPLES_PER_GROUP) {
      out.push(`- …${items.length - CONN_SAMPLES_PER_GROUP} more in this relation`);
    }
  }
  return out;
}

/// Build the agent-facing briefing for a node (markdown), inside PAYLOAD_BUDGET.
function agentPayload(id) {
  const a = graph.getNodeAttributes(id);
  const db = state.db;
  const g = state.snapshot.meta.named_graph;
  const [collection, ...rest] = id.split("/");
  const key = rest.join("/");

  const header = [
    `## HADES node: ${a.label}`,
    "",
    `- **database**: \`${db}\``,
    `- **graph**: \`${g}\``,
    `- **_id**: \`${id}\``,
    `- **collection**: \`${collection}\``,
    `- **degree (in view)**: ${a.degree}`,
  ];
  const footer = [
    "",
    "### Open this view",
    currentLink(id),
    "",
    "### Inspect from the CLI",
    "```bash",
    `hades --db ${db} db get ${collection} ${key}`,
    `hades --db ${db} db graph neighbors --graph ${g} -- ${id}`,
    "```",
  ];
  const connBlock = renderConnections(collectConnections(id));

  // Whatever the fixed sections don't use is available to the attributes.
  const fixed = [...header, ...connBlock, ...footer].join("\n").length;
  const attrs = Object.entries(a.attrs || {});
  const natural = attrs.map(([k, v]) =>
    (typeof v === "object" && v !== null ? JSON.stringify(v, null, 1) : String(v)).length + k.length + 8
  );
  const budgets = allocateBudgets(natural, PAYLOAD_BUDGET - fixed - 200);

  const attrBlock = [];
  let elided = 0;
  if (attrs.length) {
    attrBlock.push("", "### Attributes");
    attrs.forEach(([k, v], i) => {
      const rendered = briefValue(v, Math.max(80, budgets[i]));
      if (natural[i] > budgets[i]) elided++;
      attrBlock.push(`- **${k}**: ${rendered}`);
    });
    if (elided) {
      attrBlock.push(
        "",
        `> ${elided} attribute(s) summarized to fit. Run the \`db get\` command below for full values.`
      );
    }
  }
  return [...header, ...attrBlock, ...connBlock, ...footer].join("\n");
}

/// Copy text, degrading gracefully: the Clipboard API needs a secure context,
/// which plain-HTTP LAN access is NOT — so fall back to execCommand, then to a
/// modal the user can copy from by hand.
async function copyText(text, what) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      setOverlay(`${what} copied`);
      return;
    }
  } catch (_) { /* fall through */ }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "position:fixed;top:0;left:0;opacity:0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    if (ok) { setOverlay(`${what} copied`); return; }
  } catch (_) { /* fall through */ }
  showCopyModal(text, what);
}

function showCopyModal(text, what) {
  document.getElementById("modal-title").textContent = `${what} — select and copy (Ctrl+C)`;
  const ta = document.getElementById("modal-text");
  ta.value = text;
  document.getElementById("copy-modal").hidden = false;
  ta.focus();
  ta.select();
}

// ---------- context menu ----------

let ctxTarget = null;

function showContextMenu(x, y, nodeId) {
  ctxTarget = nodeId;
  const m = document.getElementById("ctxmenu");
  m.hidden = false;
  // Keep the menu inside the viewport.
  m.style.left = Math.min(x, window.innerWidth - m.offsetWidth - 8) + "px";
  m.style.top = Math.min(y, window.innerHeight - m.offsetHeight - 8) + "px";
}

function hideContextMenu() {
  document.getElementById("ctxmenu").hidden = true;
  ctxTarget = null;
}

function wireContextMenu() {
  document.getElementById("ctxmenu").onclick = async (e) => {
    const act = e.target.dataset && e.target.dataset.act;
    if (!act || !ctxTarget) return;
    const id = ctxTarget;
    hideContextMenu();
    if (act === "agent") await copyText(agentPayload(id), "Node briefing");
    else if (act === "link") await copyText(currentLink(id), "Deep link");
    else if (act === "expand") await expandNode(id);
    else if (act === "focus") focusNode(id);
    else if (act === "source") {
      selectNode(id);
      const btn = document.getElementById("load-source");
      if (btn) btn.click();
    }
  };
  document.addEventListener("click", hideContextMenu);
  document.getElementById("modal-close").onclick = () => {
    document.getElementById("copy-modal").hidden = true;
  };
  document.getElementById("share-view").onclick = () => copyText(currentLink(state.selected), "View link");
}

// ---------- interaction: drag, click-to-inspect, neighbour focus ----------

function wireInteraction() {
  let dragging = null;
  renderer.on("downNode", ({ node }) => {
    dragging = node;
    graph.setNodeAttribute(node, "highlighted", true);
    renderer.getCamera().disable();
  });
  renderer.getMouseCaptor().on("mousemovebody", (e) => {
    if (!dragging) return;
    const p = renderer.viewportToGraph(e);
    graph.setNodeAttribute(dragging, "x", p.x);
    graph.setNodeAttribute(dragging, "y", p.y);
    e.preventSigmaDefault();
  });
  const drop = () => {
    if (dragging) graph.removeNodeAttribute(dragging, "highlighted");
    dragging = null;
    renderer.getCamera().enable();
  };
  renderer.getMouseCaptor().on("mouseup", drop);

  renderer.on("clickNode", ({ node }) => selectNode(node));
  renderer.on("clickEdge", ({ edge }) => inspectEdge(edge));
  renderer.on("clickStage", () => selectNode(null));
  // Double-click a node to pull its real neighborhood from the backend and
  // merge it in — the interactive path for graphs too large to load whole.
  renderer.on("doubleClickNode", (e) => {
    if (e.preventSigmaDefault) e.preventSigmaDefault(); // suppress the default zoom
    expandNode(e.node);
  });
  // Right-click a node -> shared-reference menu (send to agent, copy link, …).
  renderer.on("rightClickNode", (e) => {
    if (e.event && e.event.original) e.event.original.preventDefault();
    if (e.preventSigmaDefault) e.preventSigmaDefault();
    selectNode(e.node);
    const ev = e.event && e.event.original;
    showContextMenu(ev ? ev.clientX : 100, ev ? ev.clientY : 100, e.node);
  });
  // Suppress the browser menu over the canvas so ours is the only one.
  renderer.getContainer().addEventListener("contextmenu", (ev) => ev.preventDefault());
}

function selectNode(id) {
  state.selected = id;
  state.neighbors = new Set();
  if (id) {
    graph.forEachNeighbor(id, (n) => state.neighbors.add(n));
    inspectNode(id);
  } else {
    setInspector('<div class="empty">Click a node or edge.</div>');
  }
  syncUrl(id); // the address bar always holds a link to what you're looking at
  renderer.refresh();
}

// ---------- inspector ----------

function inspectNode(id) {
  const a = graph.getNodeAttributes(id);
  rows(`${a.collection} · degree ${a.degree}`, a.label, id, a.attrs);
  // Source text isn't stored on file/symbol nodes — it lives in codebase_chunks
  // keyed by file_key. Offer to fetch it when we can derive that key.
  const fk = fileKeyFor(id, a);
  if (fk) appendSourceAffordance(fk);
}
function inspectEdge(edge) {
  const a = graph.getEdgeAttributes(edge);
  const [s, t] = graph.extremities(edge);
  rows(a.collection, `${graph.getNodeAttribute(s, "label")} → ${graph.getNodeAttribute(t, "label")}`, edge, a.attrs);
}
function rows(kind, title, id, attrs) {
  let html = `<div class="kv"><div class="k">${esc(kind)}</div><div class="v"><b>${esc(title)}</b></div></div>`;
  html += `<div class="kv"><div class="k">_id</div><div class="v">${esc(id)}</div></div>`;
  for (const [k, v] of Object.entries(attrs || {})) {
    html += `<div class="kv"><div class="k">${esc(k)}</div><div class="v">${esc(fmtValue(v))}</div></div>`;
  }
  setInspector(html);
}
// Collapse embedding / large numeric arrays so they don't drown out readable fields.
function fmtValue(v) {
  if (Array.isArray(v) && v.length > 12 && v.every((x) => typeof x === "number")) {
    return `[${v.length} numbers — vector]`;
  }
  return typeof v === "object" && v !== null ? JSON.stringify(v, null, 1) : String(v);
}

// The file_key a node's source is keyed under: an explicit attr, or (for file
// nodes) the node's own key. Null when the node has no associated source.
function fileKeyFor(id, a) {
  if (a.attrs && a.attrs.file_key) return a.attrs.file_key;
  if (a.collection === "codebase_files") return id.split("/").slice(1).join("/");
  return null;
}

function appendSourceAffordance(fileKey) {
  const div = document.createElement("div");
  div.innerHTML = `<button id="load-source">Load source</button><pre id="source-view" class="source"></pre>`;
  document.getElementById("inspect-body").appendChild(div);
  document.getElementById("load-source").onclick = () => loadSource(fileKey);
}

async function loadSource(fileKey) {
  const pre = document.getElementById("source-view");
  pre.textContent = "loading…";
  try {
    const r = await fetch(`/api/content?db=${encodeURIComponent(state.db)}&file_key=${encodeURIComponent(fileKey)}`);
    if (!r.ok) { pre.textContent = "(failed to load source)"; return; }
    const { chunks = [] } = await r.json();
    // textContent (not innerHTML) — source text is untrusted and must not render as HTML.
    pre.textContent = chunks.length ? chunks.map((c) => c.text).join("") : "(no source text for this node)";
  } catch (_) {
    pre.textContent = "(failed to load source)";
  }
}

// ---------- controls ----------

function buildControls() {
  const m = state.snapshot.meta;
  // Node attribute keys = union across node collections.
  const nodeKeys = unionKeys(m.attribute_index, m.node_collections);
  const edgeKeys = unionKeys(m.attribute_index, m.edge_collections);

  fillSelect("color-by", ["collection", ...nodeKeys], state.colorBy);
  fillSelect("size-by", ["degree", ...numericKeys(nodeKeys)], state.sizeBy);
  fillSelect("edge-color-by", ["collection", ...edgeKeys], state.edgeColorBy);
  fillSelect("edge-attr", ["collection", ...edgeKeys], state.edgeAttr);
  refreshEdgeValues();

  // Collection filter toggles.
  const filters = document.getElementById("filters");
  const all = [...m.node_collections];
  filters.innerHTML = all.map((c) =>
    `<label><input type="checkbox" data-col="${esc(c)}" checked>
       <span class="swatch" style="background:${state.colorMap.get(c) || "#888"}"></span>${esc(c)}</label>`
  ).join("");
  filters.querySelectorAll("input").forEach((cb) =>
    cb.addEventListener("change", () => {
      if (cb.checked) state.hidden.delete(cb.dataset.col);
      else state.hidden.add(cb.dataset.col);
      renderer.refresh();
    }));

  onChange("db-picker", (v) => { state.db = v; loadGraphList(); });
  onChange("graph-picker", (v) => loadGraph(v));
  onChange("color-by", (v) => { state.colorBy = v; recomputeColors(); syncFilterSwatches(); renderer.refresh(); });
  onChange("size-by", (v) => { state.sizeBy = v; recomputeSizes(); renderer.refresh(); });
  onChange("edge-color-by", (v) => { state.edgeColorBy = v; recomputeEdgeColors(); renderer.refresh(); });
  onChange("edge-attr", (v) => { state.edgeAttr = v; refreshEdgeValues(); renderer.refresh(); });
  onChange("edge-val", (v) => { state.edgeVal = v; renderer.refresh(); });
  document.getElementById("reload").onclick = () =>
    loadGraph(document.getElementById("graph-picker").value);
  document.getElementById("layout-toggle").onclick = () => {
    if (layout && layout.isRunning()) stopLayout();
    else runLayout();
  };
}

function refreshEdgeValues() {
  const vals = new Set(["" /* none */]);
  graph.forEachEdge((edge) => vals.add(edgeCategory(edge)));
  const sel = document.getElementById("edge-val");
  sel.innerHTML = [...vals].map((v) =>
    `<option value="${esc(v)}">${v === "" ? "— emphasize none —" : esc(v)}</option>`).join("");
  sel.value = state.edgeVal;
}

function buildLegend() {
  const el = document.getElementById("legend");
  if (!el) return;
  el.innerHTML = [...state.colorMap].map(([v, c]) =>
    `<div class="legend-row"><span class="swatch" style="background:${c}"></span>${esc(v)}</div>`
  ).join("");
}

function buildEdgeLegend() {
  const el = document.getElementById("edge-legend");
  if (!el) return;
  el.innerHTML = [...state.edgeColorMap].map(([v, c]) =>
    `<div class="legend-row"><span class="swatch" style="background:${c};border-radius:0;height:3px"></span>${esc(v)}</div>`
  ).join("");
}

function syncFilterSwatches() {
  document.querySelectorAll("#filters .swatch").forEach((s) => {
    const col = s.parentElement.querySelector("input").dataset.col;
    s.style.background = state.colorMap.get(col) || "#888";
  });
}

// ---------- small helpers ----------

function unionKeys(index, collections) {
  const s = new Set();
  for (const c of collections) (index[c] || []).forEach((k) => s.add(k));
  return [...s].sort();
}
function numericKeys(keys) {
  // Keep keys that are numeric on at least one node.
  const numeric = new Set();
  graph.forEachNode((id, a) => {
    for (const k of keys) if (Number.isFinite(Number((a.attrs || {})[k]))) numeric.add(k);
  });
  return keys.filter((k) => numeric.has(k));
}
function fillSelect(id, options, current) {
  const el = document.getElementById(id);
  el.innerHTML = options.map((o) => `<option${o === current ? " selected" : ""}>${esc(o)}</option>`).join("");
}
// Assign (not addEventListener) so re-running buildControls per graph load
// replaces the handler instead of stacking duplicates.
function onChange(id, fn) { document.getElementById(id).onchange = (e) => fn(e.target.value); }
function setStats(t) { document.getElementById("stats").innerHTML = t; }
function reportStats() {
  const m = state.snapshot.meta;
  setStats(
    `<b>${esc(m.named_graph)}</b> @ ${esc(m.database)}<br>` +
    `<b>${m.node_count}</b>/${m.node_total} nodes · <b>${m.edge_count}</b>/${m.edge_total} edges` +
    (m.partial ? ' <span id="partial">(partial slice)</span>' : "")
  );
}
function setInspector(html) { document.getElementById("inspect-body").innerHTML = html; }
function setOverlay(t) { const el = document.getElementById("overlay"); if (el) el.textContent = t; }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

wireContextMenu();
loadDatabases();
