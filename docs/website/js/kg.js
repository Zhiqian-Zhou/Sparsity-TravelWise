/* TravelWise KG page — standalone module */

const KIND_COLOR = { Station: "#1E88E5", TrainService: "#43A047", FaultEvent: "#E53935" };
const KIND_SHAPE = { Station: "ellipse", TrainService: "round-rectangle", FaultEvent: "diamond" };
const REL_COLOR  = { STOPS_AT: "#43A047", ADJACENT_TO: "#06B6D4", REPORTED_AT: "#E53935" };
const COUNTRY_FLAG = { IT: "🇮🇹", FI: "🇫🇮", NL: "🇳🇱" };

const STATE = {
  kg: null,
  cy: null,
  cySchema: null,
  filterKinds: new Set(["Station", "TrainService", "FaultEvent"]),
  filterCountry: "",
  layoutName: "fcose",
  dimNonNeighbours: true,
  showLabels: false,
  theme: localStorage.getItem("travelwise-theme") || "auto",
};

// ── Theme toggle (shared with index) ───────────────────────────────────
document.getElementById("theme-toggle").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme") || "light";
  const next = cur === "dark" ? "light" : "dark";
  localStorage.setItem("travelwise-theme", next);
  document.documentElement.setAttribute("data-theme", next);
  // Re-render graphs (canvas adapts)
  if (STATE.cySchema) drawSchema();
  if (STATE.cy)        drawSample();
});
if (STATE.theme && STATE.theme !== "auto") {
  document.documentElement.setAttribute("data-theme", STATE.theme);
}

// ── Register fcose layout extension ────────────────────────────────────
if (window.cytoscapeFcose) cytoscape.use(window.cytoscapeFcose);

// ── Data load ──────────────────────────────────────────────────────────
async function load() {
  const kg = await fetch("data/kg_sample.json").then(r => r.json());
  STATE.kg = kg;
  drawSchema();
  drawSample();
  wireToolbar();
}

// ── Schema graph ───────────────────────────────────────────────────────
function drawSchema() {
  const container = document.getElementById("kg-schema");
  if (STATE.cySchema) STATE.cySchema.destroy();

  const sch = STATE.kg.schema;
  const elements = [];
  for (const n of sch.nodes) {
    elements.push({ data: {
      id: n.label, label: n.label, kind: "schema",
      color: n.color, shape: n.shape, fields: n.fields,
      total: sch.totals[n.label],
    }});
  }
  for (const e of sch.edges) {
    elements.push({ data: {
      id: e.label, label: e.label,
      source: e.from, target: e.to, kind: "rel",
      fields: e.fields, total: sch.totals[e.label],
    }});
  }
  const cy = cytoscape({
    container, elements,
    style: [
      { selector: "node", style: {
        "label": "data(label)",
        "background-color": "data(color)",
        "background-opacity": 0.9,
        "shape": "data(shape)",
        "color": "#f3f4f6",
        "text-valign": "center", "text-halign": "center",
        "font-family": "Inter, sans-serif", "font-size": 14, "font-weight": 700,
        "text-outline-width": 3,
        "text-outline-color": "#0f172a",
        "border-width": 3, "border-color": "data(color)",
        "border-opacity": 1,
        "width": 130, "height": 80,
      }},
      { selector: "edge", style: {
        "label": "data(label)",
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "#94a3b8",
        "line-color": "#94a3b8",
        "width": 2.5,
        "font-family": "JetBrains Mono, monospace",
        "font-size": 10, "font-weight": 600,
        "color": "#cbd5e1",
        "text-background-color": "#0f172a",
        "text-background-opacity": 0.85, "text-background-padding": 3,
        "text-rotation": "autorotate",
      }},
      { selector: "edge[id='ADJACENT_TO']", style: {
        "line-style": "dashed", "line-dash-pattern": [6, 3],
        "line-color": "#06B6D4", "target-arrow-color": "#06B6D4",
      }},
      { selector: ":selected", style: {
        "border-width": 5, "border-color": "#a78bfa",
        "line-color": "#a78bfa", "target-arrow-color": "#a78bfa",
      }},
    ],
    layout: {
      name: window.cytoscapeFcose ? "fcose" : "cose",
      animate: true, animationDuration: 600,
      idealEdgeLength: 180, padding: 60,
      nodeSeparation: 100, randomize: false,
    },
    minZoom: 0.4, maxZoom: 2.5, wheelSensitivity: 0.2,
  });
  STATE.cySchema = cy;

  cy.on("tap", "node", (evt) => renderSchemaInspector(evt.target.data()));
  cy.on("tap", "edge", (evt) => renderSchemaInspector(evt.target.data()));
}

function renderSchemaInspector(d) {
  const el = document.getElementById("kg-schema-detail");
  const isNode = d.kind === "schema";
  const pillClass = isNode ? d.label.toLowerCase() : d.label.toLowerCase();
  const total = d.total ? d.total.toLocaleString() : "?";
  const fieldsHtml = (d.fields && d.fields.length)
    ? `<ul style="margin:8px 0 0 18px">${d.fields.map(f => `<li><code>${escapeHtml(f)}</code></li>`).join("")}</ul>`
    : `<p class="muted" style="margin-top:6px">No additional properties.</p>`;
  el.innerHTML = `
    <h3><span class="kg-pill ${pillClass}">${d.label}</span></h3>
    <div class="ki-row"><div class="ki-key">Type</div><div class="ki-val">${isNode ? "Node table" : "Relationship table"}</div></div>
    <div class="ki-row"><div class="ki-key">Instances in KG</div><div class="ki-val">${total}</div></div>
    <div class="ki-row"><div class="ki-key">Properties</div><div class="ki-val">${fieldsHtml}</div></div>
  `;
}

// ── Sample subgraph ────────────────────────────────────────────────────
function buildElements() {
  const elements = [];
  const visibleIds = new Set();
  for (const n of STATE.kg.sample.nodes) {
    if (!STATE.filterKinds.has(n.kind)) continue;
    if (STATE.filterCountry && n.country && n.country !== STATE.filterCountry) continue;
    elements.push({ data: {
      id: n.id,
      label: shortLabel(n),
      kind: n.kind,
      color: KIND_COLOR[n.kind] || "#888",
      shape: KIND_SHAPE[n.kind] || "ellipse",
      meta: n,
      isHub: !!n.is_hub,
      country: n.country || "",
    }});
    visibleIds.add(n.id);
  }
  for (const e of STATE.kg.sample.edges) {
    if (!visibleIds.has(e.source) || !visibleIds.has(e.target)) continue;
    elements.push({ data: {
      id: `${e.source}->${e.target}-${e.kind}`,
      source: e.source, target: e.target,
      kind: e.kind, label: e.kind,
      color: REL_COLOR[e.kind] || "#94a3b8",
    }});
  }
  return elements;
}

function drawSample() {
  const container = document.getElementById("kg-sample");
  if (STATE.cy) STATE.cy.destroy();

  const elements = buildElements();
  const cy = cytoscape({
    container, elements,
    style: [
      { selector: "node", style: {
        "label": STATE.showLabels ? "data(label)" : "",
        "background-color": "data(color)",
        "background-opacity": 0.92,
        "shape": "data(shape)",
        "color": "#f3f4f6",
        "text-valign": "bottom", "text-halign": "center",
        "text-margin-y": 4,
        "font-family": "JetBrains Mono, monospace",
        "font-size": 9, "font-weight": 500,
        "text-outline-width": 2,
        "text-outline-color": "#0f172a",
        "text-opacity": STATE.showLabels ? 1 : 0,
        "border-width": 1.5, "border-color": "data(color)",
        "border-opacity": 0.9,
        "width": 26, "height": 26,
        "transition-property": "text-opacity, border-width, background-opacity, opacity",
        "transition-duration": 180,
      }},
      { selector: "node[?isHub]", style: {
        "width": 46, "height": 46,
        "border-width": 3, "border-color": "#a78bfa",
        "background-opacity": 1,
        "font-size": 11,
      }},
      { selector: "node[kind='TrainService']", style: { "width": 30, "height": 18 }},
      { selector: "node[kind='FaultEvent']",   style: { "width": 24, "height": 24 }},
      { selector: "node:active, node:selected", style: {
        "border-width": 3, "border-color": "#fbbf24", "text-opacity": 1,
      }},
      { selector: "node.hover-target", style: {
        "border-width": 3, "border-color": "#fbbf24", "text-opacity": 1,
        "z-index": 999,
      }},
      { selector: "node.matched", style: {
        "border-width": 4, "border-color": "#fbbf24",
        "background-blacken": -0.3,
      }},
      { selector: "edge", style: {
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "data(color)",
        "line-color": "data(color)",
        "width": 1.2,
        "opacity": 0.6,
        "transition-property": "opacity, width",
        "transition-duration": 150,
      }},
      { selector: "edge[kind='ADJACENT_TO']", style: {
        "line-style": "dashed", "line-dash-pattern": [4, 3],
        "width": 1, "opacity": 0.55,
      }},
      { selector: "edge[kind='STOPS_AT']", style: { "width": 1.4, "opacity": 0.7 }},
      { selector: "edge.hover-edge", style: { "width": 3, "opacity": 1 }},
      { selector: ".dim", style: { "opacity": 0.08 }},
    ],
    layout: layoutFor(STATE.layoutName, elements.length),
    minZoom: 0.15, maxZoom: 4,
    wheelSensitivity: 0.18,
    boxSelectionEnabled: true,
  });
  STATE.cy = cy;

  // Mini-map: show a static thumbnail by re-running cytoscape on the same data
  // in a small container — this is the lightweight no-extension approach.
  setupMiniMap(cy, elements);

  // Hover-to-emphasise neighbours
  cy.on("mouseover", "node", (evt) => {
    if (!STATE.dimNonNeighbours) return;
    const n = evt.target;
    cy.elements().addClass("dim");
    n.removeClass("dim").addClass("hover-target");
    n.connectedEdges().removeClass("dim").addClass("hover-edge");
    n.neighborhood("node").removeClass("dim");
  });
  cy.on("mouseout", "node", () => {
    cy.elements().removeClass("dim hover-target hover-edge");
  });

  // Click to inspect
  cy.on("tap", "node", (evt) => renderSampleInspector(evt.target.data()));
  cy.on("tap", "edge", (evt) => renderSampleInspector(evt.target.data()));

  // Stats overlay
  document.getElementById("kg-stats").textContent =
    `${cy.nodes().length} nodes · ${cy.edges().length} edges`;
}

function layoutFor(name, n) {
  const animate = n < 200;
  if (name === "fcose" && window.cytoscapeFcose) {
    return {
      name: "fcose", animate, animationDuration: 800, randomize: false,
      idealEdgeLength: 110, nodeSeparation: 80, padding: 40,
      nodeRepulsion: 6000, gravity: 0.4, gravityRange: 3.8,
    };
  }
  if (name === "concentric") return {
    name: "concentric", animate,
    concentric: nd => nd.data("isHub") ? 100 : 1,
    levelWidth: () => 1, padding: 30, minNodeSpacing: 25,
  };
  if (name === "circle") return { name: "circle", animate, padding: 30 };
  if (name === "grid")   return { name: "grid",   animate, padding: 20 };
  return { name: "cose", animate, idealEdgeLength: 100, padding: 30,
           nodeRepulsion: 5000, edgeElasticity: 100 };
}

function setupMiniMap(_, elements) {
  // Lightweight mini-map: render the same graph in a small read-only canvas.
  const mini = document.getElementById("kg-mini");
  mini.style.display = "block";
  mini.innerHTML = "";
  cytoscape({
    container: mini,
    elements,
    style: [
      { selector: "node", style: {
        "background-color": "data(color)", "label": "",
        "width": 6, "height": 6, "border-width": 0,
      }},
      { selector: "node[?isHub]", style: { "width": 9, "height": 9 }},
      { selector: "edge", style: {
        "curve-style": "haystack", "haystack-radius": 0,
        "line-color": "data(color)", "width": 0.4, "opacity": 0.5,
        "target-arrow-shape": "none",
      }},
    ],
    layout: { name: "fcose", animate: false, randomize: false, padding: 6 },
    userZoomingEnabled: false, userPanningEnabled: false,
    boxSelectionEnabled: false, autoungrabify: true,
  });
}

function renderSampleInspector(d) {
  const el = document.getElementById("kg-sample-detail");
  if (d.source && d.target) {
    // Edge
    el.innerHTML = `
      <h3>Relationship</h3>
      <div class="ki-row"><div class="ki-key">Type</div><div class="ki-val"><span class="kg-pill ${d.kind.toLowerCase()}">${d.kind}</span></div></div>
      <div class="ki-row"><div class="ki-key">From</div><div class="ki-val">${d.source}</div></div>
      <div class="ki-row"><div class="ki-key">To</div><div class="ki-val">${d.target}</div></div>
    `;
    return;
  }
  const m = d.meta || {};
  let rows = "";
  if (d.kind === "Station") {
    rows = `
      <div class="ki-row"><div class="ki-key">station_id</div><div class="ki-val">${m.id}</div></div>
      <div class="ki-row"><div class="ki-key">country</div><div class="ki-val">${COUNTRY_FLAG[m.country] || "🌍"} ${m.country}</div></div>
      <div class="ki-row"><div class="ki-key">lat / lon</div><div class="ki-val">${m.lat ?? "—"}, ${m.lon ?? "—"}</div></div>
      <div class="ki-row"><div class="ki-key">degree</div><div class="ki-val">${m.degree ?? "—"}</div></div>
      <div class="ki-row"><div class="ki-key">avg_historical_delay</div><div class="ki-val">${m.delay ?? "—"} min</div></div>
      ${m.is_hub ? `<div class="ki-meta">🌟 hub station (top-12 in country by degree)</div>` : ""}
    `;
  } else if (d.kind === "TrainService") {
    rows = `
      <div class="ki-row"><div class="ki-key">service_id</div><div class="ki-val">${m.id}</div></div>
      <div class="ki-row"><div class="ki-key">country</div><div class="ki-val">${COUNTRY_FLAG[m.country] || "🌍"} ${m.country}</div></div>
      <div class="ki-meta">Sample TrainService visiting one of the hub stations.</div>
    `;
  } else if (d.kind === "FaultEvent") {
    rows = `
      <div class="ki-row"><div class="ki-key">fault_id</div><div class="ki-val">${m.id}</div></div>
      <div class="ki-row"><div class="ki-key">date</div><div class="ki-val">${m.date}</div></div>
      <div class="ki-row"><div class="ki-key">description</div><div class="ki-val">${escapeHtml(m.description)}</div></div>
    `;
  }
  el.innerHTML = `
    <h3><span class="kg-pill ${d.kind.toLowerCase()}">${d.kind}</span> <code>${m.id || d.label}</code></h3>
    ${rows}
    <div class="ki-actions">
      <button class="btn ghost" data-action="centre">Centre</button>
      <button class="btn ghost" data-action="neighbours">Highlight neighbours</button>
    </div>
  `;
  // Wire actions
  el.querySelector('[data-action="centre"]')?.addEventListener("click", () => {
    const n = STATE.cy.getElementById(m.id);
    if (n.length) STATE.cy.animate({ center: { eles: n }, zoom: 1.6 }, { duration: 400 });
  });
  el.querySelector('[data-action="neighbours"]')?.addEventListener("click", () => {
    const n = STATE.cy.getElementById(m.id);
    if (!n.length) return;
    STATE.cy.elements().addClass("dim");
    n.removeClass("dim").addClass("hover-target");
    n.neighborhood().removeClass("dim");
  });
}

// ── Toolbar wiring ─────────────────────────────────────────────────────
function wireToolbar() {
  // Layout switcher
  document.getElementById("kg-layout-tabs").addEventListener("click", (e) => {
    if (!e.target.dataset.layout) return;
    document.querySelectorAll("#kg-layout-tabs button").forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    STATE.layoutName = e.target.dataset.layout;
    drawSample();
  });

  // Country filter
  document.getElementById("kg-country-tabs").addEventListener("click", (e) => {
    if (e.target.dataset.country === undefined) return;
    document.querySelectorAll("#kg-country-tabs button").forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    STATE.filterCountry = e.target.dataset.country;
    drawSample();
  });

  // Kind filter
  document.querySelectorAll('.kg-checkbox input[data-kind]').forEach(cb => {
    cb.addEventListener("change", () => {
      const k = cb.dataset.kind;
      if (cb.checked) STATE.filterKinds.add(k); else STATE.filterKinds.delete(k);
      drawSample();
    });
  });

  // Search
  document.getElementById("kg-search").addEventListener("input", (e) => {
    const q = e.target.value.trim().toLowerCase();
    STATE.cy.nodes().removeClass("matched dim");
    if (!q) return;
    const matches = STATE.cy.nodes().filter(n =>
      String(n.id()).toLowerCase().includes(q) ||
      String(n.data("label") || "").toLowerCase().includes(q)
    );
    if (matches.length === 0) return;
    STATE.cy.nodes().addClass("dim");
    matches.removeClass("dim").addClass("matched");
    matches.neighborhood().removeClass("dim");
    STATE.cy.animate({
      fit: { eles: matches, padding: 80 }
    }, { duration: 400 });
  });

  // Toggles
  document.getElementById("kg-dim").addEventListener("change", (e) => {
    STATE.dimNonNeighbours = e.target.checked;
  });
  document.getElementById("kg-labels").addEventListener("change", (e) => {
    STATE.showLabels = e.target.checked;
    drawSample();
  });

  // Reset / Fit
  document.getElementById("kg-reset").addEventListener("click", () => {
    STATE.filterKinds = new Set(["Station", "TrainService", "FaultEvent"]);
    STATE.filterCountry = "";
    STATE.layoutName = "fcose";
    STATE.showLabels = false;
    document.querySelectorAll('.kg-checkbox input[data-kind]').forEach(cb => cb.checked = true);
    document.getElementById("kg-search").value = "";
    document.getElementById("kg-labels").checked = false;
    document.querySelectorAll("#kg-country-tabs button").forEach(b => b.classList.remove("active"));
    document.querySelector('#kg-country-tabs button[data-country=""]').classList.add("active");
    document.querySelectorAll("#kg-layout-tabs button").forEach(b => b.classList.remove("active"));
    document.querySelector('#kg-layout-tabs button[data-layout="fcose"]').classList.add("active");
    drawSample();
  });
  document.getElementById("kg-fit").addEventListener("click", () => {
    if (STATE.cy) STATE.cy.fit(undefined, 40);
  });
}

// ── Helpers ────────────────────────────────────────────────────────────
function shortLabel(n) {
  if (n.kind === "Station") return n.id.replace(/^[A-Z]{2}_/, "");
  if (n.kind === "TrainService") return "🚆";
  if (n.kind === "FaultEvent")  return "⚠";
  return n.id;
}
function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// ── Resize handler ─────────────────────────────────────────────────────
window.addEventListener("resize", () => {
  if (STATE.cy)        STATE.cy.resize();
  if (STATE.cySchema)  STATE.cySchema.resize();
});

// ── Boot ───────────────────────────────────────────────────────────────
load().catch(err => {
  console.error(err);
  document.querySelector("main").innerHTML =
    `<div class="card" style="padding:30px;border:2px solid var(--c-danger)">
      <h3>Failed to load KG data</h3>
      <p class="muted">${escapeHtml(err.message)}</p>
      <p class="muted">Serve over HTTP (not <code>file://</code>): try <code>python -m http.server</code> from <code>docs/website/</code>.</p>
    </div>`;
});
