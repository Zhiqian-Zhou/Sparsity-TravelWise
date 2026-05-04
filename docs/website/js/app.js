/* TravelWise dashboard — vanilla ES module */

const STATE = {
  benchmark: null,
  xai: null,
  causes: null,
  stations: null,
  predictions: null,
  kg: null,
  scenario: "A",
  metric: "test_pr_auc",
  charts: {},
  cy: { schema: null, sample: null },
  map: null,
  mapCluster: null,
  mapCountry: "",
  theme: localStorage.getItem("travelwise-theme") || "auto",
};

const COLORS = {
  logreg:    "#90A4AE",
  lgbm:      "#1E88E5",
  xgb:       "#FFA000",
  graphsage: "#43A047",
  bilstm:    "#E91E63",
};

const COUNTRY_COLOR = {
  IT: "#43A047",   // Italian green
  FI: "#1E88E5",   // Finnish blue
  NL: "#FFA000",   // Dutch orange
};

// ── Theme toggle ────────────────────────────────────────────────────────
function applyTheme(t) {
  if (t === "auto") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.setAttribute("data-theme", t);
  }
  // Re-render charts so axis colours pick up the new palette
  Object.values(STATE.charts).forEach(c => c && c.dispose && c.dispose());
  STATE.charts = {};
  if (STATE.benchmark) {
    drawBenchmarkBars();
    drawBreakdowns();
    drawShap();
    drawCauseCharts();
  }
  if (STATE.causePlayReady) drawCausePlay();
}
document.getElementById("theme-toggle").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme") || "light";
  const next = cur === "dark" ? "light" : "dark";
  STATE.theme = next;
  localStorage.setItem("travelwise-theme", next);
  applyTheme(next);
});
applyTheme(STATE.theme === "auto" ? null : STATE.theme);

// ── Data load (parallel) ───────────────────────────────────────────────
async function loadAll() {
  const fetches = [
    fetch("data/benchmark.json").then(r => r.json()),
    fetch("data/xai.json").then(r => r.json()),
    fetch("data/causes.json").then(r => r.json()),
    fetch("data/stations.json").then(r => r.json()),
    fetch("data/predictions_sample.json").then(r => r.json()),
    fetch("data/kg_sample.json").then(r => r.json()),
  ];
  const meta = fetch("data/stations_meta.json").then(r => r.ok ? r.json() : null).catch(() => null);
  const [b, x, c, s, p, kg, m] = await Promise.all([...fetches, meta]);
  STATE.benchmark = b; STATE.xai = x; STATE.causes = c;
  STATE.stations = s;  STATE.predictions = p; STATE.kg = kg;
  STATE.stationsMeta = m || {};
  hydrateHero();
  drawBenchmarkBars();
  drawBreakdowns();
  drawShap();
  drawCauseCharts();
  drawMap();
  wirePredict();
}

// ── Hero stats ─────────────────────────────────────────────────────────
function hydrateHero() {
  const b = STATE.benchmark;
  document.getElementById("stat-total").textContent = (16586834 / 1e6).toFixed(1) + " M";
  document.getElementById("stat-test").textContent  = (b.splits.test / 1e6).toFixed(2) + " M";
  const xgbB = b.models.find(m => m.model === "xgb" && m.scenario === "B");
  document.getElementById("stat-best").textContent  = xgbB.test_pr_auc.toFixed(3);
  document.getElementById("stat-cause").textContent = "+46%";
  document.getElementById("stat-kg").textContent    = "16.75 M";
}

// ── Benchmark bar charts (A and B) ─────────────────────────────────────
function fmtMetric(n, metric) {
  if (metric === "test_ece" || metric === "test_brier") return n.toFixed(3);
  return n.toFixed(3);
}

function drawBenchmarkBars() {
  for (const sc of ["A", "B"]) {
    const el = document.getElementById(`chart-bench-${sc}`);
    if (!el) continue;
    const chart = echarts.init(el, null, { renderer: "canvas" });
    STATE.charts[`bench-${sc}`] = chart;

    const rows = STATE.benchmark.models.filter(m => m.scenario === sc);
    rows.sort((a, b) => b[STATE.metric] - a[STATE.metric]);
    const colors = rows.map(r => COLORS[r.model] || "#888");

    chart.setOption({
      title: { text: `Scenario ${sc}: ${labelFor(STATE.metric)}`,
                left: "center", textStyle: { fontSize: 14, fontWeight: 600 } },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
                 valueFormatter: v => fmtMetric(v, STATE.metric) },
      grid: { left: 80, right: 30, top: 50, bottom: 30 },
      xAxis: { type: "value", min: 0, max: 1, axisLabel: { formatter: v => v.toFixed(2) } },
      yAxis: { type: "category", data: rows.map(r => r.model),
               axisLabel: { fontFamily: "JetBrains Mono", fontSize: 13 } },
      series: [{
        type: "bar",
        data: rows.map((r, i) => ({ value: r[STATE.metric],
                                      itemStyle: { color: colors[i], borderRadius: [0, 6, 6, 0] } })),
        label: { show: true, position: "right",
                  formatter: ({ value }) => fmtMetric(value, STATE.metric),
                  fontFamily: "JetBrains Mono", fontSize: 12 },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: "rgba(0,0,0,0.2)" } },
      }],
    });
  }
}
function labelFor(metric) {
  return ({
    test_pr_auc: "PR-AUC", test_f1: "F1",
    test_precision: "Precision", test_recall: "Recall",
    test_ece: "ECE (lower = better)", test_brier: "Brier",
  })[metric] || metric;
}
document.getElementById("metric-tabs").addEventListener("click", (e) => {
  if (!e.target.dataset.metric) return;
  document.querySelectorAll("#metric-tabs button").forEach(b => b.classList.remove("active"));
  e.target.classList.add("active");
  STATE.metric = e.target.dataset.metric;
  drawBenchmarkBars();
});

// ── Breakdown charts (per-country, per-position, per-class) ────────────
function drawBreakdowns() {
  drawBreakdown("chart-by-country", STATE.benchmark.by_country, "PR-AUC by country (XGB / B)");
  drawBreakdown("chart-by-position", STATE.benchmark.by_position, "PR-AUC by stop position");
  drawBreakdown("chart-by-class", STATE.benchmark.by_train_class, "PR-AUC by train class");
}
function drawBreakdown(id, data, title) {
  const el = document.getElementById(id);
  if (!el) return;
  const chart = echarts.init(el);
  STATE.charts[id] = chart;
  const keys = Object.keys(data);
  const vals = keys.map(k => data[k].pr_auc);
  const ns   = keys.map(k => data[k].n);
  chart.setOption({
    title: { text: title, left: "center", textStyle: { fontSize: 13, fontWeight: 600 } },
    tooltip: { trigger: "axis",
                formatter: p => {
                  const i = p[0].dataIndex;
                  return `<b>${keys[i]}</b><br/>PR-AUC: ${vals[i].toFixed(3)}<br/>n: ${ns[i].toLocaleString()}`;
                } },
    grid: { left: 50, right: 20, top: 40, bottom: 40 },
    xAxis: { type: "category", data: keys, axisLabel: { fontSize: 11, rotate: keys.length > 5 ? 30 : 0 } },
    yAxis: { type: "value", min: 0, max: 1, axisLabel: { formatter: v => v.toFixed(1) } },
    series: [{ type: "bar", data: vals,
                 itemStyle: { color: "#1E88E5", borderRadius: [6, 6, 0, 0] },
                 label: { show: true, position: "top",
                           formatter: ({ value }) => value.toFixed(2),
                           fontFamily: "JetBrains Mono", fontSize: 11 } }],
  });
}

// ── SHAP top-features ──────────────────────────────────────────────────
function drawShap() {
  const el = document.getElementById("chart-shap");
  if (!el) return;
  const chart = echarts.init(el);
  STATE.charts.shap = chart;
  const top = STATE.xai.scenarios[STATE.scenario].top_features.slice(0, 18).reverse();
  chart.setOption({
    title: {
      text: `Top features by mean |SHAP| — Scenario ${STATE.scenario} (winner: ${STATE.xai.scenarios[STATE.scenario].model})`,
      left: "center", textStyle: { fontSize: 14, fontWeight: 600 },
    },
    grid: { left: 200, right: 80, top: 50, bottom: 30 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
                valueFormatter: v => v.toFixed(4) },
    xAxis: { type: "value", axisLabel: { fontFamily: "JetBrains Mono" } },
    yAxis: { type: "category", data: top.map(f => f.name),
             axisLabel: { fontFamily: "JetBrains Mono", fontSize: 12 } },
    series: [{
      type: "bar",
      data: top.map(f => f.value),
      itemStyle: {
        color: { type: "linear", x: 0, y: 0, x2: 1, y2: 0,
                 colorStops: [{offset:0, color:"#7c3aed"}, {offset:1, color:"#3b82f6"}] },
        borderRadius: [0, 6, 6, 0],
      },
      label: { show: true, position: "right",
                formatter: ({ value }) => value.toFixed(3),
                fontFamily: "JetBrains Mono", fontSize: 11 },
    }],
  });
}
document.getElementById("shap-tabs").addEventListener("click", (e) => {
  if (!e.target.dataset.scenario) return;
  document.querySelectorAll("#shap-tabs button").forEach(b => b.classList.remove("active"));
  e.target.classList.add("active");
  STATE.scenario = e.target.dataset.scenario;
  drawShap();
});

// ── Cause prediction v1 vs v2 ──────────────────────────────────────────
function drawCauseCharts() {
  drawCausePerClass();
  drawCauseTransfer();
}

function drawCausePerClass() {
  const el = document.getElementById("chart-cause-perclass");
  const chart = echarts.init(el);
  STATE.charts.causePerClass = chart;
  const classes = STATE.causes.classes;
  const v1 = classes.map(c => STATE.causes.v1.models.rf.test_per_class_f1[c] || 0);
  const v2 = classes.map(c => STATE.causes.v2.models.lgbm_v2.test_per_class_f1[c] || 0);
  chart.setOption({
    title: { text: "Per-class F1 — v1 RF vs v2 LightGBM",
              left: "center", textStyle: { fontSize: 14, fontWeight: 600 } },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
                valueFormatter: v => v.toFixed(3) },
    legend: { data: ["v1 (RF)", "v2 (lgbm)"], top: 30 },
    grid: { left: 50, right: 30, top: 70, bottom: 80 },
    xAxis: { type: "category", data: classes, axisLabel: { rotate: 35, fontSize: 11 } },
    yAxis: { type: "value", min: 0, max: 0.6, axisLabel: { fontFamily: "JetBrains Mono" } },
    series: [
      { name: "v1 (RF)", type: "bar", data: v1, itemStyle: { color: "#90A4AE", borderRadius: [4,4,0,0] } },
      { name: "v2 (lgbm)", type: "bar", data: v2, itemStyle: { color: "#1E88E5", borderRadius: [4,4,0,0] } },
    ],
  });
}

function drawCauseTransfer() {
  const el = document.getElementById("chart-cause-transfer");
  const chart = echarts.init(el);
  STATE.charts.causeTransfer = chart;
  const classes = STATE.causes.classes;
  const it_v2 = classes.map(c => STATE.causes.v2.transfer.IT.distribution[c] || 0);
  const fi_v2 = classes.map(c => STATE.causes.v2.transfer.FI.distribution[c] || 0);
  chart.setOption({
    title: { text: "Cause distribution transfer (v2)",
              left: "center", textStyle: { fontSize: 14, fontWeight: 600 } },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
                valueFormatter: v => (v * 100).toFixed(1) + "%" },
    legend: { data: ["🇮🇹 Italy", "🇫🇮 Finland"], top: 30 },
    grid: { left: 50, right: 30, top: 70, bottom: 80 },
    xAxis: { type: "category", data: classes, axisLabel: { rotate: 35, fontSize: 11 } },
    yAxis: { type: "value", max: 0.7,
             axisLabel: { formatter: v => Math.round(v * 100) + "%", fontFamily: "JetBrains Mono" } },
    series: [
      { name: "🇮🇹 Italy", type: "bar", data: it_v2,
        itemStyle: { color: COUNTRY_COLOR.IT, borderRadius: [4,4,0,0] } },
      { name: "🇫🇮 Finland", type: "bar", data: fi_v2,
        itemStyle: { color: COUNTRY_COLOR.FI, borderRadius: [4,4,0,0] } },
    ],
  });
}

// ── Map ────────────────────────────────────────────────────────────────
function drawMap() {
  if (STATE.map) { STATE.map.remove(); STATE.map = null; }
  const isDark = document.documentElement.getAttribute("data-theme") === "dark"
                  || (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const map = L.map("map", { preferCanvas: true, worldCopyJump: false })
              .setView([52, 10], 4);
  STATE.map = map;
  const tiles = isDark
    ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
    : "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";
  L.tileLayer(tiles, {
    attribution: "© OpenStreetMap · © CARTO",
    maxZoom: 14,
  }).addTo(map);

  // Show ALL stations (not down-sampled) but use marker clustering so the
  // map stays readable at low zoom. At zoom 8+, individual markers appear.
  const cluster = L.markerClusterGroup({
    showCoverageOnHover: false,
    maxClusterRadius: 50,
    spiderfyOnMaxZoom: true,
    chunkedLoading: true,
  });
  STATE.mapCluster = cluster;

  applyMapFilter();
  map.addLayer(cluster);
  // Auto-fit to show all markers
  setTimeout(() => {
    const all = STATE.stations.filter(s => !STATE.mapCountry || s.country === STATE.mapCountry);
    if (all.length) {
      const lats = all.map(s => s.lat), lons = all.map(s => s.lon);
      map.fitBounds([[Math.min(...lats), Math.min(...lons)],
                     [Math.max(...lats), Math.max(...lons)]],
                    { padding: [40, 40] });
    }
  }, 100);
}

function applyMapFilter() {
  if (!STATE.mapCluster) return;
  STATE.mapCluster.clearLayers();
  const filtered = STATE.stations.filter(s =>
    !STATE.mapCountry || s.country === STATE.mapCountry
  );
  const markers = filtered.map(s => {
    const radius = 3 + Math.log2(1 + (s.degree || 1)) * 1.6;
    const t = Math.min(1, (s.delay || 0) / 8);
    const isIntl = s.domestic === false;
    // International cross-border stops get a distinct purple ring + dashed border
    const ringColor = isIntl ? "#A78BFA" : (COUNTRY_COLOR[s.country] || "#888888");
    const fillColor = COUNTRY_COLOR[s.country] || "#888888";
    return L.circleMarker([s.lat, s.lon], {
      radius:      isIntl ? Math.max(3, radius * 0.85) : radius,
      color:       ringColor,
      fillColor,
      fillOpacity: 0.35 + 0.55 * t,
      opacity:     isIntl ? 0.85 : 1,
      weight:      isIntl ? 2.5 : 1.5,
      dashArray:   isIntl ? "4 3" : null,
    }).bindTooltip(
      `<b>${s.id}</b><br/>${flagFor(s.country)} ${s.country}${isIntl ? " · 🌐 cross-border" : ""}<br/>degree <code>${s.degree}</code> · avg delay <code>${s.delay} min</code>`,
      { direction: "top", offset: [0, -6] }
    ).bindPopup(
      `<div style="font-family:sans-serif;line-height:1.5"><b>${s.id}</b>
        ${isIntl ? '<br/><span style="background:#a78bfa22;color:#7c3aed;padding:1px 6px;border-radius:3px;font-size:11px">cross-border</span>' : ""}
        <br/>Country: ${flagFor(s.country)} ${s.country}<br/>
        Coords: <code>${s.lat.toFixed(4)}, ${s.lon.toFixed(4)}</code><br/>
        Degree: <code>${s.degree}</code><br/>
        Avg historical delay: <code>${s.delay} min</code></div>`
    );
  });
  STATE.mapCluster.addLayers(markers);

  // Disclaimer banner: explain Finland (no source coords) and NL international
  updateMapDisclaimer();
}

function updateMapDisclaimer() {
  const el = document.getElementById("map-disclaimer");
  if (!el) return;
  const meta = STATE.stationsMeta || {};
  const missing = meta.missing_coords_per_country || {};
  const intl    = meta.international_per_country || {};
  let msgs = [];
  if (STATE.mapCountry === "FI" || (!STATE.mapCountry && (missing.FI || 0) > 0)) {
    msgs.push(`<span class="em">🇫🇮 Finland:</span> ${missing.FI || 0} of ${(missing.FI || 0) + 0} stations have <strong>no coordinates in the FMI source</strong> (the FI-TW dataset ships station codes without lat/lon). They exist in the KG but cannot be plotted geographically.`);
  }
  if (!STATE.mapCountry && (intl.NL || 0) > 0) {
    msgs.push(`<span class="em">🇳🇱 Netherlands:</span> ${intl.NL} stations lie outside the NL bounding box — these are <strong>genuine cross-border destinations</strong> served by NS (Brussels, Berlin, Vienna, Frankfurt, …). They're rendered with a dashed purple ring.`);
  }
  if (STATE.mapCountry === "NL" && (intl.NL || 0) > 0) {
    msgs.push(`<span class="em">🇳🇱 Netherlands:</span> ${intl.NL} of these are international cross-border destinations served by NS (dashed purple rings). The other ${(meta.domestic_per_country?.NL || 0)} are domestic.`);
  }
  if (msgs.length) {
    el.hidden = false;
    el.innerHTML = msgs.join("<br>");
  } else {
    el.hidden = true;
  }
}

function flagFor(c) { return c === "IT" ? "🇮🇹" : c === "FI" ? "🇫🇮" : c === "NL" ? "🇳🇱" : "🌍"; }

// Country filter wiring
document.getElementById("map-country-tabs").addEventListener("click", (e) => {
  if (e.target.dataset.country === undefined) return;
  document.querySelectorAll("#map-country-tabs button").forEach(b => b.classList.remove("active"));
  e.target.classList.add("active");
  STATE.mapCountry = e.target.dataset.country;
  applyMapFilter();
  // Re-fit
  const visible = STATE.stations.filter(s => !STATE.mapCountry || s.country === STATE.mapCountry);
  if (visible.length && STATE.map) {
    const lats = visible.map(s => s.lat), lons = visible.map(s => s.lon);
    STATE.map.fitBounds([[Math.min(...lats), Math.min(...lons)],
                          [Math.max(...lats), Math.max(...lons)]],
                          { padding: [40, 40] });
  }
});

// ── KG schema graph (the metadata-level KG) ────────────────────────────
function drawKgSchema() {
  const container = document.getElementById("kg-schema");
  if (STATE.cy.schema) STATE.cy.schema.destroy();
  const sch = STATE.kg.schema;

  const elements = [];
  for (const n of sch.nodes) {
    elements.push({ data: {
      id: n.label, label: n.label, kind: "schema-node",
      color: n.color, shape: n.shape, fields: n.fields,
      total: sch.totals[n.label],
    }});
  }
  for (const e of sch.edges) {
    elements.push({ data: {
      id: e.label, label: e.label,
      source: e.from, target: e.to, kind: "schema-edge",
      fields: e.fields, total: sch.totals[e.label],
    }});
  }

  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const cy = cytoscape({
    container, elements,
    style: [
      { selector: "node", style: {
        "label": "data(label)",
        "background-color": "data(color)",
        "shape": "data(shape)",
        "color": isDark ? "#f3f4f6" : "#1f2937",
        "text-valign": "center", "text-halign": "center",
        "font-family": "Inter, sans-serif", "font-size": 13, "font-weight": 600,
        "text-outline-width": 2,
        "text-outline-color": isDark ? "#0f172a" : "#ffffff",
        "border-width": 2,
        "border-color": "data(color)",
        "width": 90, "height": 60,
      }},
      { selector: "edge", style: {
        "label": "data(label)",
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "#888",
        "line-color": "#888",
        "width": 2,
        "font-size": 11,
        "color": isDark ? "#d1d5db" : "#6b7280",
        "text-background-color": isDark ? "#0f172a" : "#ffffff",
        "text-background-opacity": 0.85, "text-background-padding": 2,
        "text-rotation": "autorotate",
      }},
      { selector: ":selected", style: { "border-width": 4, "border-color": "#7c3aed" }},
    ],
    layout: { name: "cose", animate: true, idealEdgeLength: 140, padding: 30 },
    minZoom: 0.4, maxZoom: 2.5,
    wheelSensitivity: 0.2,
  });
  STATE.cy.schema = cy;

  const det = document.getElementById("kg-schema-detail");
  cy.on("tap", "node", (evt) => {
    const d = evt.target.data();
    det.innerHTML = `<span class="kg-pill ${d.label.toLowerCase()}">${d.label}</span> ` +
      `<strong>${d.total?.toLocaleString() ?? "?"}</strong> instances` +
      `<ul>${d.fields.map(f => `<li><code>${f}</code></li>`).join("")}</ul>`;
  });
  cy.on("tap", "edge", (evt) => {
    const d = evt.target.data();
    det.innerHTML = `<span class="kg-pill">${d.label}</span> ` +
      `<strong>${d.total?.toLocaleString() ?? "?"}</strong> instances` +
      (d.fields.length
        ? `<ul>${d.fields.map(f => `<li><code>${f}</code></li>`).join("")}</ul>`
        : `<div class="muted" style="margin-top:4px">No properties beyond endpoints.</div>`);
  });
}

// ── KG sample subgraph (real instances) ────────────────────────────────
function drawKgSample(layoutName = "cose") {
  const container = document.getElementById("kg-sample");
  if (STATE.cy.sample) STATE.cy.sample.destroy();
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  const KIND_COLOR = {
    "Station":     "#1E88E5",
    "TrainService":"#43A047",
    "FaultEvent":  "#E53935",
  };
  const KIND_SHAPE = {
    "Station":     "ellipse",
    "TrainService":"round-rectangle",
    "FaultEvent":  "diamond",
  };
  const elements = [];
  for (const n of STATE.kg.sample.nodes) {
    elements.push({ data: {
      id: n.id,
      label: shortLabel(n),
      kind: n.kind,
      color: KIND_COLOR[n.kind] || "#888",
      shape: KIND_SHAPE[n.kind] || "ellipse",
      meta: n,
      isHub: !!n.is_hub,
    }});
  }
  for (const e of STATE.kg.sample.edges) {
    elements.push({ data: {
      id: `${e.source}->${e.target}-${e.kind}`,
      source: e.source, target: e.target,
      kind: e.kind, label: e.kind,
    }});
  }
  const cy = cytoscape({
    container, elements,
    style: [
      { selector: "node", style: {
        "label": "data(label)",
        "background-color": "data(color)",
        "shape": "data(shape)",
        "color": isDark ? "#f3f4f6" : "#1f2937",
        "text-valign": "center", "text-halign": "center",
        "font-family": "JetBrains Mono, monospace", "font-size": 9,
        "text-outline-width": 2,
        "text-outline-color": isDark ? "#0f172a" : "#ffffff",
        "border-width": 1, "border-color": "data(color)",
        "width": 28, "height": 28,
      }},
      { selector: "node[?isHub]", style: {
        "width": 44, "height": 44, "border-width": 3, "font-size": 10,
        "border-color": "#7c3aed",
      }},
      { selector: "node[kind='TrainService']", style: { "width": 32, "height": 18 }},
      { selector: "node[kind='FaultEvent']",   style: { "width": 24, "height": 24 }},
      { selector: "edge", style: {
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "#aaa",
        "line-color": "#bbb",
        "width": 1,
        "opacity": 0.55,
      }},
      { selector: "edge[kind='STOPS_AT']",    style: { "line-color": "#43A047", "target-arrow-color": "#43A047" }},
      { selector: "edge[kind='REPORTED_AT']", style: { "line-color": "#E53935", "target-arrow-color": "#E53935" }},
      { selector: ":selected", style: { "border-width": 4, "border-color": "#7c3aed", "opacity": 1 }},
    ],
    layout: layoutFor(layoutName),
    minZoom: 0.2, maxZoom: 3,
    wheelSensitivity: 0.2,
  });
  STATE.cy.sample = cy;

  const det = document.getElementById("kg-sample-detail");
  cy.on("tap", "node", (evt) => {
    const d = evt.target.data();
    const m = d.meta || {};
    let body = "";
    if (d.kind === "Station") {
      body = `<ul>` +
        `<li>country: <code>${m.country}</code></li>` +
        `<li>lat/lon: <code>${m.lat}, ${m.lon}</code></li>` +
        `<li>degree: <code>${m.degree}</code></li>` +
        `<li>avg historical delay: <code>${m.delay} min</code></li>` +
        (m.is_hub ? `<li>🌟 hub station (selected as seed)</li>` : "") +
        `</ul>`;
    } else if (d.kind === "TrainService") {
      body = `<ul><li>service_id: <code>${m.id}</code></li><li>country: <code>${m.country}</code></li></ul>`;
    } else if (d.kind === "FaultEvent") {
      body = `<ul><li>fault_id: <code>${m.id}</code></li><li>date: <code>${m.date}</code></li><li>description: <code>${m.description}</code></li></ul>`;
    }
    det.innerHTML = `<span class="kg-pill ${d.kind.toLowerCase()}">${d.kind}</span> <code>${m.id || d.label}</code>${body}`;
  });
  cy.on("tap", "edge", (evt) => {
    const d = evt.target.data();
    det.innerHTML = `<span class="kg-pill">${d.kind}</span> <code>${d.source}</code> → <code>${d.target}</code>`;
  });
}

function shortLabel(n) {
  if (n.kind === "Station") return n.id.replace(/^[A-Z]{2}_/, "");
  if (n.kind === "TrainService") return "🚆";
  if (n.kind === "FaultEvent")  return "⚠";
  return n.id;
}
function layoutFor(name) {
  if (name === "concentric") return { name: "concentric", animate: true,
    concentric: n => n.data("isHub") ? 100 : 1, levelWidth: () => 1 };
  if (name === "grid") return { name: "grid", animate: true, padding: 20 };
  return { name: "cose", animate: true, idealEdgeLength: 90, padding: 30,
            nodeRepulsion: 6000, edgeElasticity: 100 };
}
// KG inline graphs were moved to kg.html; only wire if the tabs are present.
const kgLayoutTabs = document.getElementById("kg-layout-tabs");
if (kgLayoutTabs) {
  kgLayoutTabs.addEventListener("click", (e) => {
    if (!e.target.dataset.layout) return;
    document.querySelectorAll("#kg-layout-tabs button").forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    drawKgSample(e.target.dataset.layout);
  });
}

// ── Prediction panel ───────────────────────────────────────────────────
function wirePredict() {
  const country = document.getElementById("pred-country");
  const bucket  = document.getElementById("pred-bucket");
  const service = document.getElementById("pred-service");
  const date    = document.getElementById("pred-date");

  function bucketOf(p) {
    if (p == null) return "";
    if (p < 0.25) return "low";
    if (p < 0.5)  return "mid_low";
    if (p < 0.75) return "mid_high";
    return "high";
  }

  function matchingRows(extraFilter) {
    return STATE.predictions.filter(r => {
      if (country.value && r.country !== country.value) return false;
      if (bucket.value && bucketOf(r.p_b) !== bucket.value) return false;
      if (service.value && !String(r.service_id).includes(service.value)) return false;
      if (date.value && r.date !== date.value) return false;
      return extraFilter ? extraFilter(r) : true;
    });
  }

  function pickRow(extraFilter, errorMsg) {
    const cs = matchingRows(extraFilter);
    if (!cs.length) {
      flashCounter("No matching stop. Try a broader filter.");
      return null;
    }
    return cs[Math.floor(Math.random() * cs.length)];
  }

  function flashCounter(msg) {
    const el = document.getElementById("pred-counter");
    el.textContent = msg;
    el.style.color = "var(--c-warning)";
    setTimeout(() => el.style.color = "", 1200);
  }

  // Update the "N matches" counter live as filters change
  function updateCounter() {
    const n = matchingRows().length;
    document.getElementById("pred-counter").textContent = `${n.toLocaleString()} matching rows in 5K sample`;
  }

  function show(row) {
    if (!row) return;
    document.getElementById("pred-empty").style.display = "none";
    document.getElementById("pred-result").classList.remove("hidden");
    document.getElementById("pred-drivers-card").style.display = "block";

    // Animate dual-scenario probabilities
    animateProb("pred-prob-a", "pred-fill-a", row.p_a ?? 0);
    animateProb("pred-prob-b", "pred-fill-b", row.p_b ?? 0);

    // Scenario flip banner
    const flip = document.getElementById("pred-flip-banner");
    if (row.p_a != null && row.p_b != null) {
      const dPct = ((row.p_b - row.p_a) * 100);
      if (Math.abs(dPct) > 30) {
        flip.hidden = false;
        const dir = dPct > 0 ? "↑" : "↓";
        const phrase = dPct > 0
          ? `Inflight signal <strong>raised</strong> the probability — the model just learned a prior stop was running late.`
          : `Inflight signal <strong>ruled out</strong> the suspicion from A — prior stops were on time.`;
        flip.innerHTML = `<strong>${dir} ${Math.abs(dPct).toFixed(1)} pp shift A→B.</strong> ${phrase}`;
      } else {
        flip.hidden = true;
      }
    }

    const fmt = (v, n=2) => (v == null ? "—" : Number(v).toFixed(n));
    document.getElementById("d-service").textContent = row.service_id;
    document.getElementById("d-station").textContent = row.station_id;
    document.getElementById("d-date").textContent    = row.date;
    document.getElementById("d-pos").textContent     = `${row.stop_order ?? "—"}/${row.n_total_stops ?? "?"} (${(row.position_norm ?? 0).toFixed(2)})`;
    document.getElementById("d-class").textContent   = row.train_class_code;
    document.getElementById("d-country").textContent = `${flagFor(row.country)} ${row.country}`;
    document.getElementById("d-delay").textContent   = `${fmt(row.delay_min, 1)} min`;
    const truth = row.y_stop === 1 ? "DISRUPTED" : "ON-TIME";
    const pred  = row.yp_b === 1 ? "DISRUPTED" : "ON-TIME";
    let outcomeTag = "tag";
    if (row.y_stop === row.yp_b) outcomeTag += row.y_stop ? " success" : " primary";
    else outcomeTag += row.yp_b ? " warning" : " danger";
    document.getElementById("d-outcome").innerHTML =
      `<span class="${outcomeTag}">truth ${truth} / pred ${pred}</span>`;

    drawDrivers(row);
    drawKgBridge(row);
  }

  function animateProb(probId, fillId, p) {
    const probEl = document.getElementById(probId);
    const fillEl = document.getElementById(fillId);
    const target = Math.max(0, Math.min(1, p));
    fillEl.style.width = `${target * 100}%`;
    // Tick the number from 0 to target over ~600ms
    const start = performance.now();
    function frame(t) {
      const e = Math.min(1, (t - start) / 600);
      const eased = 1 - Math.pow(1 - e, 3);          // ease-out cubic
      probEl.textContent = (target * eased * 100).toFixed(1) + "%";
      if (e < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  // Surrogate "drivers" for this row — for each top-importance feature, compare
  // its value to the dataset mean and direction-shade green/red.
  function drawDrivers(row) {
    const card = document.getElementById("pred-drivers-card");
    const wrap = document.getElementById("pred-drivers");
    // Pull top features from the SHAP report (scenario B winning) and compute
    // per-row divergence from sample mean.
    const topB = STATE.xai.scenarios.B.top_features.slice(0, 8);
    const featDist = computeFeatureDist();
    const html = topB.map(f => {
      const v = row[f.name];
      if (v == null || isNaN(v)) return "";
      const stats = featDist[f.name];
      if (!stats) return "";
      const z = (v - stats.mean) / (stats.std || 1);
      const direction = (f.name.match(/lag|delay|severity|cum_|max_|prev_|degree|n_active/i)) ? +1 : -1;
      const sign = z * direction > 0 ? "up" : "down";
      const widthPct = Math.min(48, Math.abs(z) * 14);
      return `
        <div class="driver-row">
          <div class="driver-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</div>
          <div class="driver-bar-track">
            <div class="driver-bar-fill ${sign}" style="width:${widthPct}%"></div>
          </div>
          <div class="driver-val">${Number(v).toFixed(2)}</div>
        </div>`;
    }).filter(Boolean).join("");
    wrap.innerHTML = html || `<p class="muted">No feature snapshot for this row.</p>`;
    card.style.display = "block";
  }

  // Compute mean+std for each feature on the sample (cached)
  let _featDistCache = null;
  function computeFeatureDist() {
    if (_featDistCache) return _featDistCache;
    const featNames = STATE.xai.scenarios.B.top_features.map(f => f.name);
    const out = {};
    for (const f of featNames) {
      const vals = STATE.predictions.map(r => r[f]).filter(v => v != null && !isNaN(v));
      if (!vals.length) continue;
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const sq = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
      out[f] = { mean, std: Math.sqrt(sq) || 1 };
    }
    _featDistCache = out;
    return out;
  }

  // Wire filter inputs to live counter + auto-pick
  ["change", "input"].forEach(ev => {
    [country, bucket, service, date].forEach(el => el.addEventListener(ev, updateCounter));
  });
  updateCounter();

  document.getElementById("pred-random").addEventListener("click", () => show(pickRow()));
  document.getElementById("pred-tp").addEventListener("click", () => show(
    pickRow(r => r.y_stop === 1 && r.yp_b === 1 && r.p_b > 0.95)
  ));
  document.getElementById("pred-fp").addEventListener("click", () => show(
    pickRow(r => r.y_stop === 0 && r.yp_b === 1 && r.p_b > 0.7)
  ));
  document.getElementById("pred-fn").addEventListener("click", () => show(
    pickRow(r => r.y_stop === 1 && r.yp_b === 0 && r.delay_min > 30)
  ));
  document.getElementById("pred-flip").addEventListener("click", () => show(
    pickRow(r => Math.abs((r.p_b ?? 0) - (r.p_a ?? 0)) > 0.4)
  ));
}

// ── KG bridge mock (since we can't run Kuzu from the browser) ───────────
function drawKgBridge(row) {
  const card = document.getElementById("kg-card");
  card.style.display = "block";
  // Synthesise a plausible bridge result from the data we have
  // (if this were running server-side, we'd query Kuzu here).
  const station = STATE.stations.find(s => s.id === row.station_id);
  const neighbours = STATE.stations
    .filter(s => s.country === row.country && s.id !== row.station_id)
    .slice(0, 5).map(s => s.id);
  const result = {
    "$service_id":     row.service_id,
    "$station_id":     row.station_id,
    "$date":           row.date,
    "country":         row.country,
    "station_avg_delay": station ? station.delay : null,
    "degree":          station ? station.degree : null,
    "this_stop_delay": row.delay_min,
    "n_neighbours":    neighbours.length,
    "sample_neighbours": neighbours,
    "_query_latency_ms": 52,
  };
  let html = "";
  for (const [k, v] of Object.entries(result)) {
    let val;
    if (Array.isArray(v)) val = '[' + v.map(x => `<span class="val">"${x}"</span>`).join(", ") + ']';
    else if (typeof v === "number") val = `<span class="num">${v}</span>`;
    else val = `<span class="val">"${v}"</span>`;
    html += `  <span class="key">${k}</span>: ${val}\n`;
  }
  document.getElementById("kg-output").innerHTML = "{\n" + html + "}";
}

// ── Cause-prediction playground ────────────────────────────────────────
const CAUSE_COLORS = {
  "rolling stock":   "#1E88E5",
  "infrastructure":  "#d97706",
  "external":        "#f472b6",
  "accidents":       "#E53935",
  "logistical":      "#fbbf24",
  "engineering work":"#43A047",
  "staff":           "#a855f7",
  "weather":         "#06B6D4",
  "unknown":         "#9ca3af",
};
const CAUSE_PILL = {
  "rolling stock":   "rolling",
  "infrastructure":  "infra",
  "external":        "external",
  "accidents":       "accident",
  "logistical":      "logistical",
  "engineering work":"engineering",
  "staff":           "staff",
  "weather":         "weather",
  "unknown":         "unknown",
};

const CAUSE_STATE = {
  data: null,
  country: "NL",
  filterWeather: -1,
  filterHour: -1,
  filterPosition: "",
  filterClass: "",
  // Per-country pre-computed labels + per-country LRU cache for filter results.
  // labels[cc][i] = the cause label for row i (avoids recomputing pred_cause_group||cause_group||unknown).
  // positions[cc][i] = the bucket name for row i (origin|early|mid|late|terminus|"").
  labels: {},
  positions: {},
  // Memoized filter -> {indices, counts, total} keyed by `cc|w|h|p|cls`.
  cache: new Map(),
  // Cap memo cache at ~80 keys (3 countries × ~25 distinct filter combos in normal use).
  cacheLimit: 80,
  // Pending requestAnimationFrame handle so rapid slider drags coalesce into one paint.
  rafHandle: null,
  // Bumped on every redraw — used to discard stale renders if user races filters.
  drawTicket: 0,
};

async function loadCausePlay() {
  const wrap = document.getElementById("cause-play-wrap");
  if (wrap) wrap.classList.add("loading");
  // Browser HTTP cache + server cache headers handle the network side.
  // Once fetched, we precompute per-row label and position buckets so country
  // switches and filter changes never have to touch the raw `pred_cause_group`
  // / `position_norm` lookup paths again.
  const data = await fetch("data/cause_play.json", { cache: "force-cache" }).then(r => r.json());
  CAUSE_STATE.data = data;
  _precomputeCauseRows();
  STATE.causePlayReady = true;
  if (wrap) wrap.classList.remove("loading");
  drawCausePlay();
  wireCauseControls();
}

function _precomputeCauseRows() {
  const out_l = {}, out_p = {};
  for (const cc of Object.keys(CAUSE_STATE.data.countries || {})) {
    const rows = CAUSE_STATE.data.countries[cc];
    const labels = new Array(rows.length);
    const positions = new Array(rows.length);
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      labels[i] = r.pred_cause_group || r.cause_group || "unknown";
      if (r.is_origin === 1) positions[i] = "origin";
      else if (r.is_terminus === 1) positions[i] = "terminus";
      else {
        const pn = r.position_norm;
        if (pn == null) positions[i] = "";
        else if (pn < 0.25) positions[i] = "early";
        else if (pn < 0.75) positions[i] = "mid";
        else positions[i] = "late";
      }
    }
    out_l[cc] = labels;
    out_p[cc] = positions;
  }
  CAUSE_STATE.labels = out_l;
  CAUSE_STATE.positions = out_p;
}

function _causeLabel(row) {
  return row.pred_cause_group || row.cause_group || "unknown";
}

function _cacheKey() {
  return [
    CAUSE_STATE.country,
    CAUSE_STATE.filterWeather,
    CAUSE_STATE.filterHour,
    CAUSE_STATE.filterPosition,
    CAUSE_STATE.filterClass,
  ].join("|");
}

function _computeCauseFilter() {
  const key = _cacheKey();
  const memo = CAUSE_STATE.cache.get(key);
  if (memo) return memo;

  const cc = CAUSE_STATE.country;
  const rows = (CAUSE_STATE.data?.countries?.[cc]) || [];
  const positions = CAUSE_STATE.positions[cc];
  const labels = CAUSE_STATE.labels[cc];
  const wf = CAUSE_STATE.filterWeather;
  const hf = CAUSE_STATE.filterHour;
  const pf = CAUSE_STATE.filterPosition;
  const clsf = CAUSE_STATE.filterClass;
  const classes = CAUSE_STATE.data.classes;
  const counts = Object.fromEntries(classes.map(c => [c, 0]));

  const indices = [];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (wf >= 0 && r.weather_severity !== wf) continue;
    if (hf >= 0 && r.scheduled_arrival_hour !== hf) continue;
    if (pf && positions[i] !== pf) continue;
    if (clsf !== "" && String(r.train_class_code) !== String(clsf)) continue;
    indices.push(i);
    const c = labels[i];
    if (counts[c] !== undefined) counts[c]++;
  }

  const result = { indices, counts, total: indices.length };

  // LRU-ish: evict oldest if over the cap.
  if (CAUSE_STATE.cache.size >= CAUSE_STATE.cacheLimit) {
    const firstKey = CAUSE_STATE.cache.keys().next().value;
    if (firstKey) CAUSE_STATE.cache.delete(firstKey);
  }
  CAUSE_STATE.cache.set(key, result);
  return result;
}

function _markCachedTabs() {
  // Visual hint: tab gets a green "•" once that country's first filter is in cache.
  const tabs = document.querySelectorAll("#cause-country-tabs button");
  tabs.forEach(b => {
    const cc = b.dataset.country;
    let hit = false;
    for (const k of CAUSE_STATE.cache.keys()) { if (k.startsWith(cc + "|")) { hit = true; break; } }
    b.classList.toggle("cached", hit);
  });
}

// Coalesce rapid slider events into one render per animation frame.
function drawCausePlay() {
  if (CAUSE_STATE.rafHandle) cancelAnimationFrame(CAUSE_STATE.rafHandle);
  CAUSE_STATE.rafHandle = requestAnimationFrame(_renderCausePlay);
}

function _renderCausePlay() {
  CAUSE_STATE.rafHandle = null;
  const ticket = ++CAUSE_STATE.drawTicket;
  if (!CAUSE_STATE.data) return;
  const wrap = document.getElementById("cause-play-wrap");
  const wasCached = CAUSE_STATE.cache.has(_cacheKey());
  if (!wasCached && wrap) wrap.classList.add("loading");

  // Run heavy compute in a microtask so the loading overlay paints first.
  queueMicrotask(() => {
    if (ticket !== CAUSE_STATE.drawTicket) return;
    const { indices, counts, total } = _computeCauseFilter();
    if (ticket !== CAUSE_STATE.drawTicket) return;

    const cc = CAUSE_STATE.country;
    const rows = CAUSE_STATE.data.countries[cc];
    const labels = CAUSE_STATE.labels[cc];

    document.getElementById("cause-stats").textContent =
      `${total.toLocaleString()} rows match these filters · ${cc} sample size: ${rows.length.toLocaleString()}`;

    const classes = CAUSE_STATE.data.classes;
    const totalSafe = total || 1;
    const series = classes.map(c => ({
      name: c,
      value: counts[c] / totalSafe,
      itemStyle: { color: CAUSE_COLORS[c] || "#888" },
    })).sort((a, b) => b.value - a.value);

    const el = document.getElementById("chart-cause-play");
    if (!STATE.charts.causePlay) STATE.charts.causePlay = echarts.init(el);
    // Pass merge=false (true second arg) only on the first paint — subsequent
    // updates use ECharts' diff path which is much cheaper.
    STATE.charts.causePlay.setOption({
      animationDuration: 200,
      animationDurationUpdate: 200,
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" },
                  valueFormatter: v => (v * 100).toFixed(1) + "%" },
      grid: { left: 130, right: 50, top: 20, bottom: 30 },
      xAxis: { type: "value", max: Math.max(0.05, ...series.map(s => s.value * 1.1)),
                axisLabel: { formatter: v => Math.round(v * 100) + "%" } },
      yAxis: { type: "category", data: series.map(s => s.name),
                axisLabel: { fontSize: 11 } },
      series: [{
        type: "bar", data: series,
        label: { show: true, position: "right",
                  formatter: ({ value }) => (value * 100).toFixed(1) + "%",
                  fontFamily: "JetBrains Mono", fontSize: 11 },
      }],
    });

    // Sample table — render up to 60 rows. Build one big string then innerHTML
    // is faster than 60 separate DOM inserts.
    const sampleN = Math.min(60, indices.length);
    const tbody = document.querySelector("#cause-table tbody");
    const parts = new Array(sampleN);
    for (let i = 0; i < sampleN; i++) {
      const idx = indices[i];
      const r = rows[idx];
      const c = labels[idx];
      const conf = r.pred_max_proba != null ? (r.pred_max_proba * 100).toFixed(0) + "%" : "—";
      parts[i] = `<tr data-idx="${idx}">
        <td>${escapeHtml((r.service_id || "").slice(0, 18))}</td>
        <td>${escapeHtml((r.station_id || "").slice(0, 14))}</td>
        <td>${r.date || "—"}</td>
        <td class="num">${r.delay_min != null ? Number(r.delay_min).toFixed(0) : "—"}</td>
        <td class="num">${r.weather_severity ?? "—"}</td>
        <td><span class="pill ${CAUSE_PILL[c] || 'unknown'}">${c}</span></td>
        <td class="num">${conf}</td>
      </tr>`;
    }
    tbody.innerHTML = parts.join("");
    // One delegated listener instead of 60 individual ones.
    if (!tbody.dataset.bound) {
      tbody.addEventListener("click", (e) => {
        const tr = e.target.closest("tr[data-idx]");
        if (!tr) return;
        tbody.querySelectorAll("tr.selected").forEach(t => t.classList.remove("selected"));
        tr.classList.add("selected");
        const ccNow = CAUSE_STATE.country;
        showCauseRow(CAUSE_STATE.data.countries[ccNow][+tr.dataset.idx]);
      });
      tbody.dataset.bound = "1";
    }

    if (wrap) wrap.classList.remove("loading");
    _markCachedTabs();
  });
}

function showCauseRow(row) {
  const card = document.getElementById("cause-row-detail");
  const c = _causeLabel(row);
  const fields = [
    ["service_id", row.service_id], ["station_id", row.station_id], ["date", row.date],
    ["country", row.country], ["delay_min", row.delay_min],
    ["weather_severity", row.weather_severity], ["temperature", row.temperature],
    ["wind_speed", row.wind_speed], ["snow_depth", row.snow_depth],
    ["precipitation", row.precipitation],
    ["scheduled_arrival_hour", row.scheduled_arrival_hour],
    ["scheduled_arrival_dow", row.scheduled_arrival_dow], ["month", row.month],
    ["train_class_code", row.train_class_code], ["stop_order", row.stop_order],
    ["position_norm", row.position_norm], ["n_total_stops", row.n_total_stops],
    ["avg_historical_delay", row.avg_historical_delay], ["degree", row.degree],
    ["station_lag1_rate", row.station_lag1_rate],
    ["station_lag7_rate", row.station_lag7_rate],
    ["train_station_lag7_rate", row.train_station_lag7_rate],
  ];
  const truth = row.cause_group ? `<div class="ki-row"><div class="ki-key">Truth (NL only)</div><div class="ki-val"><span class="kg-pill ${CAUSE_PILL[row.cause_group] || 'unknown'}">${row.cause_group}</span></div></div>` : "";
  const conf = row.pred_max_proba != null ? `<div class="ki-row"><div class="ki-key">Confidence</div><div class="ki-val mono">${(row.pred_max_proba * 100).toFixed(1)}%</div></div>` : "";
  card.style.display = "block";
  card.querySelector("h3, #cause-row-content")?.remove?.();
  document.getElementById("cause-row-content").innerHTML = `
    <div class="ki-row"><div class="ki-key">Predicted cause</div><div class="ki-val"><span class="kg-pill ${CAUSE_PILL[c] || 'unknown'}">${c}</span></div></div>
    ${conf}${truth}
    <div style="margin-top:12px"><strong>Feature snapshot</strong></div>
    <div class="cause-row-grid" style="margin-top:6px">
      ${fields.filter(([k,v]) => v != null).map(([k,v]) =>
        `<div class="field"><div class="field-label">${escapeHtml(k)}</div>
         <div class="field-value">${typeof v === "number" ? v.toFixed(2) : escapeHtml(String(v))}</div></div>`
      ).join("")}
    </div>`;
}

function wireCauseControls() {
  document.getElementById("cause-country-tabs").addEventListener("click", (e) => {
    if (!e.target.dataset.country) return;
    document.querySelectorAll("#cause-country-tabs button").forEach(b => b.classList.remove("active"));
    e.target.classList.add("active");
    CAUSE_STATE.country = e.target.dataset.country;
    drawCausePlay();
  });
  const wEl = document.getElementById("cause-weather");
  const wLab = document.getElementById("cause-weather-val");
  wEl.value = -1;
  wEl.addEventListener("input", () => {
    const v = +wEl.value;
    CAUSE_STATE.filterWeather = v < 0 ? -1 : v;
    wLab.textContent = v < 0 ? "any" : `severity ${v}`;
    drawCausePlay();
  });
  // The slider min is 0; we use a "reset" by double-clicking
  wEl.addEventListener("dblclick", () => { wEl.value = 0; CAUSE_STATE.filterWeather = -1; wLab.textContent = "any"; drawCausePlay(); });

  const hEl = document.getElementById("cause-hour");
  const hLab = document.getElementById("cause-hour-val");
  hEl.addEventListener("input", () => {
    const v = +hEl.value;
    CAUSE_STATE.filterHour = v;
    hLab.textContent = v < 0 ? "any" : `${String(v).padStart(2,"0")}:00`;
    drawCausePlay();
  });
  hEl.addEventListener("dblclick", () => {
    hEl.value = 0; CAUSE_STATE.filterHour = -1; hLab.textContent = "any"; drawCausePlay();
  });

  document.getElementById("cause-position").addEventListener("change", (e) => {
    CAUSE_STATE.filterPosition = e.target.value;
    drawCausePlay();
  });
  document.getElementById("cause-class").addEventListener("change", (e) => {
    CAUSE_STATE.filterClass = e.target.value;
    drawCausePlay();
  });
}

function escapeHtml(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// ── Boot ────────────────────────────────────────────────────────────────
// Kick off both fetch waves in parallel — the cause-play 3 MB JSON used to
// load only AFTER `loadAll` finished, which delayed the playground's first
// paint by ~the longer of the two waterfalls. Now both run concurrently and
// the playground renders as soon as its own data arrives.
const causePlayBoot = loadCausePlay().catch(err => console.error("cause-play load failed:", err));
loadAll().catch(err => {
  console.error(err);
  document.querySelector("main").innerHTML =
    `<div class="card" style="border-color:var(--c-danger)"><h3>Failed to load dashboard data</h3><p class="muted">${err.message}</p><p class="muted">Make sure you're serving this folder over HTTP (not <code>file://</code>): try <code>python -m http.server</code> from <code>docs/website/</code>.</p></div>`;
});

// Resize charts on window resize
window.addEventListener("resize", () => {
  Object.values(STATE.charts).forEach(c => c && c.resize && c.resize());
});
