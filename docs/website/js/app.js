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
    if (p < 0.25) return "low";
    if (p < 0.5)  return "mid_low";
    if (p < 0.75) return "mid_high";
    return "high";
  }

  function pickRow(filterFn) {
    const candidates = STATE.predictions.filter(r => {
      if (country.value && r.country !== country.value) return false;
      if (bucket.value && bucketOf(r.p_disrupted) !== bucket.value) return false;
      if (service.value && !String(r.service_id).includes(service.value)) return false;
      if (date.value && r.date !== date.value) return false;
      return filterFn(r);
    });
    if (!candidates.length) {
      alert("No matching stop in the 5K-row sample. Try a broader filter.");
      return null;
    }
    return candidates[Math.floor(Math.random() * candidates.length)];
  }

  function show(row) {
    if (!row) return;
    document.getElementById("pred-empty").style.display = "none";
    const res = document.getElementById("pred-result");
    res.classList.remove("hidden");
    res.classList.add("fade-in");
    const probPct = (row.p_disrupted * 100).toFixed(1) + "%";
    document.getElementById("pred-prob").textContent = probPct;
    document.getElementById("pred-fill").style.width = (row.p_disrupted * 100) + "%";
    document.getElementById("d-service").textContent = row.service_id;
    document.getElementById("d-station").textContent = row.station_id;
    document.getElementById("d-date").textContent    = row.date;
    document.getElementById("d-pos").textContent     = `${row.stop_order} (norm ${row.position_norm.toFixed(2)})`;
    document.getElementById("d-class").textContent   = row.train_class_code;
    document.getElementById("d-country").textContent = row.country;
    document.getElementById("d-delay").textContent   = row.delay_min.toFixed(1) + " min";
    const truth = row.y_stop === 1 ? "DISRUPTED" : "ON-TIME";
    const pred  = row.y_pred === 1 ? "DISRUPTED" : "ON-TIME";
    let outcomeTag = "tag";
    if (row.y_stop === row.y_pred) outcomeTag += row.y_stop ? " success" : " primary";
    else outcomeTag += row.y_pred ? " warning" : " danger";
    document.getElementById("d-outcome").innerHTML =
      `<span class="${outcomeTag}">truth ${truth} / pred ${pred}</span>`;

    drawKgBridge(row);
  }

  document.getElementById("pred-random").addEventListener("click", () => show(pickRow(() => true)));
  document.getElementById("pred-tp").addEventListener("click", () => show(
    pickRow(r => r.y_stop === 1 && r.y_pred === 1 && r.p_disrupted > 0.95)
  ));
  document.getElementById("pred-fp").addEventListener("click", () => show(
    pickRow(r => r.y_stop === 0 && r.y_pred === 1 && r.p_disrupted > 0.7)
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

// ── Boot ────────────────────────────────────────────────────────────────
loadAll().catch(err => {
  console.error(err);
  document.querySelector("main").innerHTML =
    `<div class="card" style="border-color:var(--c-danger)"><h3>Failed to load dashboard data</h3><p class="muted">${err.message}</p><p class="muted">Make sure you're serving this folder over HTTP (not <code>file://</code>): try <code>python -m http.server</code> from <code>docs/website/</code>.</p></div>`;
});

// Resize charts on window resize
window.addEventListener("resize", () => {
  Object.values(STATE.charts).forEach(c => c && c.resize && c.resize());
});
