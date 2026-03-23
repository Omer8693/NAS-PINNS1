/* demo.js — Interactive demo for NAS-MCO-PINNs
   Handles parameter selection, API calls, and Plotly animations.
*/
"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let state = {
  domain:  "rectangular",
  arch:    "bayesian",
  skip:    "2",
  mode:    "pred",   // "pred" | "fem" | "error"
  window:  0,
  playing: false,
  sliceData: null,
  lossData:  null,
};

let playTimer  = null;
let currentTab = "2d";
let initialized2d  = false;
let initialized3d  = false;
let initializedLoss = false;

// ── Utility ───────────────────────────────────────────────────────────────────
function transpose(mat) {
  // mat[nx][ny] → mat[ny][nx]  (for Plotly: z[row=y][col=x])
  if (!mat || !mat.length) return mat;
  return mat[0].map((_, ci) => mat.map(row => row[ci]));
}

function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }

function setLoading(show) {
  document.getElementById("loading-overlay").style.display = show ? "flex" : "none";
}

function setError(msg) {
  const box = document.getElementById("error-box");
  if (msg) {
    document.getElementById("error-msg").textContent = msg;
    box.style.display = "block";
  } else {
    box.style.display = "none";
  }
}

// ── Parameter Selectors ────────────────────────────────────────────────────────
document.querySelectorAll("#domain-sel .domain-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#domain-sel .domain-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.domain = btn.dataset.val;
    state.window = 0;
    loadData();
  });
});

document.querySelectorAll("#arch-sel .toggle-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#arch-sel .toggle-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.arch = btn.dataset.val;
    state.window = 0;
    loadData();
  });
});

document.querySelectorAll("#skip-sel .toggle-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#skip-sel .toggle-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.skip = btn.dataset.val;
    state.window = 0;
    // Update FEM savings hint
    const pct = state.skip === "2" ? "52%" : "71%";
    document.getElementById("skip-hint").textContent = pct + " FEM savings";
    loadData();
  });
});

document.querySelectorAll("#mode-sel .toggle-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#mode-sel .toggle-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.mode = btn.dataset.val;
    updateCharts();
  });
});

document.getElementById("time-slider").addEventListener("input", e => {
  state.window = parseInt(e.target.value);
  updateCharts();
});

// ── Tab switching ──────────────────────────────────────────────────────────────
function showTab(tab) {
  currentTab = tab;
  ["2d", "3d", "loss"].forEach(t => {
    document.getElementById(`view-${t}`).style.display = t === tab ? "block" : "none";
    document.getElementById(`tab${t}`).classList.toggle("active", t === tab);
  });
  if (state.sliceData) {
    if (tab === "2d"   && !initialized2d)   render2D();
    if (tab === "3d"   && !initialized3d)   render3D();
    if (tab === "loss" && !initializedLoss) renderLoss();
    else if (tab !== "loss") updateCharts();
  }
}

// ── Play / Pause ──────────────────────────────────────────────────────────────
function togglePlay() {
  state.playing = !state.playing;
  document.getElementById("play-icon").className =
    state.playing ? "fa-solid fa-pause" : "fa-solid fa-play";
  if (state.playing) {
    playTimer = setInterval(() => {
      if (!state.sliceData) return;
      const n = Object.keys(state.sliceData.windows).length;
      state.window = (state.window + 1) % n;
      document.getElementById("time-slider").value = state.window;
      updateCharts();
    }, 800);
  } else {
    clearInterval(playTimer);
  }
}

function stepWindow(delta) {
  if (!state.sliceData) return;
  const n = Object.keys(state.sliceData.windows).length;
  state.window = clamp(state.window + delta, 0, n - 1);
  document.getElementById("time-slider").value = state.window;
  updateCharts();
}

// ── Data loading ───────────────────────────────────────────────────────────────
function loadData() {
  stopPlay();
  setLoading(true);
  setError(null);
  initialized2d = initialized3d = initializedLoss = false;

  const url = `/api/slice?domain=${state.domain}&arch=${state.arch}&skip=${state.skip}`;
  const lurl = `/api/loss?domain=${state.domain}&arch=${state.arch}&skip=${state.skip}`;

  Promise.all([
    fetch(url).then(r => r.ok ? r.json() : r.json().then(e => { throw new Error(e.error); })),
    fetch(lurl).then(r => r.ok ? r.json() : null).catch(() => null),
  ]).then(([slice, loss]) => {
    state.sliceData = slice;
    state.lossData  = loss;
    setLoading(false);

    const n = Object.keys(slice.windows).length;
    const slider = document.getElementById("time-slider");
    slider.max   = n - 1;
    slider.value = 0;
    state.window = 0;

    // Stats
    updateStats();

    // Render active tab
    if (currentTab === "2d")   { initialized2d = false;   render2D(); }
    if (currentTab === "3d")   { initialized3d = false;   render3D(); }
    if (currentTab === "loss") { initializedLoss = false; renderLoss(); }

    // Check training status
    checkTrainingStatus();
  }).catch(err => {
    setLoading(false);
    setError(err.message || "Data not yet available. Training may still be running.");
    checkTrainingStatus();
  });
}

function stopPlay() {
  if (state.playing) {
    state.playing = false;
    document.getElementById("play-icon").className = "fa-solid fa-play";
    clearInterval(playTimer);
  }
}

// ── Get current Z matrix ──────────────────────────────────────────────────────
function getCurrentZ(wi) {
  const win = state.sliceData.windows[String(wi)];
  if (!win) return { z: [], zmin: 0, zmax: 1, label: "" };

  let raw, label;
  if (state.mode === "pred") {
    raw = win.T_pred; label = "PINN Prediction [°C]";
  } else if (state.mode === "fem") {
    raw = win.T_fem;  label = "FEM Reference [°C]";
  } else {
    // error = |pred - fem|
    raw = win.T_pred.map((row, i) =>
      row.map((v, j) => Math.abs(v - win.T_fem[i][j]))
    );
    label = "|PINN - FEM| [°C]";
  }

  const flat = raw.flat();
  const zmin = Math.min(...flat);
  const zmax = Math.max(...flat);
  const z    = transpose(raw);  // (ny, nx)

  return { z, zmin, zmax, label };
}

// ── 2D Heatmap ────────────────────────────────────────────────────────────────
function render2D() {
  if (!state.sliceData) return;
  const { z, zmin, zmax, label } = getCurrentZ(state.window);
  const d = state.sliceData;
  const cscale = state.mode === "error" ? "Hot" : "Turbo";

  Plotly.react("plot-2d", [{
    type: "heatmap",
    x: d.xi, y: d.yi, z,
    colorscale: cscale,
    zmin, zmax,
    colorbar: {
      title: { text: "T [°C]", side: "right", font: { color: "#aaa", size: 11 } },
      tickfont: { color: "#aaa", size: 10 },
    },
    hovertemplate: "x=%{x:.3f}m  y=%{y:.3f}m<br>T=%{z:.1f}°C<extra></extra>",
  }], {
    paper_bgcolor: "transparent",
    plot_bgcolor:  "#1a2035",
    font:  { color: "#ccc", family: "Inter" },
    xaxis: { title: "x [m]", gridcolor: "#2a3456", color: "#aaa", zeroline: false },
    yaxis: { title: "y [m]", gridcolor: "#2a3456", color: "#aaa", zeroline: false, scaleanchor: "x" },
    margin: { t: 20, b: 50, l: 60, r: 80 },
    annotations: [{
      text: `z = ${d.z_val.toFixed(3)} m  (mid-slice)`,
      xref: "paper", yref: "paper", x: 0.01, y: 0.99,
      showarrow: false, font: { color: "#aaa", size: 11 },
    }],
  }, { displayModeBar: true, modeBarButtonsToRemove: ["toImage"], responsive: true });

  initialized2d = true;
  updateTimelabel(label);
}

// ── 3D Surface ────────────────────────────────────────────────────────────────
function render3D() {
  if (!state.sliceData) return;
  const { z, zmin, zmax, label } = getCurrentZ(state.window);
  const d = state.sliceData;
  const cscale = state.mode === "error" ? "Hot" : "Turbo";

  Plotly.react("plot-3d", [{
    type: "surface",
    x: d.xi, y: d.yi, z,
    colorscale: cscale,
    cmin: zmin, cmax: zmax,
    colorbar: {
      title: { text: "T [°C]", side: "top", font: { color: "#aaa", size: 11 } },
      tickfont: { color: "#aaa", size: 10 },
      thickness: 14,
    },
    hovertemplate: "x=%{x:.3f}m<br>y=%{y:.3f}m<br>T=%{z:.1f}°C<extra></extra>",
    contours: {
      z: { show: true, usecolormap: true, highlightcolor: "white", project: { z: false } }
    },
  }], {
    paper_bgcolor: "transparent",
    scene: {
      bgcolor: "#1a2035",
      xaxis: { title: "x [m]", color: "#aaa", gridcolor: "#2a3456" },
      yaxis: { title: "y [m]", color: "#aaa", gridcolor: "#2a3456" },
      zaxis: { title: "T [°C]", color: "#aaa", gridcolor: "#2a3456" },
      camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } },
    },
    font:   { color: "#ccc", family: "Inter" },
    margin: { t: 20, b: 20, l: 20, r: 20 },
  }, { displayModeBar: true, responsive: true });

  initialized3d = true;
  updateTimelabel(label);
}

// ── Loss Curves ───────────────────────────────────────────────────────────────
function renderLoss() {
  if (!state.lossData) {
    Plotly.react("plot-loss", [], {
      paper_bgcolor: "transparent",
      annotations: [{
        text: "Loss data not yet available",
        xref: "paper", yref: "paper", x: 0.5, y: 0.5,
        showarrow: false, font: { color: "#888", size: 14 },
      }]
    }, { displayModeBar: false });
    return;
  }

  const lossGroups = state.lossData;  // array of windows
  const lineKeys   = ["L_total", "L_bc", "L_phys", "L_ic"];
  const lineColors = { L_total: "#3b82f6", L_bc: "#f97316", L_phys: "#22c55e", L_ic: "#a855f7" };
  const lineNames  = { L_total: "Total", L_bc: "BC", L_phys: "Physics", L_ic: "IC" };

  const traces = [];
  lossGroups.forEach((lossHist, wi) => {
    lineKeys.forEach(k => {
      const vals = lossHist[k];
      if (!vals) return;
      traces.push({
        x: Array.from({length: vals.length}, (_, i) => i),
        y: vals,
        type: "scatter", mode: "lines",
        name: `W${wi} ${lineNames[k]}`,
        line: { color: lineColors[k], width: 1.2, dash: wi === state.window ? "solid" : "dot" },
        opacity: wi === state.window ? 1.0 : 0.35,
        legendgroup: lineNames[k],
        showlegend: wi === 0,
      });
    });
  });

  Plotly.react("plot-loss", traces, {
    paper_bgcolor: "transparent",
    plot_bgcolor:  "#1a2035",
    font:   { color: "#ccc", family: "Inter" },
    xaxis:  { title: "Epoch", gridcolor: "#2a3456", color: "#aaa" },
    yaxis:  { title: "Loss", type: "log", gridcolor: "#2a3456", color: "#aaa" },
    legend: { font: { size: 10 }, bgcolor: "rgba(26,32,53,0.9)", bordercolor: "#2a3456", borderwidth: 1 },
    margin: { t: 20, b: 50, l: 70, r: 20 },
  }, { displayModeBar: false, responsive: true });

  initializedLoss = true;
}

// ── Update charts on window/mode change ───────────────────────────────────────
function updateCharts() {
  if (!state.sliceData) return;
  updateStats();

  if (currentTab === "2d" && initialized2d) {
    const { z, zmin, zmax } = getCurrentZ(state.window);
    const cscale = state.mode === "error" ? "Hot" : "Turbo";
    Plotly.update("plot-2d",
      { z: [z], zmin: [zmin], zmax: [zmax], colorscale: [cscale] }
    );
  }
  if (currentTab === "3d" && initialized3d) {
    const { z, zmin, zmax } = getCurrentZ(state.window);
    const cscale = state.mode === "error" ? "Hot" : "Turbo";
    Plotly.update("plot-3d",
      { z: [z], cmin: [zmin], cmax: [zmax], colorscale: [cscale] }
    );
  }
  if (currentTab === "loss" && initializedLoss) {
    renderLoss();  // re-render to highlight active window
  }
}

function updateTimelabel(label) {
  const title2d = document.getElementById("chart2d-title");
  const title3d = document.getElementById("chart3d-title");
  const wi = state.window;
  const d  = state.sliceData;
  if (title2d) title2d.textContent = `${label} — Window ${wi}`;
  if (title3d) title3d.textContent = `3D Surface: ${label} — Window ${wi}`;
  if (d) {
    const hint = document.getElementById("time-hint");
    if (hint) hint.textContent = `window ${wi + 1} / ${Object.keys(d.windows).length}`;
  }
}

function updateStats() {
  if (!state.sliceData) return;
  const d   = state.sliceData;
  const wi  = state.window;
  const win = d.windows[String(wi)];
  if (!win) return;

  const flat_pred = win.T_pred.flat();
  const flat_fem  = win.T_fem.flat();
  const err  = flat_pred.map((v, i) => Math.abs(v - flat_fem[i]));
  const mae  = (err.reduce((a, b) => a + b, 0) / err.length).toFixed(2);
  const tmin = Math.min(...flat_pred).toFixed(1);
  const tmax = Math.max(...flat_pred).toFixed(1);

  // Mean MAE across all windows
  const nWin = Object.keys(d.windows).length;
  let sumMae = 0;
  for (let w = 0; w < nWin; w++) {
    const ww = d.windows[String(w)];
    if (!ww) continue;
    const e = ww.T_pred.flat().map((v, i) => Math.abs(v - ww.T_fem.flat()[i]));
    sumMae += e.reduce((a, b) => a + b, 0) / e.length;
  }
  const meanMae = (sumMae / nWin).toFixed(2);

  const skip   = parseInt(state.skip);
  const total  = 21;  // total time steps
  const femPts = Math.ceil(total / skip);
  const saved  = total - femPts;
  const pct    = Math.round(saved / total * 100);

  document.getElementById("stat-mae").textContent      = `${mae} °C`;
  document.getElementById("stat-mae-mean").textContent  = `${meanMae} °C`;
  document.getElementById("stat-trange").textContent    = `${tmin} – ${tmax} °C`;
  document.getElementById("stat-fem").textContent       = `${saved}/${total} steps (${pct}%)`;

  updateTimelabel(state.mode === "pred" ? "PINN" : state.mode === "fem" ? "FEM" : "|Error|");
}

// ── Training status polling ────────────────────────────────────────────────────
function checkTrainingStatus() {
  fetch("/api/status").then(r => r.json()).then(s => {
    const box = document.getElementById("train-status");
    if (s.done < s.total) {
      box.style.display = "block";
      document.getElementById("train-progress").style.width = s.percent + "%";
      document.getElementById("train-pct").textContent = `${s.done} / ${s.total} runs complete`;
      setTimeout(checkTrainingStatus, 15000);  // re-check every 15s
    } else {
      box.style.display = "none";
    }
  }).catch(() => {});
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadData();
checkTrainingStatus();
