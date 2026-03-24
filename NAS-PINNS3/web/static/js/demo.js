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
    const femPct = {"1":"0%","2":"52%","4":"71%","6":"81%"};
    document.getElementById("skip-hint").textContent = (femPct[state.skip] || "") + " FEM savings";
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

    // Warn if geometry metadata is missing (old data format)
    if (!slice.geometry) {
      console.warn("⚠️  Geometry metadata not found. Using default visualization. " +
                   "Please regenerate data with updated run_3d_v2.py to see domain-specific geometry.");
    }

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

  // Get geometry metadata for annotations
  const geom = d.geometry || { type: "rectangular", params: {} };
  const p = geom.params;

  // Build traces and shapes based on domain type
  const traces = [{
    type: "heatmap",
    x: d.xi, y: d.yi, z,
    colorscale: cscale,
    zmin, zmax,
    colorbar: {
      title: { text: "T [°C]", side: "right", font: { color: "#aaa", size: 11 } },
      tickfont: { color: "#aaa", size: 10 },
    },
    hovertemplate: "x=%{x:.3f}m  y=%{y:.3f}m<br>T=%{z:.1f}°C<extra></extra>",
  }];

  // Helper: convert data coordinates to plot coordinates for shapes
  const xMin = d.xi[0], xMax = d.xi[d.xi.length - 1];
  const yMin = d.yi[0], yMax = d.yi[d.yi.length - 1];

  // Build shapes to highlight domain boundaries
  const shapes = [];
  let annotText = `z = ${d.z_val.toFixed(3)} m  (mid-slice)`;

  if (geom.type === "lshape") {
    // L-shape: draw boundary lines showing the cut corner
    const cut_x = p.cut_x || 0.3;
    const cut_y = p.cut_y || 0.3;
    
    // Draw the outer boundary
    shapes.push({
      type: "rect",
      xref: "x", yref: "y",
      x0: xMin, y0: yMin, x1: xMax, y1: yMax,
      line: { color: "#888", width: 1, dash: "dash" },
      fillcolor: "transparent",
    });

    // Draw L-shape domain boundary (show the removed quadrant)
    shapes.push({
      type: "rect",
      xref: "x", yref: "y",
      x0: cut_x, y0: cut_y, x1: xMax, y1: yMax,
      line: { color: "#ff6b6b", width: 2, dash: "dash" },
      fillcolor: "rgba(255, 107, 107, 0.1)",
    });

    annotText = `L-Shape: z = ${d.z_val.toFixed(3)} m  (cut_x=${cut_x}m, cut_y=${cut_y}m)`;
    
    // Add text annotation showing removed region
    traces.push({
      x: [cut_x + (xMax - cut_x) / 2],
      y: [cut_y + (yMax - cut_y) / 2],
      text: ["Removed"],
      mode: "text",
      textposition: "middle center",
      textfont: { color: "#ff6b6b", size: 10 },
      showlegend: false,
      hoverinfo: "skip",
    });
  } else if (geom.type === "cylinder") {
    const R = p.R || 0.25;
    annotText = `Cylinder: z = ${d.z_val.toFixed(3)} m  (R=${R}m)`;
    
    // Draw circle boundary
    const cx = xMin + (xMax - xMin) / 2;
    const cy = yMin + (yMax - yMin) / 2;
    // Note: Plotly's circle support via shapes is limited, so we'll just note it in the annotation
  } else if (geom.type === "stacked") {
    const L = p.L_cube || 0.5;
    const N = p.N_stack || 2;
    
    // Draw square boundary
    shapes.push({
      type: "rect",
      xref: "x", yref: "y",
      x0: 0, y0: 0, x1: L, y1: L,
      line: { color: "#FF9800", width: 2 },
      fillcolor: "transparent",
    });
    
    // Determine which cube we're slicing
    let cubeNum = 1;
    let distToCube1Mid = Math.abs(d.z_val - L/2);
    let distToCube2Mid = Math.abs(d.z_val - 3*L/2);
    if (distToCube2Mid < distToCube1Mid && N > 1) cubeNum = 2;
    
    annotText = `Stacked Cubes: ${N}×${L}m  [Cube ${cubeNum}]  z = ${d.z_val.toFixed(3)} m`;
  } else {
    annotText = `${geom.type}: z = ${d.z_val.toFixed(3)} m`;
  }

  Plotly.react("plot-2d", traces, {
    paper_bgcolor: "transparent",
    plot_bgcolor:  "#1a2035",
    font:  { color: "#ccc", family: "Inter" },
    xaxis: { title: "x [m]", gridcolor: "#2a3456", color: "#aaa", zeroline: false },
    yaxis: { title: "y [m]", gridcolor: "#2a3456", color: "#aaa", zeroline: false, scaleanchor: "x" },
    margin: { t: 20, b: 50, l: 60, r: 80 },
    shapes: shapes,
    annotations: [{
      text: annotText,
      xref: "paper", yref: "paper", x: 0.01, y: 0.99,
      showarrow: false, font: { color: "#aaa", size: 11 },
    }],
  }, { displayModeBar: true, modeBarButtonsToRemove: ["toImage"], responsive: true });

  initialized2d = true;
  updateTimelabel(label);
}

// ── 3D Surface ────────────────────────────────────────────────────────────────
function buildLShapeMask(xi, yi, cut_x, cut_y) {
  /**
   * Build a 2D grid of NaN for regions outside L-shape domain.
   * L-shape = (x <= cut_x) OR (y <= cut_y)
   */
  const ny = yi.length;
  const nx = xi.length;
  const mask = Array(ny);
  for (let j = 0; j < ny; j++) {
    mask[j] = Array(nx);
    for (let i = 0; i < nx; i++) {
      const x = xi[i];
      const y = yi[j];
      const inside = (x <= cut_x) || (y <= cut_y);
      mask[j][i] = inside ? 1.0 : NaN;  // 1.0 inside, NaN outside
    }
  }
  return mask;
}

function applyMaskToValues(values, mask) {
  /**
   * Apply mask: set values to NaN where mask is null/NaN.
   * This makes Plotly not render those points.
   */
  if (!mask) return values;
  const masked = Array(values.length);
  for (let j = 0; j < values.length; j++) {
    masked[j] = Array(values[j].length);
    for (let i = 0; i < values[j].length; i++) {
      // If value is already null, keep it
      if (values[j][i] === null) {
        masked[j][i] = null;
      }
      // If mask says this point is outside, set to NaN
      else if (mask[j][i] === null || isNaN(mask[j][i])) {
        masked[j][i] = NaN;
      }
      // Otherwise keep the value
      else {
        masked[j][i] = values[j][i];
      }
    }
  }
  return masked;
}

function buildDomainBoundary(geom, z_val) {
  /**
   * Create 3D boundary traces that show actual domain geometry
   */
  const traces = [];
  
  if (geom.type === "rectangular") {
    const p = geom.params;
    const Lx = p.Lx || 1.3;
    const Ly = p.Ly || 0.6;
    
    // Draw rectangle boundary at z_val
    const corners = [
      [0, 0], [Lx, 0], [Lx, Ly], [0, Ly], [0, 0]
    ];
    const x_pts = corners.map(c => c[0]);
    const y_pts = corners.map(c => c[1]);
    const z_pts = corners.map(() => z_val);
    
    traces.push({
      type: "scatter3d",
      x: x_pts, y: y_pts, z: z_pts,
      mode: "lines",
      line: { color: "#1565C0", width: 4 },
      name: "Domain Boundary",
      showlegend: true,
      hoverinfo: "skip",
    });
  }
  else if (geom.type === "cylinder") {
    const p = geom.params;
    const R = p.R || 0.25;
    const nPts = 32;
    const x_pts = [], y_pts = [];
    
    // Create circle boundary
    for (let i = 0; i <= nPts; i++) {
      const angle = (i / nPts) * 2 * Math.PI;
      x_pts.push(R * Math.cos(angle));
      y_pts.push(R * Math.sin(angle));
    }
    const z_pts = x_pts.map(() => z_val);
    
    traces.push({
      type: "scatter3d",
      x: x_pts, y: y_pts, z: z_pts,
      mode: "lines",
      line: { color: "#2E7D32", width: 4 },
      name: "Domain Boundary",
      showlegend: true,
      hoverinfo: "skip",
    });
  }
  else if (geom.type === "stacked") {
    const p = geom.params;
    const L = p.L_cube || 0.5;
    
    // Draw square boundary at z_val
    const corners = [
      [0, 0], [L, 0], [L, L], [0, L], [0, 0]
    ];
    const x_pts = corners.map(c => c[0]);
    const y_pts = corners.map(c => c[1]);
    const z_pts = corners.map(() => z_val);
    
    traces.push({
      type: "scatter3d",
      x: x_pts, y: y_pts, z: z_pts,
      mode: "lines",
      line: { color: "#E65100", width: 4 },
      name: "Domain Boundary",
      showlegend: true,
      hoverinfo: "skip",
    });
    
    // Add interface line at z = L
    if (p.N_stack > 1) {
      traces.push({
        type: "scatter3d",
        x: x_pts, y: y_pts, z: corners.map(() => L),
        mode: "lines",
        line: { color: "#FF9800", width: 3, dash: "dash" },
        name: "Cube Interface",
        showlegend: true,
        hoverinfo: "skip",
      });
    }
  }
  else if (geom.type === "lshape") {
    const p = geom.params;
    const Lx = p.Lx || 0.8;
    const Ly = p.Ly || 0.8;
    const cut_x = p.cut_x || 0.3;
    const cut_y = p.cut_y || 0.3;
    
    // L-shape boundary (outer + inner notch)
    const corners = [
      [0, 0], [Lx, 0], [Lx, cut_y], [cut_x, cut_y], 
      [cut_x, Ly], [0, Ly], [0, 0]
    ];
    const x_pts = corners.map(c => c[0]);
    const y_pts = corners.map(c => c[1]);
    const z_pts = corners.map(() => z_val);
    
    traces.push({
      type: "scatter3d",
      x: x_pts, y: y_pts, z: z_pts,
      mode: "lines",
      line: { color: "#D32F2F", width: 4 },
      name: "Domain Boundary",
      showlegend: true,
      hoverinfo: "skip",
    });
  }
  
  return traces;
}

function maskDataForDomain(data, geom, Tmat, d) {
  /**
   * Apply geometry-specific masking to temperature data.
   * Returns: {surfaceZ, colorData}
   */
  const ny = Tmat.length;
  const nx = Tmat[0] ? Tmat[0].length : 0;
  
  // Filter out null/NaN values for min/max calculation
  const validVals = Tmat.flat().filter(v => v !== null && !isNaN(v));
  if (validVals.length === 0) return { surfaceZ: Tmat, colorData: Tmat };
  
  const vmin = Math.min(...validVals);
  const vmax = Math.max(...validVals);
  const tempRange = vmax - vmin || 1;
  const zScale = 0.9; // Moderate 3D elevation for balance
  
  let surfaceZ = Tmat.map(row => 
    row.map(t => {
      if (t === null || isNaN(t)) return null;
      return d.z_val + ((t - vmin) / tempRange) * zScale;
    })
  );
  
  let colorData = Tmat;

  if (geom.type === "lshape") {
    const p = geom.params;
    const lshapeMask = buildLShapeMask(d.xi, d.yi, p.cut_x, p.cut_y);
    surfaceZ = applyMaskToValues(surfaceZ, lshapeMask);
    colorData = applyMaskToValues(Tmat, lshapeMask);
  } 
  else if (geom.type === "cylinder") {
    const p = geom.params;
    const R = p.R || 0.25;
    const cylMask = Array(ny);
    for (let j = 0; j < ny; j++) {
      cylMask[j] = Array(nx);
      for (let i = 0; i < nx; i++) {
        const x = d.xi[i];
        const y = d.yi[j];
        const r = Math.sqrt(x*x + y*y);
        cylMask[j][i] = (r <= R) ? 1.0 : null;
      }
    }
    surfaceZ = applyMaskToValues(surfaceZ, cylMask);
    colorData = applyMaskToValues(Tmat, cylMask);
  }
  else if (geom.type === "stacked") {
    // Stacked cubes: square cross-section L_cube × L_cube
    const p = geom.params;
    const L = p.L_cube || 0.5;
    const stackedMask = Array(ny);
    for (let j = 0; j < ny; j++) {
      stackedMask[j] = Array(nx);
      for (let i = 0; i < nx; i++) {
        const x = d.xi[i];
        const y = d.yi[j];
        // Square domain [0, L_cube] × [0, L_cube]
        stackedMask[j][i] = (x >= 0 && x <= L && y >= 0 && y <= L) ? 1.0 : null;
      }
    }
    surfaceZ = applyMaskToValues(surfaceZ, stackedMask);
    colorData = applyMaskToValues(Tmat, stackedMask);
  }
  
  return { surfaceZ, colorData };
}

function render3D() {
  if (!state.sliceData) return;
  const { z: Tmat, zmin, zmax, label } = getCurrentZ(state.window);
  const d = state.sliceData;
  const cscale = state.mode === "error" ? "Hot" : "Turbo";

  // Get geometry metadata (if available)
  const geom = d.geometry || { type: "rectangular", params: {} };
  
  // Apply domain-specific masking and get 3D surface
  const { surfaceZ, colorData } = maskDataForDomain(Tmat, geom, Tmat, d);
  
  // Build annotation text
  let annotText = `${geom.type}: z-mid = ${d.z_val.toFixed(3)} m`;
  if (geom.type === "lshape") {
    const p = geom.params;
    annotText = `L-Shape: z-mid = ${d.z_val.toFixed(3)} m  (cut_x=${p.cut_x}m, cut_y=${p.cut_y}m)`;
  } else if (geom.type === "cylinder") {
    const p = geom.params;
    annotText = `Cylinder: z-mid = ${d.z_val.toFixed(3)} m  (R=${p.R}m)`;
  } else if (geom.type === "stacked") {
    const p = geom.params;
    annotText = `Stacked Cubes: ${p.N_stack}×${p.L_cube}m cube  z-mid = ${d.z_val.toFixed(3)} m`;
  }

  const ny = Tmat.length;
  const nx = Tmat[0] ? Tmat[0].length : 0;
  const zScale = 0.9; // Moderate 3D elevation for balance

  const traces = [{
    type: "surface",
    x: d.xi, y: d.yi, z: surfaceZ,
    surfacecolor: colorData,
    colorscale: cscale,
    cmin: zmin, cmax: zmax,
    colorbar: {
      title: { text: "T [°C]", side: "top", font: { color: "#aaa", size: 11 } },
      tickfont: { color: "#aaa", size: 10 },
      thickness: 14,
    },
    hovertemplate: "x=%{x:.3f}m<br>y=%{y:.3f}m<br>T=%{customdata:.1f}°C<extra></extra>",
    customdata: colorData,
  }];

  // Add domain boundary visualization
  const boundaryTraces = buildDomainBoundary(geom, d.z_val);
  traces.push(...boundaryTraces);

  // For stacked cubes, add visible wireframe edges and domain box
  let annotations3d = [{
    text: annotText,
    x: d.xi[Math.floor(nx/2)], y: d.yi[Math.floor(ny/2)], z: d.z_val + zScale * 0.5,
    showarrow: false, font: { color: "#aaa", size: 10 },
  }];

  if (geom.type === "stacked") {
    const p = geom.params;
    const L = p.L_cube || 0.5;
    const interfaceZ = p.L_cube; // Interface at z = L_cube
    // Interface is already drawn by buildDomainBoundary
  }

  Plotly.react("plot-3d", traces, {
    paper_bgcolor: "transparent",
    scene: {
      bgcolor: "#1a2035",
      xaxis: { title: "x [m]", color: "#aaa", gridcolor: "#2a3456" },
      yaxis: { title: "y [m]", color: "#aaa", gridcolor: "#2a3456" },
      zaxis: { title: "z [m]", color: "#aaa", gridcolor: "#2a3456",
               range: [d.z_val - 0.05, d.z_val + zScale + 0.15] },
      camera: { eye: { x: 1.2, y: 1.2, z: 1.0 } },
    },
    font:   { color: "#ccc", family: "Inter" },
    margin: { t: 20, b: 20, l: 20, r: 20 },
    annotations3d: annotations3d,
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
    const { z: Tmat, zmin, zmax } = getCurrentZ(state.window);
    const cscale = state.mode === "error" ? "Hot" : "Turbo";
    const d = state.sliceData;
    const geom = d.geometry || { type: "rectangular", params: {} };
    
    // Apply domain-specific masking
    const { surfaceZ, colorData } = maskDataForDomain(Tmat, geom, Tmat, d);
    
    Plotly.update("plot-3d",
      { z: [surfaceZ], surfacecolor: [colorData], customdata: [colorData], cmin: [zmin], cmax: [zmax], colorscale: [cscale] }
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
