'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let framesManifest = null;
let currentFrame   = 0;
let playInterval   = null;
let NTIMES         = 10;

const ARCH_LABELS   = { bayesian:'Bayesian (TPE)', nsga2:'NSGA-II', nsga3:'NSGA-III' };
const DOMAIN_LABELS = {
    rectangle:'Rectangle 2D', circle:'Circle 2D', lshape:'L-Shape 2D',
    rectangular:'Rectangular 3D', cylinder:'Cylinder 3D',
    stacked:'Stacked 3D', lshape3d:'L-Shape 3D'
};
const DOMAIN_LABELS_SHORT = {
    rectangle:'Rectangle', circle:'Circle', lshape:'L-Shape',
    rectangular:'Rectangular', cylinder:'Cylinder', stacked:'Stacked'
};

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    showOverlay('Loading…');
    try {
        const res  = await fetch('data/manifest_frames.json');
        framesManifest = await res.json();
        NTIMES = framesManifest.times ? framesManifest.times.length : 10;
    } catch {
        showOverlay('Could not load manifest — run generate_frames.py first.');
        return;
    }
    initSelectors();
    initControls();
});

// ── Overlay ───────────────────────────────────────────────────────────────────
function showOverlay(msg) {
    const o = document.getElementById('viewer-overlay');
    if (!o) return;
    o.textContent = msg;
    o.style.display = 'flex';
}
function hideOverlay() {
    const o = document.getElementById('viewer-overlay');
    if (o) o.style.display = 'none';
}

// ── Selectors ─────────────────────────────────────────────────────────────────
function initSelectors() {
    document.getElementById('sel-dim').addEventListener('change',    () => { updateDomainOpts(); loadFrame(0); });
    document.getElementById('sel-domain').addEventListener('change', () => { updateArchOpts();   loadFrame(0); });
    document.getElementById('sel-arch').addEventListener('change',   () => { updateKOpts();      loadFrame(0); });
    document.getElementById('sel-k').addEventListener('change',      () => loadFrame(0));
    updateDomainOpts();
    loadFrame(0);
}

function dimKey()  { return document.getElementById('sel-dim').value; }
function dimData() { return framesManifest[dimKey()] || {}; }

function updateDomainOpts() {
    const sel = document.getElementById('sel-domain');
    sel.innerHTML = Object.keys(dimData())
        .map(d => `<option value="${d}">${DOMAIN_LABELS_SHORT[d]||d}</option>`).join('');
    updateArchOpts();
}
function updateArchOpts() {
    const domain = document.getElementById('sel-domain').value;
    const sel    = document.getElementById('sel-arch');
    sel.innerHTML = Object.keys((dimData()[domain])||{})
        .map(a => `<option value="${a}">${ARCH_LABELS[a]||a}</option>`).join('');
    updateKOpts();
}
function updateKOpts() {
    const domain = document.getElementById('sel-domain').value;
    const arch   = document.getElementById('sel-arch').value;
    const ks     = ((dimData()[domain]||{})[arch])||[];
    document.getElementById('sel-k').innerHTML = ks
        .map(k => `<option value="${k}">k = ${k}  (Δt = ${(k*1.5).toFixed(1)} s)</option>`).join('');
}

// ── Frame loading ─────────────────────────────────────────────────────────────
function frameSrc(tidx) {
    const dim    = dimKey();
    const domain = document.getElementById('sel-domain').value;
    const arch   = document.getElementById('sel-arch').value;
    const k      = document.getElementById('sel-k').value;
    return `frames/${dim}_${domain}_${arch}_k${k}_t${String(tidx).padStart(2,'0')}.png`;
}

function loadFrame(tidx, keepPlaying = false) {
    if (!keepPlaying) stopPlay();
    currentFrame = tidx;
    const img  = document.getElementById('viewer-img');
    const times = framesManifest.times || [3,6,9,12,15,18,21,24,27,30];

    showOverlay('Loading…');
    img.onload  = () => { hideOverlay(); };
    img.onerror = () => { showOverlay('Frame not found — run generate_frames.py'); };
    img.src = frameSrc(tidx);

    document.getElementById('time-label').textContent = `t = ${times[tidx]?.toFixed(0) || '—'} s`;
    document.getElementById('time-slider').value = tidx;

    // Preload adjacent frames
    [tidx-1, tidx+1].forEach(i => {
        if (i >= 0 && i < NTIMES) { const p = new Image(); p.src = frameSrc(i); }
    });
}

// ── Controls ──────────────────────────────────────────────────────────────────
function initControls() {
    const slider = document.getElementById('time-slider');
    slider.max   = NTIMES - 1;
    slider.addEventListener('input', e => {
        stopPlay();
        loadFrame(parseInt(e.target.value));
    });
    document.getElementById('play-btn').addEventListener('click', togglePlay);
}

function togglePlay() {
    if (playInterval) { stopPlay(); return; }
    document.getElementById('play-btn').textContent = '⏸';
    playInterval = setInterval(() => {
        loadFrame((currentFrame + 1) % NTIMES, true);
    }, 700);
}
function stopPlay() {
    if (!playInterval) return;
    clearInterval(playInterval);
    playInterval = null;
    document.getElementById('play-btn').textContent = '▶';
}

// ── Gallery helpers (best_k, timeline, step) ──────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Tab switching for best_k gallery
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const group = btn.dataset.group;
            document.querySelectorAll(`.tab-btn[data-group="${group}"]`)
                    .forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const target = btn.dataset.target;
            document.querySelectorAll(`.tab-panel[data-group="${group}"]`)
                    .forEach(p => p.style.display = p.dataset.panel === target ? 'block' : 'none');
        });
    });
});

// ── Lightbox ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const lb      = document.getElementById('lightbox');
    const lbImg   = document.getElementById('lb-img');
    const lbCap   = document.getElementById('lb-caption');
    const lbClose = document.getElementById('lb-close');

    function openLightbox(src, caption) {
        lbImg.src = src;
        lbCap.textContent = caption || '';
        lb.classList.add('open');
        lb.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
    }
    function closeLightbox() {
        lb.classList.remove('open');
        lb.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        lbImg.src = '';
    }

    // Delegate clicks on zoomable images
    document.addEventListener('click', e => {
        const img = e.target.closest('.result-figures img, .img-row-2 img, .img-row-3 img');
        if (img) {
            const caption = img.closest('figure')?.querySelector('figcaption')?.textContent || '';
            openLightbox(img.src, caption);
        }
    });

    lbClose.addEventListener('click', e => { e.stopPropagation(); closeLightbox(); });
    lb.addEventListener('click', closeLightbox);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });
});
