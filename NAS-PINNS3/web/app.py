"""
NAS-MCO-PINNs  —  Web Application
Flask backend serving pre-computed results and interactive demo.
"""
import os, json, glob, math
from flask import Flask, render_template, jsonify, request, send_from_directory, Response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "level8_nas_mco_pinn", "results")
V2_DIR   = os.path.join(DATA_DIR, "v2")
IMG_DIR  = os.path.join(BASE_DIR, "static", "img")

app = Flask(__name__)


def _nan_to_null(obj):
    """Recursively replace float NaN/Inf with None so JSON stays valid."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_null(v) for v in obj]
    return obj


def json_response(data):
    """Return a JSON response that replaces NaN with null."""
    return Response(json.dumps(_nan_to_null(data)), mimetype="application/json")


# ─── Pages ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/demo")
def demo():
    return render_template("demo.html")

@app.route("/results")
def results_page():
    return render_template("results.html")


# ─── API ──────────────────────────────────────────────────────────────────────

@app.route("/api/available")
def api_available():
    """Which (domain, arch, skip) combinations have v2 slice data."""
    available = []
    if os.path.exists(V2_DIR):
        for fpath in glob.glob(os.path.join(V2_DIR, "*_slice.json")):
            name = os.path.basename(fpath).replace("_slice.json", "")
            # name pattern:  {domain}_{arch}_skip{s}
            # e.g. rectangular_bayesian_skip2
            # split from right to get skip part and arch
            parts = name.rsplit("_", 2)   # ['rectangular', 'bayesian', 'skip2']
            if len(parts) == 3:
                domain, arch, skip_str = parts
                available.append({
                    "domain": domain,
                    "arch":   arch,
                    "skip":   skip_str.replace("skip", ""),
                })
    return jsonify(available)


@app.route("/api/slice")
def api_slice():
    domain = request.args.get("domain", "rectangular")
    arch   = request.args.get("arch",   "bayesian")
    skip   = request.args.get("skip",   "2")
    path   = os.path.join(V2_DIR, f"{domain}_{arch}_skip{skip}_slice.json")
    if not os.path.exists(path):
        return jsonify({"error": "Data not yet available — training may still be running."}), 404
    with open(path) as f:
        data = json.load(f)
    return json_response(data)


@app.route("/api/loss")
def api_loss():
    domain = request.args.get("domain", "rectangular")
    arch   = request.args.get("arch",   "bayesian")
    skip   = request.args.get("skip",   "2")
    path   = os.path.join(V2_DIR, f"{domain}_{arch}_skip{skip}_loss.json")
    if not os.path.exists(path):
        return jsonify({"error": "Loss data not yet available."}), 404
    with open(path) as f:
        data = json.load(f)
    return json_response(data)


@app.route("/api/mae")
def api_mae():
    result = {}
    for version, fname in [("v1", "results_3d.json"), ("v2", "results_3d_v2.json")]:
        base = DATA_DIR if version == "v1" else V2_DIR
        fpath = os.path.join(base, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                result[version] = json.load(f)
    path_2d = os.path.join(DATA_DIR, "level8_skip_results.json")
    if os.path.exists(path_2d):
        with open(path_2d) as f:
            result["2d"] = json.load(f)
    return jsonify(result)


@app.route("/api/status")
def api_status():
    """Training progress check."""
    total = 48   # 4 domains × 3 archs × 4 skips
    done  = len(glob.glob(os.path.join(V2_DIR, "*_slice.json")))
    return jsonify({"total": total, "done": done, "percent": int(done / total * 100)})


# ─── Static results images ────────────────────────────────────────────────────

@app.route("/img/<path:filename>")
def serve_img(filename):
    return send_from_directory(IMG_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
