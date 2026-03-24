#!/bin/bash
# Quick start guide for NAS-PINNS3 web interface

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NAS-PINNS3 Web Interface — Quick Start"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Navigate to web directory
cd \"$(dirname \"$0\")/NAS-PINNS3/web\" || exit 1

echo ""
echo "📁 Working directory: $(pwd)"
echo ""

# Check if data exists
DATA_DIR=\"../level8_nas_mco_pinn/results/v2\"
if [ ! -d \"$DATA_DIR\" ]; then
    echo \"❌ Data directory not found: $DATA_DIR\"
    echo \"   Run: python3 run_3d_v2.py (from NAS-PINNS3 directory)\"
    exit 1
fi

# Check Flask
if ! python3 -c \"import flask\" 2>/dev/null; then
    echo \"⚠️  Flask not installed. Installing...\"
    pip3 install flask
fi

echo \"✅ Dependencies ready\"
echo \"✅ Data found in: $DATA_DIR\"
echo \"\"
echo \"🚀 Starting Flask server on http://localhost:5000\"
echo \"   Interactive demo: http://localhost:5000/demo\"
echo \"   Results page: http://localhost:5000/results\"
echo \"\"
echo \"Press Ctrl+C to stop the server\"
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"
echo \"\"

# Launch Flask
python3 app.py
