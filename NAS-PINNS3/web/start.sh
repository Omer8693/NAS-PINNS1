#!/bin/bash
# Start NAS-MCO-PINNs web server
pkill -f "web/app.py" 2>/dev/null
sleep 1
cd "$(dirname "$0")"
nohup python3 app.py > /tmp/flask_nas.log 2>&1 &
echo "Server started at http://localhost:5000 (PID=$!)"
echo "Logs: tail -f /tmp/flask_nas.log"
