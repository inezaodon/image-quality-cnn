#!/bin/bash
# Start the face quality scoring web demo locally.
set -euo pipefail
cd "$(dirname "$0")"

if ! python3 -c "import torch" 2>/dev/null; then
  echo "Installing web dependencies (first run may take a minute)..."
  pip3 install --user -r requirements-web.txt
fi

cd web
echo "Starting server at http://127.0.0.1:5000"
python3 app.py
