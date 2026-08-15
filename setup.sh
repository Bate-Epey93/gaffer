#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
fi
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt
echo "Done. Activate with: source .venv/bin/activate"
./.venv/bin/python -c "import pandas, numpy, pulp, fastapi; print('deps ok')"
