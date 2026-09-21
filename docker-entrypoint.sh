#!/bin/sh
set -eu

host="${OLLAMA_HOST:-http://127.0.0.1:11434}"
model="${LLM_MODEL:-llama3.1}"

echo "waiting for Ollama at ${host}"
i=0
while [ "$i" -lt 60 ]; do
  if python -c "import urllib.request; urllib.request.urlopen('${host}/api/tags', timeout=2)" 2>/dev/null; then
    break
  fi
  i=$((i + 1))
  sleep 2
done

echo "pulling ${model} if missing"
python - "$host" "$model" <<'PY'
import json, sys, urllib.request
host, model = sys.argv[1], sys.argv[2]
req = urllib.request.Request(
    f"{host}/api/pull",
    data=json.dumps({"name": model, "stream": False}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    urllib.request.urlopen(req, timeout=600).read()
    print(f"model {model} ready")
except Exception as exc:
    print(f"warning: could not pull {model}: {exc}")
PY

if ls data/papers/*.pdf >/dev/null 2>&1; then
  echo "ingesting data/papers"
  python -m dsqa ingest data/papers || echo "warning: ingest failed (UI will still start)"
else
  echo "no PDFs in data/papers; start the UI and ingest later"
fi

exec python -m dsqa ui --host 0.0.0.0 --port 8765
