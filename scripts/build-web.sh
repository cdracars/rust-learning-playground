#!/usr/bin/env bash
# Build the WebAssembly module and inline it into a single self-contained
# HTML page at dist/photon.html (open it directly, no server needed).
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release --target wasm32-unknown-unknown -p photon-wasm
wasm=target/wasm32-unknown-unknown/release/photon_wasm.wasm
mkdir -p dist
python3 - "$wasm" web/index.html dist/photon.html <<'PY'
import base64, sys
wasm, src, out = sys.argv[1:]
b64 = base64.b64encode(open(wasm, "rb").read()).decode()
html = open(src).read()
assert "__PHOTON_WASM_BASE64__" in html
open(out, "w").write(html.replace("__PHOTON_WASM_BASE64__", b64))
PY
echo "Wrote dist/photon.html ($(du -h dist/photon.html | cut -f1))"
