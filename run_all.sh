#!/usr/bin/env bash
# run_all.sh — start all wisdom-tree services in the correct order
#
# Usage:
#   ./run_all.sh [s2s_service.py options, e.g. --whisper-model large-v3]
#
# Start order:
#   1. Build Rust daemon
#   2. s2s_service.py      (wait until /tmp/s2s.sock appears)
#   3. wisdom-tree         (Rust daemon, creates /tmp/wakeword.sock)
#   4. keypress_listener.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
YLW='\033[1;33m'   # [WAKE]
BLU='\033[0;34m'   # [S2S]
GRN='\033[0;32m'   # [RUST]
RED='\033[0;31m'
NC='\033[0m'

_wake() { printf "${YLW}[WAKE] %s${NC}\n" "$*" >&2; }
_s2s()  { printf "${BLU}[S2S]  %s${NC}\n" "$*" >&2; }
_rust() { printf "${GRN}[RUST] %s${NC}\n" "$*" >&2; }
_err()  { printf "${RED}[ERR]  %s${NC}\n" "$*" >&2; }

# ---------------------------------------------------------------------------
# 0. All args go to s2s_service (no wake-word flags any more)
# ---------------------------------------------------------------------------
S2S_ARGS=("$@")

# ---------------------------------------------------------------------------
# 1. Check AnythingLLM
# ---------------------------------------------------------------------------
_s2s "Checking AnythingLLM at localhost:3001 …"
if curl -sf --max-time 3 http://localhost:3001/api/ping -o /dev/null; then
    _s2s "AnythingLLM: OK"
else
    _err "AnythingLLM not reachable at localhost:3001"
    _err "Start AnythingLLM, then re-run this script."
    _err "(Continuing anyway — LLM calls will fail until it's up.)"
fi

# ---------------------------------------------------------------------------
# 2. Build Rust daemon
# ---------------------------------------------------------------------------
_rust "Building …"
cargo build --release 2>&1 | while IFS= read -r line; do _rust "$line"; done

# ---------------------------------------------------------------------------
# 3. Activate Python venv
# ---------------------------------------------------------------------------
if [[ ! -f ".venv/bin/activate" ]]; then
    _err "Python venv not found. Run:"
    _err "  python3 -m venv .venv && source .venv/bin/activate"
    _err "  pip install -r requirements.txt -r requirements_s2s.txt"
    exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# ---------------------------------------------------------------------------
# PIDs + cleanup
# ---------------------------------------------------------------------------
S2S_PID=""
RUST_PID=""
WAKE_PID=""

cleanup() {
    echo ""
    echo "==> Shutting down …"
    [[ -n "$WAKE_PID" ]] && kill "$WAKE_PID" 2>/dev/null || true
    [[ -n "$RUST_PID" ]] && kill "$RUST_PID" 2>/dev/null || true
    [[ -n "$S2S_PID"  ]] && kill "$S2S_PID"  2>/dev/null || true
    rm -f /tmp/wakeword.sock /tmp/s2s.sock
    wait 2>/dev/null || true
    echo "==> Done."
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# 4. Start s2s_service.py — wait for /tmp/s2s.sock to appear
# ---------------------------------------------------------------------------
_s2s "Starting S2S service …"
python3 s2s_service.py ${S2S_ARGS[@]+"${S2S_ARGS[@]}"} \
    1> >(while IFS= read -r l; do printf "${BLU}[S2S]  %s${NC}\n" "$l"; done) \
    2> >(while IFS= read -r l; do printf "${BLU}[S2S]  %s${NC}\n" "$l" >&2; done) &
S2S_PID=$!

_s2s "Waiting for models to load (this can take ~30 s on first run) …"
WAITED=0
until [[ -S /tmp/s2s.sock ]]; do
    sleep 1
    WAITED=$((WAITED + 1))
    if ! kill -0 "$S2S_PID" 2>/dev/null; then
        _err "s2s_service.py exited unexpectedly — check logs above"
        exit 1
    fi
    if [[ $WAITED -ge 120 ]]; then
        _err "s2s_service.py did not become ready within 120 s"
        exit 1
    fi
done
_s2s "S2S service ready (${WAITED}s)"

# ---------------------------------------------------------------------------
# 5. Start Rust daemon
# ---------------------------------------------------------------------------
_rust "Starting daemon …"
RUST_LOG="${RUST_LOG:-info}" ./target/release/wisdom-tree \
    2> >(while IFS= read -r l; do printf "${GRN}[RUST] %s${NC}\n" "$l" >&2; done) &
RUST_PID=$!
sleep 0.5   # let the Rust daemon bind /tmp/wakeword.sock

# ---------------------------------------------------------------------------
# 6. Start remote keypress listener
# ---------------------------------------------------------------------------
_wake "Starting remote keypress listener …"
python3 keypress_listener.py \
    2> >(while IFS= read -r l; do printf "${YLW}[WAKE] %s${NC}\n" "$l" >&2; done) &
WAKE_PID=$!

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
echo ""
printf "${GRN}==> All services running. Press Ctrl-C to stop.${NC}\n"
echo "    [S2S]  PID $S2S_PID"
echo "    [RUST] PID $RUST_PID"
echo "    [WAKE] PID $WAKE_PID"
echo ""

# Poll until any one process dies, then let the EXIT trap clean everything up.
while kill -0 "$S2S_PID" 2>/dev/null \
   && kill -0 "$RUST_PID" 2>/dev/null \
   && kill -0 "$WAKE_PID" 2>/dev/null; do
    sleep 1
done

if ! kill -0 "$S2S_PID"  2>/dev/null; then _err "s2s_service.py exited"; fi
if ! kill -0 "$RUST_PID" 2>/dev/null; then _err "Rust daemon exited";    fi
if ! kill -0 "$WAKE_PID" 2>/dev/null; then _err "keypress_listener exited"; fi
