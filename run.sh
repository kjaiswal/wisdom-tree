#!/usr/bin/env bash
# run.sh — build and launch wisdom-tree (daemon + keypress sidecar only)
#
# Usage:
#   ./run.sh
#
# Prerequisites:
#   - Rust toolchain (cargo)
#   - Python venv at .venv with requirements installed
#     python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
#   - macOS Input Monitoring permission for your terminal app

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------------------------------------------------------- #
# 1. Build Rust daemon
# --------------------------------------------------------------------------- #
echo "==> Building Rust daemon …"
cargo build --release 2>&1

DAEMON_BIN="$SCRIPT_DIR/target/release/wisdom-tree"

# --------------------------------------------------------------------------- #
# 2. Activate Python venv
# --------------------------------------------------------------------------- #
if [[ ! -f ".venv/bin/activate" ]]; then
  echo "ERROR: Python venv not found. Run:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# --------------------------------------------------------------------------- #
# 3. Cleanup on exit (Ctrl-C or normal exit)
# --------------------------------------------------------------------------- #
DAEMON_PID=""
SIDECAR_PID=""

cleanup() {
  echo ""
  echo "==> Shutting down …"
  [[ -n "$SIDECAR_PID" ]] && kill "$SIDECAR_PID" 2>/dev/null || true
  [[ -n "$DAEMON_PID"  ]] && kill "$DAEMON_PID"  2>/dev/null || true
  rm -f /tmp/wakeword.sock
  wait 2>/dev/null || true
  echo "==> Done."
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# 4. Start Rust daemon
# --------------------------------------------------------------------------- #
echo "==> Starting Rust daemon …"
RUST_LOG="${RUST_LOG:-wisdom_tree=debug,info}" "$DAEMON_BIN" &
DAEMON_PID=$!

# Give the daemon a moment to bind the socket.
sleep 0.5

# --------------------------------------------------------------------------- #
# 5. Start keypress listener sidecar
# --------------------------------------------------------------------------- #
echo "==> Starting keypress listener …"
python3 keypress_listener.py &
SIDECAR_PID=$!

echo "==> Both processes running. Press play/pause on your Satechi remote to activate."
echo "    Daemon PID : $DAEMON_PID"
echo "    Sidecar PID: $SIDECAR_PID"

while kill -0 "$DAEMON_PID" 2>/dev/null && kill -0 "$SIDECAR_PID" 2>/dev/null; do
  sleep 1
done

if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
  echo "==> Rust daemon exited — stopping."
else
  echo "==> Python sidecar exited — stopping."
fi
