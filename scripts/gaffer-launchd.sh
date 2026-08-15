#!/bin/bash
#
# The command launchd actually runs. Two modes:
#
#   gaffer-launchd.sh server        long-lived: the API and dashboard
#   gaffer-launchd.sh autorefresh   one shot: refresh the data if it is stale
#
# It exists so the plists stay dumb. Everything that needs a decision at run
# time -- which address to bind, where the virtualenv is, how big the log has
# grown -- is decided here, where it can be read and changed, rather than being
# frozen into an XML file.
#
# Bind address, in order of precedence:
#
#   1. $GAFFER_HOST, if set. This is the override; set it to 127.0.0.1 to make
#      the server local-only again.
#   2. The Mac's Tailscale address, if Tailscale is installed and up. This is
#      the one we want: the iPhone reaches the Mac over the tailnet and nothing
#      else can see the port at all.
#   3. 0.0.0.0, with a loud warning. This binds every interface, so the
#      dashboard is also reachable by anything else on the home wifi. There is
#      no authentication in gaffer, by design, so that is a real exposure --
#      hence the warning. Set GAFFER_REQUIRE_TAILSCALE=1 to refuse this and
#      wait for Tailscale instead.
#
# Bash 3.2 compatible: that is what /bin/bash is on macOS.

set -u
set -o pipefail

MODE="${1:-server}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
PORT="${GAFFER_PORT:-8770}"
LOG_DIR="${GAFFER_LOG_DIR:-$HOME/Library/Logs/gaffer}"
LOG_MAX_BYTES="${GAFFER_LOG_MAX_BYTES:-5242880}"   # 5 MB, then rotate once

mkdir -p "$LOG_DIR"

case "$MODE" in
  server|autorefresh) LOG_FILE="$LOG_DIR/$MODE.log" ;;
  *)
    echo "usage: $(basename "$0") server|autorefresh" >&2
    exit 64
    ;;
esac

# Rotate before launchd's redirection is replaced by ours, so the running
# process never writes into a file that has been renamed out from under it.
if [ -f "$LOG_FILE" ]; then
  SIZE="$(/usr/bin/stat -f %z "$LOG_FILE" 2>/dev/null || echo 0)"
  if [ "$SIZE" -gt "$LOG_MAX_BYTES" ] 2>/dev/null; then
    mv -f "$LOG_FILE" "$LOG_FILE.1"
  fi
fi
exec >>"$LOG_FILE" 2>&1
# fd 3 is the log, kept separate from stdout so that log lines written inside a
# `$( ... )` capture -- resolving the Tailscale address does exactly that -- end
# up in the log rather than in the captured value.
exec 3>&1

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >&3
}

if [ ! -x "$PYTHON" ]; then
  log "FATAL: no interpreter at $PYTHON. Run $PROJECT_DIR/setup.sh first."
  exit 78
fi

cd "$PROJECT_DIR" || { log "FATAL: cannot cd to $PROJECT_DIR"; exit 78; }

# ---------------------------------------------------------------------------
# autorefresh: one shot, and it decides for itself whether to do any work
# ---------------------------------------------------------------------------

if [ "$MODE" = "autorefresh" ]; then
  log "autorefresh: starting (project $PROJECT_DIR)"
  "$PYTHON" -m gaffer.cli autorefresh
  RC=$?
  log "autorefresh: exit $RC"
  exit $RC
fi

# ---------------------------------------------------------------------------
# server: work out where to bind, then hand over to uvicorn
# ---------------------------------------------------------------------------

# `tailscale ip -4` blocks if the daemon is wedged, and this runs at login when
# Tailscale may still be starting. Cap it rather than hanging the LaunchAgent.
run_with_timeout() {
  local seconds="$1"; shift
  local out; out="$(mktemp -t gaffer-ts)"
  "$@" >"$out" 2>/dev/null &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$((seconds * 10))" ]; then
      kill -9 "$pid" 2>/dev/null
      break
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  cat "$out"
  rm -f "$out"
  return $rc
}

find_tailscale() {
  local candidate
  for candidate in \
    "${TAILSCALE_BIN:-}" \
    /Applications/Tailscale.app/Contents/MacOS/Tailscale \
    /usr/local/bin/tailscale \
    /opt/homebrew/bin/tailscale \
    /usr/bin/tailscale
  do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  command -v tailscale 2>/dev/null && return 0
  return 1
}

is_ipv4() {
  echo "$1" | grep -Eq '^([0-9]{1,3}\.){3}[0-9]{1,3}$'
}

tailscale_ip() {
  local bin; bin="$(find_tailscale)" || return 1
  log "tailscale: found $bin"
  local ip
  ip="$(run_with_timeout 5 "$bin" ip -4 | head -n 1 | tr -d '[:space:]')"
  if [ -z "$ip" ]; then
    log "tailscale: 'ip -4' returned nothing (daemon down, or not logged in)"
    return 1
  fi
  if ! is_ipv4 "$ip"; then
    log "tailscale: 'ip -4' returned something that is not an IPv4 address: $ip"
    return 1
  fi
  case "$ip" in
    100.*) : ;;
    *) log "tailscale: NOTE $ip is outside the usual 100.64.0.0/10 tailnet range" ;;
  esac
  echo "$ip"
}

HOST=""
SOURCE=""

if [ -n "${GAFFER_HOST:-}" ]; then
  HOST="$GAFFER_HOST"
  SOURCE="GAFFER_HOST"
else
  IP="$(tailscale_ip)"
  if [ -n "$IP" ]; then
    HOST="$IP"
    SOURCE="tailscale"
  fi
fi

if [ -z "$HOST" ]; then
  if [ -n "${GAFFER_REQUIRE_TAILSCALE:-}" ]; then
    log "no Tailscale address and GAFFER_REQUIRE_TAILSCALE is set: refusing to"
    log "bind 0.0.0.0. Waiting 60s; launchd will start us again after that."
    sleep 60
    exit 75
  fi
  HOST="0.0.0.0"
  SOURCE="fallback"
  log "############################################################"
  log "# Tailscale is not available, so binding 0.0.0.0 instead."
  log "# That makes the dashboard reachable by EVERY device on this"
  log "# network, and gaffer has no authentication by design."
  log "# To pin it down: install Tailscale and it will be picked up"
  log "# on the next restart, or set GAFFER_HOST in the LaunchAgent"
  log "# (~/Library/LaunchAgents/com.gaffer.server.plist), or set"
  log "# GAFFER_REQUIRE_TAILSCALE=1 to refuse this fallback."
  log "############################################################"
fi

log "server: binding $HOST:$PORT (address source: $SOURCE)"
log "server: http://$HOST:$PORT/  <- open this on the iPhone"
exec "$PYTHON" -m gaffer.cli serve --host "$HOST" --port "$PORT" --log-level info
