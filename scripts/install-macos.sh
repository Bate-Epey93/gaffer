#!/bin/bash
#
# Install gaffer as two launchd LaunchAgents:
#
#   com.gaffer.server       the API and dashboard, started at login, restarted
#                           if it dies, bound where the iPhone can reach it
#   com.gaffer.autorefresh  hourly; refreshes the FPL data only when it has
#                           actually gone stale
#
# Safe to re-run: it rewrites the plists, unloads whatever was there and loads
# the new ones. Re-running is in fact the way to change the port.
#
# Everything is written with absolute paths, and this project lives under a
# directory with a space in its name, so every expansion here is quoted.
#
# Bash 3.2 compatible: that is what /bin/bash is on macOS.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
RUNNER="$SCRIPT_DIR/gaffer-launchd.sh"

AGENT_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/gaffer"
UID_NUM="$(id -u)"

SERVER_LABEL="com.gaffer.server"
REFRESH_LABEL="com.gaffer.autorefresh"
SERVER_PLIST="$AGENT_DIR/$SERVER_LABEL.plist"
REFRESH_PLIST="$AGENT_DIR/$REFRESH_LABEL.plist"

PORT="${GAFFER_PORT:-8770}"
INTERVAL="${GAFFER_REFRESH_INTERVAL:-3600}"
FORCE_HOST="${GAFFER_HOST:-}"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
say()   { printf '%s\n' "$*"; }
warn()  { printf '\033[33m%s\033[0m\n' "$*"; }
die()   { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  cat <<USAGE
usage: $(basename "$0") [--port N] [--host ADDR] [--interval SECONDS]

  --port N          port for the dashboard (default $PORT)
  --host ADDR       pin the bind address instead of auto-detecting it.
                    Use 127.0.0.1 to make the server local-only again.
                    Leave unset to prefer the Tailscale address and fall
                    back to 0.0.0.0.
  --interval SEC    how often launchd runs autorefresh (default $INTERVAL)

Re-run at any time; it replaces whatever is installed.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --port)     PORT="${2:-}";      shift 2 || die "--port needs a value" ;;
    --host)     FORCE_HOST="${2:-}"; shift 2 || die "--host needs a value" ;;
    --interval) INTERVAL="${2:-}";  shift 2 || die "--interval needs a value" ;;
    -h|--help)  usage; exit 0 ;;
    *)          usage >&2; die "unknown argument: $1" ;;
  esac
done

echo "$PORT"     | grep -Eq '^[0-9]+$'  || die "--port must be a number, got '$PORT'"
echo "$INTERVAL" | grep -Eq '^[0-9]+$'  || die "--interval must be a number, got '$INTERVAL'"
[ "$INTERVAL" -ge 60 ] || die "--interval below 60s would hammer the FPL API"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

bold "gaffer -> launchd"
say  "project   $PROJECT_DIR"

[ -x "$PYTHON" ] || die "no interpreter at $PYTHON -- run $PROJECT_DIR/setup.sh first"
[ -f "$RUNNER" ] || die "missing $RUNNER"
chmod +x "$RUNNER" 2>/dev/null || true
( cd "$PROJECT_DIR" && "$PYTHON" -c "import gaffer" >/dev/null 2>&1 ) \
  || die "$PYTHON cannot import gaffer from $PROJECT_DIR"

mkdir -p "$AGENT_DIR" "$LOG_DIR" || die "could not create $AGENT_DIR / $LOG_DIR"
say  "python    $PYTHON"
say  "logs      $LOG_DIR"

# ---------------------------------------------------------------------------
# Writing the plists
# ---------------------------------------------------------------------------

# The project path is user-controlled text going into XML. Escape it.
xml_escape() {
  printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

X_RUNNER="$(xml_escape "$RUNNER")"
X_PROJECT="$(xml_escape "$PROJECT_DIR")"
X_LOGDIR="$(xml_escape "$LOG_DIR")"

# An optional <key>/<string> pair, emitted only when the value is non-empty.
host_entry=""
if [ -n "$FORCE_HOST" ]; then
  host_entry="        <key>GAFFER_HOST</key>
        <string>$(xml_escape "$FORCE_HOST")</string>
"
fi

write_plist() {
  local target="$1" body="$2" tmp
  tmp="$(mktemp -t gaffer-plist)" || die "mktemp failed"
  printf '%s' "$body" > "$tmp"
  if ! plutil -lint "$tmp" >/dev/null; then
    plutil -lint "$tmp" >&2
    rm -f "$tmp"
    die "generated plist for $target is not valid"
  fi
  mv -f "$tmp" "$target" || die "could not write $target"
  chmod 644 "$target"
}

SERVER_BODY='<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>'"$SERVER_LABEL"'</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>'"$X_RUNNER"'</string>
        <string>server</string>
    </array>
    <key>WorkingDirectory</key>
    <string>'"$X_PROJECT"'</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin</string>
        <key>GAFFER_PORT</key>
        <string>'"$PORT"'</string>
        <key>GAFFER_LOG_DIR</key>
        <string>'"$X_LOGDIR"'</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
'"$host_entry"'    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>ProcessType</key>
    <string>Interactive</string>
    <key>StandardOutPath</key>
    <string>'"$X_LOGDIR"'/server.launchd.log</string>
    <key>StandardErrorPath</key>
    <string>'"$X_LOGDIR"'/server.launchd.log</string>
</dict>
</plist>
'

REFRESH_BODY='<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>'"$REFRESH_LABEL"'</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>'"$X_RUNNER"'</string>
        <string>autorefresh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>'"$X_PROJECT"'</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin</string>
        <key>GAFFER_LOG_DIR</key>
        <string>'"$X_LOGDIR"'</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>StartInterval</key>
    <integer>'"$INTERVAL"'</integer>
    <key>ProcessType</key>
    <string>Background</string>
    <key>LowPriorityIO</key>
    <true/>
    <key>StandardOutPath</key>
    <string>'"$X_LOGDIR"'/autorefresh.launchd.log</string>
    <key>StandardErrorPath</key>
    <string>'"$X_LOGDIR"'/autorefresh.launchd.log</string>
</dict>
</plist>
'

write_plist "$SERVER_PLIST"  "$SERVER_BODY"
write_plist "$REFRESH_PLIST" "$REFRESH_BODY"
say "plists    $SERVER_PLIST"
say "          $REFRESH_PLIST  (every ${INTERVAL}s)"

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

unload_agent() {
  local label="$1" plist="$2" i=0
  launchctl bootout "gui/$UID_NUM/$label" >/dev/null 2>&1
  [ -f "$plist" ] && launchctl unload -w "$plist" >/dev/null 2>&1
  # bootout is asynchronous; wait for the job to actually go away so that the
  # bootstrap below does not race it and fail with "service already loaded".
  while launchctl print "gui/$UID_NUM/$label" >/dev/null 2>&1; do
    i=$((i + 1))
    [ "$i" -gt 50 ] && break
    sleep 0.2
  done
}

load_agent() {
  local label="$1" plist="$2" err
  err="$(launchctl bootstrap "gui/$UID_NUM" "$plist" 2>&1)" && return 0
  # Older systems, and some sandboxes, only have the deprecated verb.
  if launchctl load -w "$plist" >/dev/null 2>&1; then return 0; fi
  die "could not load $label: $err"
}

# Where server.log ends right now. The log survives reinstalls, so the "which
# address did it bind" line has to be read from what this run appends, not from
# whatever the last run left behind.
LOG_MARK=0
[ -f "$LOG_DIR/server.log" ] && \
  LOG_MARK="$(/usr/bin/stat -f %z "$LOG_DIR/server.log" 2>/dev/null || echo 0)"

unload_agent "$SERVER_LABEL"  "$SERVER_PLIST"
unload_agent "$REFRESH_LABEL" "$REFRESH_PLIST"
load_agent   "$SERVER_LABEL"  "$SERVER_PLIST"
load_agent   "$REFRESH_LABEL" "$REFRESH_PLIST"
launchctl kickstart -k "gui/$UID_NUM/$SERVER_LABEL" >/dev/null 2>&1

# ---------------------------------------------------------------------------
# Did it come up?
# ---------------------------------------------------------------------------

say ""
printf 'starting the server '
BOUND=""
HEALTH=""
i=0
while [ "$i" -lt 90 ]; do
  if [ -z "$BOUND" ] && [ -f "$LOG_DIR/server.log" ]; then
    BOUND="$(tail -c "+$((LOG_MARK + 1))" "$LOG_DIR/server.log" 2>/dev/null \
             | grep 'server: binding' | tail -n 1 \
             | sed -e 's/.*binding //' -e 's/ .*//')"
  fi
  if [ -n "$BOUND" ]; then
    # /api/health answers "loading" while the first projection set is being
    # built, so waiting for the first *response* would call it done too early.
    HEALTH="$(curl -s -m 3 "http://127.0.0.1:$PORT/api/health" 2>/dev/null \
              | head -c 20000 | sed -n 's/.*"status":"\([a-z]*\)".*/\1/p' | head -n 1)"
    [ "$HEALTH" = "ok" ] && break
  fi
  printf '.'
  i=$((i + 1))
  sleep 1
done
printf '\n'

HOST_PART="${BOUND%%:*}"
[ -n "$HOST_PART" ] || HOST_PART="0.0.0.0"

if [ "$HEALTH" = "ok" ]; then
  bold "server is up and healthy on $BOUND"
elif [ -n "$HEALTH" ]; then
  warn "the server is answering but reports '$HEALTH', not 'ok'."
  warn "It is probably still building its first projection set. Watch it with:"
  warn "  tail -f \"$LOG_DIR/server.log\""
else
  warn "the server has not answered http://127.0.0.1:$PORT/api/health yet."
  warn "Give it a moment, then check:"
  warn "  tail -f \"$LOG_DIR/server.log\""
  warn "  launchctl print gui/$UID_NUM/$SERVER_LABEL | head -30"
fi

# The address to type into the phone. When bound to 0.0.0.0 the wildcard is not
# a usable address, so offer the real ones.
say ""
bold "Open this on the iPhone"
if [ "$HOST_PART" = "0.0.0.0" ]; then
  for iface in en0 en1 en2; do
    ip="$(ipconfig getifaddr "$iface" 2>/dev/null)"
    [ -n "$ip" ] && say "  http://$ip:$PORT/      (this Mac on $iface)"
  done
  say "  http://$(hostname -s).local:$PORT/"
elif [ "$HOST_PART" = "127.0.0.1" ] || [ "$HOST_PART" = "localhost" ]; then
  say "  http://127.0.0.1:$PORT/   -- LOCAL ONLY. The phone cannot reach this."
  say "  Re-run without --host 127.0.0.1 to make it reachable."
else
  say "  http://$HOST_PART:$PORT/"
fi
say ""
say "In Safari: Share > Add to Home Screen. It then opens full screen, with its"
say "own icon, like an installed app."

case "$HOST_PART" in
  100.*)
    say ""
    bold "Bound to the Tailscale address."
    say "Only devices on your tailnet can see it. This is the intended setup."
    ;;
  0.0.0.0)
    say ""
    warn "NOTE: bound to 0.0.0.0 because Tailscale was not found."
    warn "The dashboard is reachable by every device on this wifi, and gaffer"
    warn "has no login by design. To close that:"
    warn "  - install Tailscale, then: launchctl kickstart -k gui/$UID_NUM/$SERVER_LABEL"
    warn "  - or re-run: $SCRIPT_DIR/install-macos.sh --host 127.0.0.1   (local only)"
    ;;
esac

# ---------------------------------------------------------------------------
# The bit everyone forgets
# ---------------------------------------------------------------------------

say ""
bold "The Mac must be awake for the phone to reach it"
say "A sleeping Mac answers nothing. The display can sleep; the Mac cannot."
say "  System Settings > Lock Screen        -- display sleep is fine, leave it"
say "  System Settings > Battery > Options  -- turn ON 'Prevent automatic sleeping"
say "                                          on power adapter when the display"
say "                                          is off' (desktops: Energy Saver >"
say "                                          'Prevent automatic sleeping when"
say "                                          the display is off')"
say "  Same panel                           -- turn ON 'Wake for network access'"
say "From the terminal instead:  sudo pmset -c sleep 0 womp 1   (check: pmset -g)"

say ""
bold "Commands"
say "  status        launchctl list | grep gaffer"
say "  restart       launchctl kickstart -k gui/$UID_NUM/$SERVER_LABEL"
say "  refresh now   launchctl kickstart gui/$UID_NUM/$REFRESH_LABEL"
say "                (or: \"$PYTHON\" -m gaffer.cli autorefresh --force)"
say "  what would    \"$PYTHON\" -m gaffer.cli autorefresh --dry-run"
say "  it do?"
say "  server log    tail -f \"$LOG_DIR/server.log\""
say "  refresh log   tail -f \"$LOG_DIR/autorefresh.log\""
say "  remove        \"$SCRIPT_DIR/uninstall-macos.sh\""
say ""
say "Refreshes run every ${INTERVAL}s and do nothing unless the data is older"
say "than 24h -- or older than 2h when the next deadline is inside 12h."
say "The backtest is never run on a timer; start it by hand."
