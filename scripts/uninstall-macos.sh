#!/bin/bash
#
# Remove the two gaffer LaunchAgents. Idempotent: running it twice, or running
# it when nothing is installed, is a no-op that still exits 0.
#
# Data (data/cache) and logs are left alone by default -- 35 MB of cache is
# expensive to rebuild and the logs are the only record of what happened. Pass
# --logs to delete the logs too.
#
# Bash 3.2 compatible: that is what /bin/bash is on macOS.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENT_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/gaffer"
UID_NUM="$(id -u)"

LABELS="com.gaffer.server com.gaffer.autorefresh"
REMOVE_LOGS=0

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
say()  { printf '%s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --logs) REMOVE_LOGS=1; shift ;;
    -h|--help)
      cat <<USAGE
usage: $(basename "$0") [--logs]

Unloads and deletes the gaffer LaunchAgents. Safe to re-run.
  --logs   also delete $LOG_DIR
USAGE
      exit 0
      ;;
    *) say "unknown argument: $1" >&2; exit 64 ;;
  esac
done

bold "removing gaffer LaunchAgents"

for label in $LABELS; do
  plist="$AGENT_DIR/$label.plist"
  was_loaded=0
  launchctl print "gui/$UID_NUM/$label" >/dev/null 2>&1 && was_loaded=1

  launchctl bootout "gui/$UID_NUM/$label" >/dev/null 2>&1
  [ -f "$plist" ] && launchctl unload -w "$plist" >/dev/null 2>&1

  i=0
  while launchctl print "gui/$UID_NUM/$label" >/dev/null 2>&1; do
    i=$((i + 1))
    [ "$i" -gt 25 ] && break
    sleep 0.2
  done

  if launchctl print "gui/$UID_NUM/$label" >/dev/null 2>&1; then
    say "  $label: still loaded -- try again, or log out and back in"
  elif [ "$was_loaded" -eq 1 ]; then
    say "  $label: unloaded"
  else
    say "  $label: was not loaded"
  fi

  if [ -f "$plist" ]; then
    rm -f "$plist" && say "  $label: removed $plist"
  else
    say "  $label: no plist at $plist"
  fi
done

# A server started outside launchd (or one launchd has lost track of) would
# keep the port. Say so rather than leaving a mystery process behind.
if pgrep -f "gaffer.cli serve" >/dev/null 2>&1; then
  say ""
  say "note: a 'gaffer.cli serve' process is still running. If you did not start"
  say "      it by hand, stop it with:  pkill -f 'gaffer.cli serve'"
fi

if [ "$REMOVE_LOGS" -eq 1 ]; then
  rm -rf "$LOG_DIR" && say "  logs: removed $LOG_DIR"
else
  say ""
  say "logs kept at $LOG_DIR (pass --logs to delete them)"
fi
say "data kept at $PROJECT_DIR/data"
say ""
say "reinstall with: \"$SCRIPT_DIR/install-macos.sh\""
