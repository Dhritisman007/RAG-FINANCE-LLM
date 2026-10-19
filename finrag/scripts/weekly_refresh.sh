#!/bin/bash
# Weekly SEC filing refresh — run by launchd (com.finrag.refresh) on Mondays at 09:00.
#   ./weekly_refresh.sh            download new filings and rebuild the store
#   ./weekly_refresh.sh --dry-run  only report what's new
# Logs to finrag/data/logs/refresh.log and posts a macOS notification.

FINRAG="/Volumes/Crucial X9/RAG LLM/finrag"
PYTHON="/opt/anaconda3/bin/python3"
LOG="$FINRAG/data/logs/refresh.log"

notify() { /usr/bin/osascript -e "display notification \"$1\" with title \"FinRAG filing refresh\""; }

if [ ! -d "$FINRAG" ]; then   # external drive not plugged in: nothing to do (and nowhere to log)
  notify "Skipped: the Crucial X9 drive isn't connected."
  exit 0
fi
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "===== $(date '+%Y-%m-%d %H:%M') refresh $* ====="
cd "$FINRAG" || exit 1

if [ "$1" = "--dry-run" ]; then
  out=$("$PYTHON" scripts/refresh_filings.py 2>&1); echo "$out"
  notify "Dry run: $(echo "$out" | grep -E 'new filing|Up to date' | head -1)"
  exit 0
fi

# Rebuilding the store under a running evaluation would change what it retrieves mid-run.
REBUILD="--rebuild"
if pgrep -f "scripts/evaluate.py" >/dev/null; then
  REBUILD=""
  echo "An evaluation is running — downloading only; rebuild later with: python scripts/refresh_filings.py --rebuild"
fi

out=$(/usr/bin/caffeinate -i "$PYTHON" scripts/refresh_filings.py --apply $REBUILD 2>&1); status=$?
echo "$out"
summary=$(echo "$out" | grep -E "new filing|Up to date|Downloaded" | head -2 | tr '\n' ' ')
if [ $status -ne 0 ]; then
  notify "Failed (exit $status) — see data/logs/refresh.log"
elif echo "$out" | grep -q "now has"; then
  notify "$summary— store rebuilt. Restart the UI to search the new filings."
else
  notify "${summary:-Done.}"
fi
