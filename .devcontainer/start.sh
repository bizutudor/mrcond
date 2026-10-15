#!/usr/bin/env bash
# Starts the API + web UI in the background on every container start (log: /tmp/mrcond-serve.log).
# If the UI is not reachable, run `mrcond serve` in a terminal instead.
if ! curl -fs http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  setsid nohup mrcond serve --port 8000 >/tmp/mrcond-serve.log 2>&1 </dev/null &
fi
