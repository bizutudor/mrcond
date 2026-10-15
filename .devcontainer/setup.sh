#!/usr/bin/env bash
# One-time setup for GitHub Codespaces / dev containers.
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -e ".[dev,reports]"
npm --prefix web ci
npm --prefix web run build
# Download the public Track R documents in the background, so the app can start right away.
# The FDA examples work without them; each manufacturer document appears on the review page once
# it has been downloaded. Progress: /tmp/mrcond-fetch.log
setsid nohup mrcond fetch >/tmp/mrcond-fetch.log 2>&1 </dev/null &
