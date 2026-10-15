#!/usr/bin/env bash
# One-time setup for GitHub Codespaces / dev containers.
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -e ".[dev,reports]"
npm --prefix web ci
npm --prefix web run build
# Download the public Track R documents for the review page. The FDA examples work without them,
# so a site that is unreachable is reported but does not stop the setup.
mrcond fetch || echo "Some manufacturer documents could not be downloaded; the FDA examples still work."
