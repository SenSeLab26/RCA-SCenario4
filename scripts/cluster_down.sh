#!/usr/bin/env bash
# Tear the whole cluster down and free ports 30080 / 30686.
set -euo pipefail

CLUSTER=rca4

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "==> Deleting cluster '$CLUSTER'..."
  kind delete cluster --name "$CLUSTER"
  echo "==> Done. Collected run data under runs/ and data/ is untouched."
else
  echo "Cluster '$CLUSTER' does not exist - nothing to do."
fi
