#!/usr/bin/env bash
# Stage One: build the cluster environment (Step 1 + Step 2 of the scenario).
#
# Safe to re-run. If the cluster already exists it is reused, any nodes stopped
# by a previous chaos run are restarted, and the images/manifests are refreshed.
set -euo pipefail

CLUSTER=rca4
NS=rca4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: '$1' is not installed. $2" >&2
    exit 1
  }
}

need docker "Start Docker Desktop or OrbStack."
need kind    "Install it with: brew install kind"
need kubectl "Install it with: brew install kubectl"

docker info >/dev/null 2>&1 || {
  echo "ERROR: the Docker daemon is not reachable. Start Docker Desktop / OrbStack first." >&2
  exit 1
}

# --- 1. Create (or reuse) the five-node cluster -------------------------------
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "==> Cluster '$CLUSTER' already exists, reusing it."
  echo "==> Restarting any nodes a previous chaos run left stopped..."
  for node in $(kind get nodes --name "$CLUSTER"); do
    if [ "$(docker inspect -f '{{.State.Running}}' "$node" 2>/dev/null)" != "true" ]; then
      echo "    starting $node"
      docker start "$node" >/dev/null
    fi
  done
else
  echo "==> Creating five-node cluster '$CLUSTER' (this takes a minute or two)..."
  kind create cluster --config "$ROOT/k8s/kind-cluster.yaml"
fi

kubectl config use-context "kind-$CLUSTER" >/dev/null

echo "==> Waiting for all nodes to report Ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=180s

# --- 2. Build the application image and side-load it into the nodes ----------
echo "==> Building rca4-app:latest..."
docker build -q -t rca4-app:latest "$ROOT/app"

echo "==> Loading the image into the cluster nodes..."
kind load docker-image rca4-app:latest --name "$CLUSTER"

# --- 3. Deploy Jaeger, the backend replicas and the gateway ------------------
echo "==> Applying manifests..."
kubectl apply -f "$ROOT/k8s/00-namespace.yaml"
kubectl apply -f "$ROOT/k8s/10-jaeger.yaml"
kubectl apply -f "$ROOT/k8s/20-order-backend.yaml"
kubectl apply -f "$ROOT/k8s/30-api-gateway.yaml"

# A previous run may have left a replacement pod built from an older image.
kubectl -n "$NS" rollout restart deployment/order-backend deployment/api-gateway >/dev/null

echo "==> Waiting for rollouts..."
kubectl -n "$NS" rollout status deployment/jaeger --timeout=180s
kubectl -n "$NS" rollout status deployment/order-backend --timeout=180s
kubectl -n "$NS" rollout status deployment/api-gateway --timeout=180s

echo
echo "==> Replica placement (each replica must be on its own node):"
kubectl -n "$NS" get pods -l app=order-backend \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,STATUS:.status.phase' --no-headers

echo
echo "==> Smoke test through the load balancer:"
for i in 1 2 3 4 5 6; do
  curl -s --max-time 5 http://127.0.0.1:30080/order \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print("    served by", d["upstream"]["pod"], "on", d["upstream"]["node"])' \
    || echo "    request $i failed"
done

cat <<EOF

=====================================================================
Cluster is up.

  API gateway : http://localhost:30080/order
  Jaeger UI   : http://localhost:30686   (service: api-gateway)

Next: run the experiment.

  python3 scripts/load_generator.py --duration 300 --rps 42 --chaos-at 60

=====================================================================
EOF
