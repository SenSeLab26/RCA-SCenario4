#!/usr/bin/env bash
# Put the cluster back to a healthy 3-replicas-on-3-nodes baseline after a
# chaos run, without rebuilding anything. Use this between repeat experiments.
set -euo pipefail

CLUSTER=rca4
NS=rca4

echo "==> Restarting any stopped nodes..."
for node in $(kind get nodes --name "$CLUSTER"); do
  if [ "$(docker inspect -f '{{.State.Running}}' "$node" 2>/dev/null)" != "true" ]; then
    echo "    starting $node"
    docker start "$node" >/dev/null
  fi
done

echo "==> Waiting for nodes to report Ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=180s

# --- Undo the faults that do not undo themselves ----------------------------

# network_partition removes the Service selector label from a pod. That also
# orphans it from its ReplicaSet, so the pod keeps running forever while a
# replacement is created alongside it. Delete any such stragglers.
ORPHANS=$(kubectl -n "$NS" get pods -l '!app' -o name 2>/dev/null || true)
if [ -n "$ORPHANS" ]; then
  echo "==> Removing pods orphaned by a partition fault:"
  echo "$ORPHANS" | sed 's/^/    /'
  echo "$ORPHANS" | xargs -r kubectl -n "$NS" delete --grace-period=0 --force >/dev/null 2>&1
fi

# config_error repoints the gateway with `kubectl set env`. Clearing it returns
# the Deployment to the value baked into the manifest.
if kubectl -n "$NS" set env deployment/api-gateway --list 2>/dev/null | grep -q "^BACKEND_URL="; then
  echo "==> Clearing the gateway BACKEND_URL override left by config_error..."
  kubectl -n "$NS" set env deployment/api-gateway BACKEND_URL- >/dev/null
  kubectl -n "$NS" apply -f "$(dirname "$0")/../k8s/30-api-gateway.yaml" >/dev/null
fi

# dependency_down scales the backend to zero.
CURRENT=$(kubectl -n "$NS" get deployment order-backend -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 3)
if [ "$CURRENT" != "3" ]; then
  echo "==> Restoring order-backend to 3 replicas (was $CURRENT)..."
  kubectl -n "$NS" scale deployment/order-backend --replicas=3 >/dev/null
fi

# Re-spread the replicas. After a chaos run they sit on whichever nodes were
# alive at the time; a fresh rollout redistributes them via the anti-affinity
# rule so the next experiment starts from the same geometry.
echo "==> Redistributing order-backend replicas..."
kubectl -n "$NS" rollout restart deployment/order-backend >/dev/null
kubectl -n "$NS" rollout status deployment/order-backend --timeout=180s

echo
kubectl -n "$NS" get pods -l app=order-backend \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,READY:.status.containerStatuses[0].ready' --no-headers
echo
echo "==> Baseline restored. Ready for another run."
