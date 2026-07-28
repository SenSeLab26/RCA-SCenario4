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
