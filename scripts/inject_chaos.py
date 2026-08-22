"""Step 3: inject the fault - the "sniper" shot.

Two modes:

  --mode node  (default)  Hard-kills the Docker container backing one of the
                          kind worker nodes that is hosting an order-backend
                          replica. This is the real scenario: the node vanishes
                          without warning, exactly like a yanked power cable.
                          Kubernetes has to *detect* the loss first (~40s of
                          node-monitor grace period), during which the load
                          balancer keeps sending traffic into a black hole.

  --mode pod              Force-deletes one replica pod, leaving the node alive.
                          Kubernetes notices immediately, so recovery is much
                          faster and the "detection" phase is nearly absent.
                          Useful as a contrast run.

The exact moment of the fault is written to the run directory, because every
downstream feature (`time_since_fault`) and the training label are measured
relative to it.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

NAMESPACE = "rca4"
CLUSTER = "rca4"


def run(cmd):
    """Run a command and return stdout, raising with useful context on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def backend_pods():
    """Return [(pod_name, node_name, ready), ...] for the order-backend replicas."""
    raw = run(
        [
            "kubectl", "-n", NAMESPACE, "get", "pods",
            "-l", "app=order-backend",
            "-o", "json",
        ]
    )
    pods = []
    for item in json.loads(raw).get("items", []):
        name = item["metadata"]["name"]
        node = item["spec"].get("nodeName", "unscheduled")
        statuses = item.get("status", {}).get("containerStatuses") or []
        ready = bool(statuses) and all(s.get("ready") for s in statuses)
        pods.append((name, node, ready))
    return pods


def pick_target(pods, prefer=None):
    """Choose a Ready replica to destroy."""
    ready = [p for p in pods if p[2]]
    if not ready:
        raise RuntimeError("no Ready order-backend replicas found - is the cluster healthy?")

    if prefer:
        for pod, node, _ in ready:
            if prefer in (pod, node):
                return pod, node
        raise RuntimeError(f"'{prefer}' is not a Ready order-backend pod or its node")

    return random.choice(ready)[:2]


def main():
    parser = argparse.ArgumentParser(description="Terminate one order-backend replica or its node.")
    parser.add_argument("--mode", choices=["node", "pod"], default="node",
                        help="kill the whole node (default) or just the pod")
    parser.add_argument("--target", default=None,
                        help="specific pod or node name to kill (default: pick one at random)")
    parser.add_argument("--run-dir", default=None,
                        help="run directory to record the fault event into")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be killed and exit")
    args = parser.parse_args()

    pods = backend_pods()
    print("Current order-backend replicas:")
    for pod, node, ready in pods:
        print(f"  {pod:<34} node={node:<18} ready={ready}")

    target_pod, target_node = pick_target(pods, args.target)

    if args.dry_run:
        print(f"\n[dry run] would kill {args.mode}: pod={target_pod} node={target_node}")
        return

    print(f"\n>>> SNIPER SHOT: killing {args.mode} "
          f"({'node ' + target_node if args.mode == 'node' else 'pod ' + target_pod})")

    fault_at = time.time()
    if args.mode == "node":
        # `docker kill` rather than `stop`: no SIGTERM, no graceful shutdown.
        # The node simply ceases to exist from the cluster's point of view.
        run(["docker", "kill", target_node])
        detail = f"node container {target_node} hard-killed (hosted {target_pod})"
    else:
        run([
            "kubectl", "-n", NAMESPACE, "delete", "pod", target_pod,
            "--grace-period=0", "--force",
        ])
        detail = f"pod {target_pod} force-deleted on node {target_node}"

    print(f">>> {detail}")
    print(f">>> fault_at = {fault_at:.3f} ({time.strftime('%H:%M:%S', time.localtime(fault_at))})")

    event = {
        "mode": args.mode,
        "target_pod": target_pod,
        "target_node": target_node,
        "fault_at": fault_at,
        "detail": detail,
        "replicas_before": [{"pod": p, "node": n, "ready": r} for p, n, r in pods],
    }

    if args.run_dir:
        os.makedirs(args.run_dir, exist_ok=True)
        path = os.path.join(args.run_dir, "chaos_event.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(event, handle, indent=2)
        print(f">>> event recorded to {path}")
    else:
        print(json.dumps(event, indent=2))

    if args.mode == "node":
        print("\nRemember: the node stays powered off until you run "
              "scripts/reset_cluster.sh (or cluster_up.sh).")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
