"""Unified fault injector for the consolidated scenarios.


None of these are simulations. The memory leak really allocates memory the
process never frees until the kernel OOMKills the container. The CPU saturation
really burns cycles until the container's CPU limit throttles it. The partition
really removes a pod from the load balancer while leaving it running and
healthy. The faults are genuine; only the workload is synthetic.

    FAULT                SCENARIO  MECHANISM
    memory_leak          1         backend retains memory -> real OOMKill
    cpu_saturation       1         backend burns CPU -> real CFS throttling
    dependency_slow      2         backend service time grows -> gateway 504s
    dependency_down      2         scale the backend to zero replicas
    config_error         2         point the gateway at a hostname that does not exist
    network_partition    3         drop the pod's Service label: alive but unreachable
    pod_failure          4         force-delete a replica
    node_failure         4         docker kill the node container

Usage:
    python3 scripts/inject_fault.py --fault dependency_down --run-dir runs/<id>
    python3 scripts/inject_fault.py --list
    python3 scripts/inject_fault.py --fault memory_leak --dry-run
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.parse

NAMESPACE = "rca4"
BACKEND_LABEL = "app=order-backend"

# Which consolidated scenario each fault belongs to, and how it is described in
# the dataset. This mapping is what turns four experiments into one labelled
# dataset: `fault_type` becomes the target of the multi-class RCA classifier.
FAULTS = {
    "memory_leak":       {"scenario": 1, "summary": "backend leaks memory until the kernel OOMKills it"},
    "cpu_saturation":    {"scenario": 1, "summary": "backend burns CPU until the container limit throttles it"},
    "dependency_slow":   {"scenario": 2, "summary": "backend slows down until the gateway times out"},
    "dependency_down":   {"scenario": 2, "summary": "backend scaled to zero: connection refused"},
    "config_error":      {"scenario": 2, "summary": "gateway pointed at a hostname that does not resolve"},
    "network_partition": {"scenario": 3, "summary": "replica alive and healthy but removed from the load balancer"},
    "pod_failure":       {"scenario": 4, "summary": "one replica force-deleted"},
    "node_failure":      {"scenario": 4, "summary": "the node hosting a replica is killed outright"},
}


def run(cmd, check=True):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout.strip()


def backend_pods():
    """Return [(pod, node, ready), ...] for the backend replicas."""
    raw = run(["kubectl", "-n", NAMESPACE, "get", "pods", "-l", BACKEND_LABEL, "-o", "json"])
    pods = []
    for item in json.loads(raw).get("items", []):
        statuses = item.get("status", {}).get("containerStatuses") or []
        pods.append((
            item["metadata"]["name"],
            item["spec"].get("nodeName", "unscheduled"),
            bool(statuses) and all(s.get("ready") for s in statuses),
        ))
    return pods


def pick_ready(pods, prefer=None):
    ready = [p for p in pods if p[2]]
    if not ready:
        raise RuntimeError("no Ready backend replicas found - is the cluster healthy?")
    if prefer:
        for pod, node, _ in ready:
            if prefer in (pod, node):
                return pod, node
        raise RuntimeError(f"'{prefer}' is not a Ready backend pod or its node")
    return random.choice(ready)[:2]


def pod_control(pod, params):
    """Set fault parameters on one replica through its /control endpoint.

    Done over `kubectl exec` rather than by changing environment variables,
    because changing env restarts the pod and would destroy the healthy
    baseline that must sit in front of the fault in the dataset.
    """
    query = urllib.parse.urlencode(params)
    code = (
        "import urllib.request;"
        f"print(urllib.request.urlopen('http://localhost:8000/control?{query}').read().decode())"
    )
    return run(["kubectl", "-n", NAMESPACE, "exec", pod, "--", "python", "-c", code])


# --- The faults --------------------------------------------------------------

def fault_memory_leak(args, pods):
    """Scenario 1: a real leak, ended by a real OOMKill.

    Applied to every replica so the whole tier degrades together, which is what
    a code-level leak actually looks like.
    """
    targets = [p for p, _, ready in pods if ready]
    for pod in targets:
        pod_control(pod, {"leak_mb_per_sec": args.leak_mb_per_sec})
    return {"targets": targets,
            "detail": f"leaking {args.leak_mb_per_sec} MB/s on {len(targets)} replicas"}


def fault_cpu_saturation(args, pods):
    """Scenario 1: real CPU burn, throttled by the container's CPU limit."""
    targets = [p for p, _, ready in pods if ready]
    for pod in targets:
        pod_control(pod, {"cpu_burn_ms": args.cpu_ms})
    return {"targets": targets,
            "detail": f"burning {args.cpu_ms} ms of CPU per request on {len(targets)} replicas"}


def fault_dependency_slow(args, pods):
    """Scenario 2: the backend slows until the gateway's timeout fires.

    This is the report's cascading timeout, with one difference: the queue is
    real. The backend still reports success while the gateway gives up on it,
    which is exactly the symptom-versus-cause split the RCA model must resolve.
    """
    targets = [p for p, _, ready in pods if ready]
    for pod in targets:
        pod_control(pod, {"extra_delay_ms": args.delay_ms})
    return {"targets": targets,
            "detail": f"adding {args.delay_ms} ms to every request on {len(targets)} replicas"}


def fault_dependency_down(args, pods):
    """Scenario 2: the dependency disappears entirely."""
    run(["kubectl", "-n", NAMESPACE, "scale", "deployment/order-backend", "--replicas=0"])
    return {"targets": [p for p, _, _ in pods],
            "detail": "order-backend scaled to 0 replicas; every endpoint removed"}


def fault_config_error(args, pods):
    """Scenario 2: a deploy that points the gateway at the wrong port.

    A wrong *port* rather than a wrong *hostname*, and the reason is a measured
    finding worth reporting on its own.

    We first tried a hostname that does not resolve. It never reached a single
    user. The new gateway Pod could not resolve the name, so it never passed its
    readiness probe, so the rolling update never completed, so Kubernetes kept
    the old working Pod serving traffic. The bad configuration was contained
    entirely by the readiness gate. That is correct and desirable behaviour, but
    it means the fault produces no observable incident.

    A wrong port behaves differently and is just as realistic a typo. The
    hostname resolves, so the Pod starts and becomes Ready, and then every
    request fails at connect time. Because the failure is a refused connection
    rather than a timeout, the errors are fast - which is exactly what separates
    this fault from dependency_down, where requests hang for the full timeout.
    """
    bad_url = f"http://localhost:{args.bad_port}/order"
    run(["kubectl", "-n", NAMESPACE, "set", "env", "deployment/api-gateway",
         f"BACKEND_URL={bad_url}"])
    run(["kubectl", "-n", NAMESPACE, "rollout", "status",
         "deployment/api-gateway", "--timeout=90s"], check=False)
    return {"targets": ["api-gateway"],
            "detail": f"gateway BACKEND_URL repointed to {bad_url} (wrong port)"}


def fault_network_partition(args, pods):
    """Scenario 3: the replica is alive and healthy, but unreachable.

    Removing the Service's selector label takes the pod out of the load balancer
    without touching the process. The pod keeps passing its own health checks and
    has no idea it has been isolated, which is the asymmetry that makes a
    partition different from a crash.
    """
    pod, node = pick_ready(pods, args.target)
    # Removing the Service selector label was tried first. It did not work:
    # taking the label off orphans the Pod from its ReplicaSet, which
    # immediately creates a replacement, so capacity never actually drops and
    # the partition is invisible. Failing only the readiness probe keeps the Pod
    # inside its ReplicaSet, so no replacement appears and capacity really falls.
    pod_control(pod, {"fail_healthz": 1})
    return {"targets": [pod],
            "detail": f"{pod} on {node} removed from the Service while still running"}


def fault_pod_failure(args, pods):
    """Scenario 4: one replica destroyed; Kubernetes replaces it."""
    pod, node = pick_ready(pods, args.target)
    run(["kubectl", "-n", NAMESPACE, "delete", "pod", pod, "--grace-period=0", "--force"])
    return {"targets": [pod], "detail": f"pod {pod} force-deleted on node {node}"}


def fault_node_failure(args, pods):
    """Scenario 4: the machine hosting a replica is killed outright.

    This stays on `docker kill` rather than any in-cluster tool, because a kind
    node is a container on the host - outside the cluster's own reach.
    """
    pod, node = pick_ready(pods, args.target)
    run(["docker", "kill", node])
    return {"targets": [node], "detail": f"node container {node} hard-killed (hosted {pod})"}


HANDLERS = {
    "memory_leak": fault_memory_leak,
    "cpu_saturation": fault_cpu_saturation,
    "dependency_slow": fault_dependency_slow,
    "dependency_down": fault_dependency_down,
    "config_error": fault_config_error,
    "network_partition": fault_network_partition,
    "pod_failure": fault_pod_failure,
    "node_failure": fault_node_failure,
}


def main():
    parser = argparse.ArgumentParser(description="Inject one fault into the cluster.")
    parser.add_argument("--fault", choices=sorted(FAULTS), help="which fault to inject")
    parser.add_argument("--list", action="store_true", help="list the faults and exit")
    parser.add_argument("--target", default=None, help="specific pod or node to hit")
    parser.add_argument("--run-dir", default=None, help="record the fault event here")
    parser.add_argument("--dry-run", action="store_true", help="show what would happen")
    parser.add_argument("--leak-mb-per-sec", type=float, default=2.0,
                        help="memory_leak: MB retained per second of wall clock "
                             "(default 2.0, which reaches a 256Mi limit in roughly 90 s). "
                             "Deliberately time-based, not per-request: a per-request "
                             "leak throttles itself as the service slows and never "
                             "reaches the limit.")
    parser.add_argument("--cpu-ms", type=float, default=90,
                        help="cpu_saturation: ms of CPU burned per request (default 90)")
    parser.add_argument("--delay-ms", type=float, default=2000,
                        help="dependency_slow: ms added per request (default 2000)")
    parser.add_argument("--bad-port", type=int, default=9999,
                        help="config_error: the hostname to point the gateway at")
    args = parser.parse_args()

    if args.list:
        print(f"{'fault':<20} {'scenario':>8}  mechanism")
        print("-" * 78)
        for name, meta in sorted(FAULTS.items(), key=lambda kv: (kv[1]["scenario"], kv[0])):
            print(f"{name:<20} {meta['scenario']:>8}  {meta['summary']}")
        return

    if not args.fault:
        parser.error("--fault is required (or use --list)")

    pods = backend_pods()
    print("Current backend replicas:")
    for pod, node, ready in pods:
        print(f"  {pod:<34} node={node:<16} ready={ready}")

    meta = FAULTS[args.fault]
    if args.dry_run:
        print(f"\n[dry run] would inject '{args.fault}' (scenario {meta['scenario']}): "
              f"{meta['summary']}")
        return

    print(f"\n>>> INJECTING '{args.fault}' (scenario {meta['scenario']}): {meta['summary']}")

    fault_at = time.time()
    outcome = HANDLERS[args.fault](args, pods)

    print(f">>> {outcome['detail']}")
    print(f">>> fault_at = {fault_at:.3f} "
          f"({time.strftime('%H:%M:%S', time.localtime(fault_at))})")

    event = {
        "fault_type": args.fault,
        "scenario": meta["scenario"],
        "mode": args.fault,          # kept so existing tooling keeps working
        "fault_at": fault_at,
        "detail": outcome["detail"],
        "targets": outcome["targets"],
        "target_pod": outcome["targets"][0] if outcome["targets"] else None,
        "target_node": next((n for p, n, _ in pods if p == (outcome["targets"] or [None])[0]),
                            outcome["targets"][0] if args.fault == "node_failure" else None),
        "replicas_before": [{"pod": p, "node": n, "ready": r} for p, n, r in pods],
    }

    if args.run_dir:
        os.makedirs(args.run_dir, exist_ok=True)
        path = os.path.join(args.run_dir, "chaos_event.json")
        with open(path, "w") as handle:
            json.dump(event, handle, indent=2)
        print(f">>> event recorded to {path}")
    else:
        print(json.dumps(event, indent=2))

    if args.fault in ("node_failure", "dependency_down", "config_error", "network_partition"):
        print("\nThis fault does not undo itself. Run scripts/reset_cluster.sh "
              "before the next experiment.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
