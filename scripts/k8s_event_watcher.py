"""Record the orchestrator side of self-healing.

The request traces tell us when *users* stopped feeling the failure. They do not
tell us what Kubernetes was doing, or when. This watcher fills that gap: it polls
the cluster while a run is in progress and captures the exact moment of each
step Kubernetes takes to repair itself.

The two measurements are different, and the difference is the interesting part.
In our recorded runs the replacement replica became Ready around T+75s, but the
system was not restabilized until T+134s. Kubernetes finished healing nearly a
minute before users stopped noticing.

WHAT IT CAPTURES

Kubernetes stamps a `lastTransitionTime` on every pod and node condition, so we
do not have to catch transitions in the instant they happen - we poll, and read
the authoritative timestamps off the objects. The healing sequence is:

    fault              the node is killed              (from chaos_event.json)
    node NotReady      control plane gives up on it    (node Ready condition)
    pod deleted        eviction is requested           (deletionTimestamp)
    pod created        ReplicaSet makes a replacement  (creationTimestamp)
    pod scheduled      scheduler places it on a node   (PodScheduled condition)
    containers ready   the container is running        (ContainersReady)
    pod Ready          readiness probe passes, traffic (Ready condition)

Note the last two are separate on purpose: the gap between them is our simulated
15 second warm-up, during which the pod exists but refuses traffic.

USAGE

Run it in a second terminal, started just before the load generator:

    python3 scripts/k8s_event_watcher.py --run-dir runs/<run_id> --duration 300

It writes k8s_timeline.json and k8s_timeline.csv into the run directory, and
prints the healing breakdown when it finishes.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMESPACE = "rca4"
LABEL = "app=order-backend"

# The pod conditions we care about, in the order Kubernetes satisfies them.
POD_CONDITIONS = ["PodScheduled", "Initialized", "ContainersReady", "Ready"]


def parse_ts(value):
    """Kubernetes RFC3339 timestamp -> epoch seconds."""
    if not value:
        return None
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def kubectl_json(args):
    """Run a kubectl command and parse its JSON, returning None on failure."""
    result = subprocess.run(
        ["kubectl", *args, "-o", "json"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def snapshot_pods(observed):
    """Record the current state of every backend pod, keyed by pod name."""
    data = kubectl_json(["-n", NAMESPACE, "get", "pods", "-l", LABEL])
    if not data:
        return
    for item in data.get("items", []):
        meta = item["metadata"]
        name = meta["name"]
        record = observed.setdefault(name, {"pod": name})

        record["node"] = item.get("spec", {}).get("nodeName")
        record["created"] = parse_ts(meta.get("creationTimestamp"))
        if meta.get("deletionTimestamp"):
            record["deleted"] = parse_ts(meta["deletionTimestamp"])
        record["phase"] = item.get("status", {}).get("phase")

        for condition in item.get("status", {}).get("conditions", []):
            if condition["type"] in POD_CONDITIONS and condition["status"] == "True":
                # Keep the earliest time we saw this condition satisfied.
                key = condition["type"]
                stamp = parse_ts(condition.get("lastTransitionTime"))
                if stamp and (key not in record or record[key] is None):
                    record[key] = stamp

        # When the container process actually began running. Without this the
        # container start and the readiness warm-up are indistinguishable, since
        # ContainersReady only turns true once the readiness probe passes - which
        # our WARMUP_SECONDS gate deliberately delays.
        for status in item.get("status", {}).get("containerStatuses") or []:
            started_at = status.get("state", {}).get("running", {}).get("startedAt")
            if started_at and record.get("ContainerStarted") is None:
                record["ContainerStarted"] = parse_ts(started_at)


def snapshot_nodes(observed):
    """Record when each node's Ready condition last changed."""
    data = kubectl_json(["get", "nodes"])
    if not data:
        return
    for item in data.get("items", []):
        name = item["metadata"]["name"]
        for condition in item.get("status", {}).get("conditions", []):
            if condition["type"] != "Ready":
                continue
            record = observed.setdefault(name, {"node": name})
            status = condition["status"]      # "True", "False" or "Unknown"
            stamp = parse_ts(condition.get("lastTransitionTime"))
            record["ready_status"] = status
            if status != "True" and "not_ready_at" not in record:
                record["not_ready_at"] = stamp
            record["last_transition"] = stamp


def build_timeline(pods, nodes, chaos):
    """Turn the raw observations into the healing sequence, relative to the fault."""
    fault_at = chaos.get("fault_at")
    killed_pod = chaos.get("target_pod")
    killed_node = chaos.get("target_node")

    def rel(stamp):
        if stamp is None or fault_at is None:
            return None
        return round(stamp - fault_at, 1)

    events = []

    if fault_at:
        events.append({"event": "fault injected",
                       "detail": f"{chaos.get('mode', 'node')} {killed_node or killed_pod}",
                       "epoch": fault_at, "t_rel": 0.0})

    # The control plane noticing the node is gone.
    node = nodes.get(killed_node, {})
    if node.get("not_ready_at"):
        events.append({"event": "node marked NotReady",
                       "detail": killed_node,
                       "epoch": node["not_ready_at"],
                       "t_rel": rel(node["not_ready_at"])})

    # The old replica being evicted.
    old = pods.get(killed_pod, {})
    if old.get("deleted"):
        events.append({"event": "old replica deletion requested",
                       "detail": killed_pod,
                       "epoch": old["deleted"], "t_rel": rel(old["deleted"])})

    # The replacement is any pod created after the fault.
    replacements = [
        record for name, record in pods.items()
        if name != killed_pod and record.get("created") and fault_at
        and record["created"] >= fault_at - 2
    ]
    replacements.sort(key=lambda r: r["created"])

    for record in replacements:
        name = record["pod"]
        for label, key in [("replacement created", "created"),
                           ("replacement scheduled", "PodScheduled"),
                           ("replacement container started", "ContainerStarted"),
                           ("replacement Ready (accepting traffic)", "Ready")]:
            if record.get(key):
                events.append({"event": label,
                               "detail": f"{name} on {record.get('node')}",
                               "epoch": record[key], "t_rel": rel(record[key])})

    events.sort(key=lambda e: e["epoch"])
    return events, replacements


def print_timeline(events, replacements, fault_at, mode="node"):
    print()
    print("KUBERNETES SELF-HEALING TIMELINE")
    print("=" * 72)
    if not events:
        print("No healing events captured. Was the watcher running during the fault?")
        return

    print(f"{'t_rel':>8}  {'event':<38} detail")
    print("-" * 72)
    for event in events:
        value = event["t_rel"]
        if value is not None and -1.0 < value < 0:
            value = 0.0   # same-second as the fault; K8s stamps to the second
        t_rel = f"{value:+.0f}s" if value is not None else "?"
        print(f"{t_rel:>8}  {event['event']:<38} {event['detail']}")
    print("-" * 72)

    # The breakdown is the point: which stage actually cost the time.
    by_event = {e["event"]: e["t_rel"] for e in events if e["t_rel"] is not None}
    node_mode = mode == "node"
    stages = [
        ("Detection (fault -> node NotReady)", 0.0,
         by_event.get("node marked NotReady"), node_mode),
        ("Eviction (NotReady -> replica deleted)",
         by_event.get("node marked NotReady"),
         by_event.get("old replica deletion requested"), node_mode),
        ("Scheduling (created -> scheduled)",
         by_event.get("replacement created"), by_event.get("replacement scheduled"), True),
        ("Container start (scheduled -> running)",
         by_event.get("replacement scheduled"),
         by_event.get("replacement container started"), True),
        ("Warm-up (running -> Ready, the readiness gate)",
         by_event.get("replacement container started"),
         by_event.get("replacement Ready (accepting traffic)"), True),
    ]
    print("\nWHERE THE TIME WENT")
    for label, start, end, applies in stages:
        if not applies:
            print(f"  {label:<48} n/a for a pod kill")
        elif start is None or end is None:
            print(f"  {label:<48} not captured")
        else:
            print(f"  {label:<48} {end - start:5.0f}s")

    ready = by_event.get("replacement Ready (accepting traffic)")
    if ready is not None:
        print(f"\n  Kubernetes finished healing at T+{ready:.0f}s.")
        print("  Compare this against the restabilization time measured from the")
        print("  request traces: the gap between them is the backlog still draining")
        print("  after the orchestrator considered the repair complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Record Kubernetes self-healing events during a chaos run.")
    parser.add_argument("--run-dir", required=True,
                        help="run directory to read chaos_event.json from and write results into")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="how long to watch, in seconds")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="polling interval in seconds")
    args = parser.parse_args()

    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(ROOT, args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    if kubectl_json(["get", "nodes"]) is None:
        print("ERROR: cannot reach the cluster. Is it running? "
              "Try: bash scripts/cluster_up.sh", file=sys.stderr)
        sys.exit(1)

    print(f"Watching Kubernetes for {args.duration:.0f}s "
          f"(polling every {args.interval:.0f}s). Run the load generator now.")

    pods, nodes = {}, {}
    deadline = time.time() + args.duration
    try:
        while time.time() < deadline:
            snapshot_pods(pods)
            snapshot_nodes(nodes)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nInterrupted - writing what was captured so far.")

    # The fault time is written by inject_chaos.py, so read it at the end.
    chaos_path = os.path.join(run_dir, "chaos_event.json")
    chaos = {}
    if os.path.exists(chaos_path):
        with open(chaos_path) as handle:
            chaos = json.load(handle)
    else:
        print(f"WARNING: {chaos_path} not found, so events cannot be placed "
              "relative to the fault.")

    events, replacements = build_timeline(pods, nodes, chaos)
    print_timeline(events, replacements, chaos.get("fault_at"),
                   mode=chaos.get("mode", "node"))

    json_path = os.path.join(run_dir, "k8s_timeline.json")
    with open(json_path, "w") as handle:
        json.dump({"fault": chaos, "events": events,
                   "pods_observed": pods, "nodes_observed": nodes}, handle, indent=2)

    csv_path = os.path.join(run_dir, "k8s_timeline.csv")
    with open(csv_path, "w") as handle:
        handle.write("t_rel,epoch,event,detail\n")
        for event in events:
            t_rel = "" if event["t_rel"] is None else event["t_rel"]
            handle.write(f"{t_rel},{event['epoch']},{event['event']},{event['detail']}\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
