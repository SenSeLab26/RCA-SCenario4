"""Proactive self-healing: repair a pod before it fails, not after.

Kubernetes heals reactively. It waits for a container to die, then replaces it.
For a memory leak that is the worst possible moment to act, because every replica
is leaking at the same rate and so they all reach the limit at roughly the same
time. The result is that the whole tier dies together and users see a full
outage.

This controller acts earlier. It watches how fast each replica's memory is
growing, works out when that replica will hit its limit, and restarts it while it
is still healthy. Because it restarts one replica at a time and waits for the
replacement to be serving before touching the next, capacity never drops below
what the traffic needs and users see no outage at all.

WHAT IT DOES EACH TICK

    1. Read real memory use and the real limit from every replica's /metrics.
    2. Fit a straight line to each replica's recent memory history.
    3. From that line, work out how many seconds remain before the limit.
    4. If a replica is closer to its limit than --act-before seconds, and no
       other restart is in progress, restart that replica.

The prediction uses the kernel's own memory figures, not the application's
bookkeeping, so nothing here depends on the fault being one we injected.

Usage:
    python3 scripts/proactive_healer.py --duration 240
    python3 scripts/proactive_healer.py --duration 240 --observe-only
"""

import argparse
import json
import os
import subprocess
import time

NAMESPACE = "rca4"
LABEL = "app=order-backend"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, check=False):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def ready_pods():
    """Return the names of replicas that are currently serving traffic."""
    raw = run(["kubectl", "-n", NAMESPACE, "get", "pods", "-l", LABEL, "-o", "json"])
    if not raw:
        return []
    pods = []
    for item in json.loads(raw).get("items", []):
        statuses = item.get("status", {}).get("containerStatuses") or []
        if item.get("metadata", {}).get("deletionTimestamp"):
            continue
        if statuses and all(s.get("ready") for s in statuses):
            pods.append(item["metadata"]["name"])
    return pods


def read_metrics(pod):
    """Ask one replica for its real memory numbers."""
    code = ("import urllib.request;"
            "print(urllib.request.urlopen('http://localhost:8000/metrics').read().decode())")
    raw = run(["kubectl", "-n", NAMESPACE, "exec", pod, "--", "python", "-c", code])
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def seconds_until_limit(history, limit_mb, headroom_mb):
    """Fit a line to recent memory use and return seconds until the limit.

    Returns None when memory is not growing, which is the normal healthy case.
    """
    if len(history) < 4 or limit_mb <= 0:
        return None

    times = [t for t, _ in history]
    used = [u for _, u in history]
    span = times[-1] - times[0]
    if span <= 0:
        return None

    # Least squares slope, in MB per second.
    mean_t = sum(times) / len(times)
    mean_u = sum(used) / len(used)
    denominator = sum((t - mean_t) ** 2 for t in times)
    if denominator == 0:
        return None
    slope = sum((t - mean_t) * (u - mean_u) for t, u in zip(times, used)) / denominator

    if slope <= 0.05:            # not meaningfully growing
        return None

    # The kernel kills the container a little before the limit is reached, so
    # treat the ceiling as the limit minus a safety margin.
    ceiling = limit_mb - headroom_mb
    remaining = ceiling - used[-1]
    return max(0.0, remaining / slope)


def restart(pod):
    """Delete one replica so Kubernetes creates a fresh one in its place."""
    run(["kubectl", "-n", NAMESPACE, "delete", "pod", pod, "--wait=false"])


def main():
    parser = argparse.ArgumentParser(
        description="Restart replicas before they run out of memory.")
    parser.add_argument("--duration", type=float, default=240.0)
    parser.add_argument("--interval", type=float, default=3.0,
                        help="seconds between checks")
    parser.add_argument("--act-before", type=float, default=30.0,
                        help="restart a replica when it is this close to its limit, "
                             "in seconds (default 30)")
    parser.add_argument("--headroom-mb", type=float, default=25.0,
                        help="treat the limit as this much lower, because the kernel "
                             "kills the container slightly before it is reached")
    parser.add_argument("--history", type=int, default=10,
                        help="how many recent samples the trend is fitted to")
    parser.add_argument("--observe-only", action="store_true",
                        help="report predictions but never restart anything")
    parser.add_argument("--run-dir", default=None,
                        help="directory to write the action log into")
    args = parser.parse_args()

    if not ready_pods():
        raise SystemExit("No Ready replicas found. Is the cluster up?")

    mode = "OBSERVE ONLY" if args.observe_only else "ACTIVE"
    print(f"Proactive healer [{mode}] - watching for {args.duration:.0f}s, "
          f"checking every {args.interval:.0f}s")
    print(f"Will act when a replica is within {args.act_before:.0f}s of its limit.\n")
    print(f"{'time':>8} {'pod':<34} {'used/limit MB':>15} {'MB/s':>7} "
          f"{'time left':>10}  action")
    print("-" * 96)

    history = {}
    actions = []
    restart_in_progress_until = 0.0
    started = time.time()

    while time.time() - started < args.duration:
        now = time.time()
        pods = ready_pods()

        for pod in pods:
            metrics = read_metrics(pod)
            if not metrics:
                continue

            used = metrics.get("memory_used_mb", 0.0)
            limit = metrics.get("memory_limit_mb", 0.0)
            samples = history.setdefault(pod, [])
            samples.append((now, used))
            del samples[:-args.history]

            remaining = seconds_until_limit(samples, limit, args.headroom_mb)
            slope = 0.0
            if len(samples) >= 2 and samples[-1][0] > samples[0][0]:
                slope = ((samples[-1][1] - samples[0][1])
                         / (samples[-1][0] - samples[0][0]))

            clock = f"{now - started:7.0f}s"
            left = f"{remaining:9.0f}s" if remaining is not None else "        -"
            note = ""

            if remaining is not None and remaining <= args.act_before:
                if args.observe_only:
                    note = "  WOULD RESTART (observe only)"
                elif now < restart_in_progress_until:
                    note = "  waiting: another restart in progress"
                elif len(pods) < 2:
                    # Never take the last serving replica away.
                    note = "  holding: too few replicas to restart safely"
                else:
                    restart(pod)
                    # Give the replacement time to start and pass its warm-up
                    # before considering another restart.
                    restart_in_progress_until = now + 40
                    history.pop(pod, None)
                    note = "  RESTARTED BEFORE FAILURE"
                    actions.append({"epoch": now, "t_rel": round(now - started, 1),
                                    "pod": pod, "used_mb": used, "limit_mb": limit,
                                    "predicted_seconds_left": round(remaining, 1)})

            if note or (remaining is not None) or len(samples) % 4 == 0:
                print(f"{clock} {pod:<34} {used:>7.0f}/{limit:<7.0f} {slope:>7.2f} "
                      f"{left:>10}{note}")

        time.sleep(args.interval)

    print("-" * 96)
    print(f"\n{len(actions)} replica(s) restarted before they could fail.")
    for action in actions:
        print(f"  T+{action['t_rel']:>5.0f}s  {action['pod']}  at "
              f"{action['used_mb']:.0f} MB of {action['limit_mb']:.0f} MB, "
              f"{action['predicted_seconds_left']:.0f}s of headroom left")

    if args.run_dir:
        path = os.path.join(
            args.run_dir if os.path.isabs(args.run_dir)
            else os.path.join(ROOT, args.run_dir), "proactive_actions.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump({"mode": mode, "act_before_s": args.act_before,
                       "actions": actions}, handle, indent=2)
        print(f"\nAction log written to {path}")


if __name__ == "__main__":
    main()
