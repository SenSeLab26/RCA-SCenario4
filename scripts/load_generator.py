"""Step 2 + Step 3 + Step 4: drive continuous traffic, fire the fault mid-flight,
and keep hammering while Kubernetes heals.

This is an *open-loop* generator: it sends requests at a fixed wall-clock rate
regardless of how slow the system becomes. That matters. A closed-loop
generator (N threads looping as fast as they can) automatically slows down when
the system slows down, which hides the brownout. Open-loop keeps the offered
load constant, so a capacity loss shows up as growing queue delay - the signal
this whole scenario is built around.

It also writes a host-side metrics CSV. Jaeger is the source of truth for the
dataset, but having an independent measurement recorded outside the cluster is
what lets you tell "the system got slow" apart from "the tracing pipeline got
slow".

Usage:
    python3 scripts/load_generator.py --duration 300 --rps 42 --chaos-at 60
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


class SecondBuckets:
    """Thread-safe per-second aggregation of request outcomes."""

    def __init__(self):
        self.lock = threading.Lock()
        self.buckets = {}

    def record(self, sent_at, latency_ms, ok, pod):
        second = int(sent_at)
        with self.lock:
            bucket = self.buckets.setdefault(
                second, {"latencies": [], "ok": 0, "err": 0, "pods": set(), "lag": []}
            )
            bucket["latencies"].append(latency_ms)
            bucket["pods"].add(pod) if pod else None
            if ok:
                bucket["ok"] += 1
            else:
                bucket["err"] += 1

    def record_lag(self, second, lag_s):
        with self.lock:
            bucket = self.buckets.setdefault(
                second, {"latencies": [], "ok": 0, "err": 0, "pods": set(), "lag": []}
            )
            bucket["lag"].append(lag_s)

    def snapshot(self, second):
        with self.lock:
            bucket = self.buckets.get(second)
            if not bucket:
                return None
            latencies = sorted(bucket["latencies"])
            return {
                "sent": len(latencies),
                "ok": bucket["ok"],
                "err": bucket["err"],
                "p50_ms": round(percentile(latencies, 0.50), 1),
                "p95_ms": round(percentile(latencies, 0.95), 1),
                "max_ms": round(latencies[-1], 1) if latencies else 0.0,
                "pods": sorted(bucket["pods"]),
                "max_lag_s": round(max(bucket["lag"]), 2) if bucket["lag"] else 0.0,
            }

    def all_seconds(self):
        with self.lock:
            return sorted(self.buckets.keys())


def send_one(session, url, timeout, buckets):
    sent_at = time.time()
    pod = None
    ok = False
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code == 200:
            ok = True
            body = response.json()
            pod = body.get("upstream", {}).get("pod")
    except requests.exceptions.RequestException:
        ok = False
    latency_ms = (time.time() - sent_at) * 1000
    buckets.record(sent_at, latency_ms, ok, pod)


def dispatcher(pool, session, args, buckets, stop_event, start_epoch):
    """Submit requests on a fixed schedule of start_epoch + i/rps."""
    total = int(args.duration * args.rps)
    interval = 1.0 / args.rps
    for i in range(total):
        if stop_event.is_set():
            return
        target = start_epoch + i * interval
        now = time.time()
        if target > now:
            time.sleep(target - now)
        else:
            # We are behind schedule: the client itself is the bottleneck.
            buckets.record_lag(int(target), now - target)
        pool.submit(send_one, session, args.url, args.timeout, buckets)


def reporter(buckets, stop_event, start_epoch, fault_holder):
    """Print a one-line-per-second live view of the system."""
    print()
    print("   time     rps   ok  err   p50ms   p95ms   maxms  replicas serving")
    print("   " + "-" * 66)
    second = int(start_epoch)
    while not stop_event.is_set():
        # Report a second only once it is safely in the past, so slow requests
        # that started in it have had a chance to land.
        if time.time() < second + 3:
            time.sleep(0.2)
            continue

        stats = buckets.snapshot(second)
        clock = time.strftime("%H:%M:%S", time.localtime(second))
        if stats:
            marker = ""
            fault_at = fault_holder.get("fault_at")
            if fault_at and int(fault_at) == second:
                marker = "   <== FAULT INJECTED"
            elif stats["err"] > 0:
                marker = "   <== errors"
            elif len(stats["pods"]) < 3:
                marker = "   <== degraded"
            if stats["max_lag_s"] > 0.5:
                marker += f"  [client lag {stats['max_lag_s']}s]"

            print(
                f"   {clock}  {stats['sent']:>4} {stats['ok']:>4} {stats['err']:>4}"
                f"  {stats['p50_ms']:>7} {stats['p95_ms']:>7} {stats['max_ms']:>7}"
                f"  {len(stats['pods'])}{marker}"
            )
        second += 1


def fire_chaos(args, run_dir, delay, stop_event, fault_holder):
    """Wait `delay` seconds into the run, then invoke inject_chaos.py."""
    if stop_event.wait(delay):
        return
    print(f"\n{'=' * 70}")
    print(f">>> T+{delay}s - injecting the fault")
    print(f"{'=' * 70}")
    cmd = [
        sys.executable,
        os.path.join(HERE, "inject_chaos.py"),
        "--mode", args.chaos_mode,
        "--run-dir", run_dir,
    ]
    if args.chaos_target:
        cmd += ["--target", args.chaos_target]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"!!! chaos injection FAILED:\n{result.stderr.strip()}", file=sys.stderr)
        return
    event_path = os.path.join(run_dir, "chaos_event.json")
    if os.path.exists(event_path):
        with open(event_path) as handle:
            fault_holder.update(json.load(handle))
    print(f"{'=' * 70}\n")


def write_metrics(buckets, run_dir, fault_at):
    path = os.path.join(run_dir, "loadgen_metrics.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "second_epoch", "t_rel", "clock", "sent", "ok", "err",
            "p50_ms", "p95_ms", "max_ms", "distinct_pods", "pods", "max_client_lag_s",
        ])
        for second in buckets.all_seconds():
            stats = buckets.snapshot(second)
            if not stats:
                continue
            t_rel = round(second - fault_at, 0) if fault_at else ""
            writer.writerow([
                second, t_rel, time.strftime("%H:%M:%S", time.localtime(second)),
                stats["sent"], stats["ok"], stats["err"],
                stats["p50_ms"], stats["p95_ms"], stats["max_ms"],
                len(stats["pods"]), "|".join(stats["pods"]), stats["max_lag_s"],
            ])
    return path


def main():
    parser = argparse.ArgumentParser(description="Continuous load generator with mid-run chaos injection.")
    parser.add_argument("--url", default="http://127.0.0.1:30080/order")
    parser.add_argument("--rps", type=float, default=42.0,
                        help="offered requests per second (default 42 = 70%% of the 3-replica "
                             "capacity of ~60/s, which becomes 105%% when a replica is lost)")
    parser.add_argument("--duration", type=float, default=300.0, help="run length in seconds")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="client-side timeout; must exceed the gateway BACKEND_TIMEOUT (5s)")
    parser.add_argument("--workers", type=int, default=192, help="client thread pool size")
    parser.add_argument("--chaos-at", type=float, default=None,
                        help="seconds into the run at which to inject the fault (omit to run clean)")
    parser.add_argument("--chaos-mode", choices=["node", "pod"], default="node")
    parser.add_argument("--chaos-target", default=None, help="specific pod/node to kill")
    parser.add_argument("--run-id", default=None, help="override the generated run id")
    args = parser.parse_args()

    run_id = args.run_id or (
        time.strftime("run_%Y%m%d_%H%M%S") + ("_" + args.chaos_mode if args.chaos_at else "_clean")
    )
    run_dir = os.path.join(ROOT, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Fail fast if the gateway is not reachable - better than a wall of errors.
    try:
        requests.get(args.url, timeout=10).raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"ERROR: cannot reach the API gateway at {args.url}\n  {exc}", file=sys.stderr)
        print("Is the cluster up? Try: bash scripts/cluster_up.sh", file=sys.stderr)
        sys.exit(1)

    print(f"Run id      : {run_id}")
    print(f"Target      : {args.url}")
    print(f"Offered load: {args.rps} req/s for {args.duration:.0f}s "
          f"({int(args.rps * args.duration)} requests)")
    if args.chaos_at:
        print(f"Chaos       : {args.chaos_mode} kill at T+{args.chaos_at:.0f}s")
    else:
        print("Chaos       : none (baseline run)")

    buckets = SecondBuckets()
    stop_event = threading.Event()
    fault_holder = {}
    start_epoch = time.time()

    session = requests.Session()
    session.headers["Connection"] = "close"
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=args.workers, max_retries=0)
    session.mount("http://", adapter)

    threads = [threading.Thread(target=reporter,
                                args=(buckets, stop_event, start_epoch, fault_holder),
                                daemon=True)]
    if args.chaos_at:
        threads.append(threading.Thread(target=fire_chaos,
                                        args=(args, run_dir, args.chaos_at, stop_event, fault_holder),
                                        daemon=True))
    for thread in threads:
        thread.start()

    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        dispatcher(pool, session, args, buckets, stop_event, start_epoch)
    except KeyboardInterrupt:
        print("\nInterrupted - draining in-flight requests...")
        stop_event.set()

    print("\nSchedule complete - waiting for in-flight requests to drain...")
    pool.shutdown(wait=True)
    time.sleep(4)  # let the reporter flush the final seconds
    stop_event.set()
    end_epoch = time.time()

    fault_at = fault_holder.get("fault_at")
    meta = {
        "run_id": run_id,
        "url": args.url,
        "rps": args.rps,
        "duration_s": args.duration,
        "workers": args.workers,
        "client_timeout_s": args.timeout,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "chaos_at_offset_s": args.chaos_at,
        "chaos_mode": args.chaos_mode if args.chaos_at else None,
        "fault_at": fault_at,
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)

    metrics_path = write_metrics(buckets, run_dir, fault_at)

    print(f"\nRun directory : {run_dir}")
    print(f"Host metrics  : {metrics_path}")
    print(f"Run metadata  : {os.path.join(run_dir, 'run_meta.json')}")
    if not fault_at and args.chaos_at:
        print("WARNING: no chaos_event.json was recorded - the fault may not have fired.")
    print(f"\nNext:\n  python3 scripts/extract_data.py --run-dir runs/{run_id}")


if __name__ == "__main__":
    main()
