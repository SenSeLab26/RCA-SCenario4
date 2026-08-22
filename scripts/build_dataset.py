"""Step 5b: flatten the raw traces into the recovery time series.

Scenarios 1-3 produced one row per trace, labelled healthy/crash. This scenario
needs something different: one row per *second* of the incident, labelled with
how much longer the healing has left to run.

For every second we compute what an SRE would look at on a dashboard - request
rate, error rate, p50/p95/max latency, mean queue wait, and how many distinct
backend replicas actually served traffic - and then we work out, after the fact,
the exact second at which the system became stable again. The gap between the
two is the training label:

    seconds_to_restabilize = restabilized_at - now

Restabilization is defined concretely, not by eye: p95 latency back within
`--tolerance` of the pre-fault baseline, error rate back down to the baseline
floor, and all expected replicas serving traffic again - holding for most of a
`--hold` second window. Note "the baseline floor" rather than "zero": a healthy
kube-proxy still drops the occasional SYN, and the finish line has to be "as
good as before the fault", not "perfect".

Usage:
    python3 scripts/build_dataset.py --run-dir runs/run_20260725_141530_node
"""

import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GATEWAY_OP = "api-gateway.order"

FIELDNAMES = [
    "run_id", "chaos_mode", "second_epoch", "t_rel", "clock", "phase",
    "req_count", "ok_count", "err_count", "err_rate",
    "p50_ms", "p95_ms", "max_ms", "mean_ms", "mean_queue_wait_ms",
    "active_pods", "pods_missing", "pods",
    "baseline_p95_ms", "p95_ratio", "p95_slope_5s", "err_rate_roll5",
    "time_since_fault", "seconds_to_restabilize",
]


def percentile(sorted_values, fraction):
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def tag_map(span):
    return {tag.get("key"): tag.get("value") for tag in span.get("tags", [])}


def load_json(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: {path} not found ({what}).")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def collect_gateway_spans(traces):
    """Pull the one gateway span out of each trace, as a flat list of records."""
    records = []
    for trace in traces:
        for span in trace.get("spans", []):
            if span.get("operationName") != GATEWAY_OP:
                continue
            tags = tag_map(span)
            is_error = bool(tags.get("error")) or tags.get("otel.status_code") == "ERROR"
            records.append({
                "start_epoch": span["startTime"] / 1_000_000.0,
                "latency_ms": span["duration"] / 1000.0,
                "is_error": is_error,
                "backend_pod": tags.get("backend.pod"),
                "queue_wait_ms": tags.get("backend.queue_wait_ms"),
            })
    return records


def aggregate_by_second(records):
    buckets = {}
    for record in records:
        second = int(record["start_epoch"])
        bucket = buckets.setdefault(
            second, {"latencies": [], "err": 0, "ok": 0, "pods": set(), "queue": []}
        )
        bucket["latencies"].append(record["latency_ms"])
        if record["is_error"]:
            bucket["err"] += 1
        else:
            bucket["ok"] += 1
            if record["backend_pod"]:
                bucket["pods"].add(record["backend_pod"])
            if isinstance(record["queue_wait_ms"], (int, float)):
                bucket["queue"].append(float(record["queue_wait_ms"]))
    return buckets


def build_rows(buckets, fault_at, expected_replicas):
    """Turn the per-second buckets into a dense, gap-free list of rows."""
    seconds = sorted(buckets.keys())
    rows = []
    for second in range(seconds[0], seconds[-1] + 1):
        bucket = buckets.get(second)
        if bucket:
            latencies = sorted(bucket["latencies"])
            req_count = len(latencies)
            rows.append({
                "second_epoch": second,
                "t_rel": second - int(fault_at) if fault_at else "",
                "clock": time.strftime("%H:%M:%S", time.localtime(second)),
                "req_count": req_count,
                "ok_count": bucket["ok"],
                "err_count": bucket["err"],
                "err_rate": round(bucket["err"] / req_count, 4) if req_count else 0.0,
                "p50_ms": round(percentile(latencies, 0.50), 1),
                "p95_ms": round(percentile(latencies, 0.95), 1),
                "max_ms": round(latencies[-1], 1),
                "mean_ms": round(sum(latencies) / req_count, 1),
                "mean_queue_wait_ms": round(sum(bucket["queue"]) / len(bucket["queue"]), 1)
                                      if bucket["queue"] else 0.0,
                "active_pods": len(bucket["pods"]),
                "pods": "|".join(sorted(bucket["pods"])),
            })
        else:
            # A second with no traces at all. Rare, but it must not create a
            # hole in the time series - the slope features assume dense seconds.
            rows.append({
                "second_epoch": second,
                "t_rel": second - int(fault_at) if fault_at else "",
                "clock": time.strftime("%H:%M:%S", time.localtime(second)),
                "req_count": 0, "ok_count": 0, "err_count": 0, "err_rate": 0.0,
                "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0,
                "mean_queue_wait_ms": 0.0, "active_pods": 0, "pods": "",
            })

        rows[-1]["pods_missing"] = max(0, expected_replicas - rows[-1]["active_pods"])
    return rows


def compute_baseline(rows, fault_at, window_start=-60, window_end=-5):
    """Median pre-fault p95. This is the 'normal' the system has to return to."""
    if not fault_at:
        return None
    pre = [r["p95_ms"] for r in rows
           if r["t_rel"] != "" and window_start <= r["t_rel"] <= window_end and r["req_count"] > 0]
    if len(pre) < 5:
        pre = [r["p95_ms"] for r in rows
               if r["t_rel"] != "" and r["t_rel"] < 0 and r["req_count"] > 0]
    if not pre:
        return None
    return round(median(pre), 1)


def baseline_error_rate(rows):
    """The error rate the cluster shows when nothing is wrong.

    It is not zero. kube-proxy's iptables DNAT occasionally drops a SYN under a
    high rate of short-lived connections, and the TCP retransmit makes that one
    request slow. "Restabilized" therefore has to mean "back to how it behaved
    before the fault", not "flawless" - otherwise a single unlucky packet
    anywhere in the recovery window moves the finish line.
    """
    pre = [r["err_rate"] for r in rows if r["t_rel"] != "" and r["t_rel"] < 0 and r["req_count"] > 0]
    if not pre:
        return 0.0
    return sum(pre) / len(pre)


def find_restabilization(rows, baseline_p95, base_err_rate, expected_replicas,
                         tolerance, hold, hold_fraction):
    """First post-fault second from which the system stays healthy for `hold` seconds.

    Returns (t_rel_of_restabilization, latency_threshold, error_threshold).
    """
    # A relative tolerance alone is brittle when the baseline is tiny, so also
    # allow a small absolute slack.
    threshold = max(baseline_p95 * tolerance, baseline_p95 + 20)
    err_threshold = max(base_err_rate * 2.0, 0.02)

    incident = [r for r in rows if r["t_rel"] != "" and r["t_rel"] >= 0]
    by_t = {r["t_rel"]: r for r in incident}
    if not incident:
        return None, threshold, err_threshold

    def is_healthy(row):
        return (row["err_rate"] <= err_threshold
                and row["active_pods"] >= expected_replicas
                and row["p95_ms"] <= threshold)

    # Requiring every second in the window to be healthy makes the label hostage
    # to one noisy second, so require a strong majority instead - but insist the
    # window *starts* healthy, so the timestamp still marks a real transition.
    needed = int(round(hold * hold_fraction))
    last_t = max(by_t)
    for row in incident:
        start = row["t_rel"]
        if start + hold - 1 > last_t:
            break  # not enough runway left to confirm a hold
        window = [by_t[t] for t in range(start, start + hold) if t in by_t]
        if len(window) < hold or not is_healthy(window[0]):
            continue
        if sum(1 for w in window if is_healthy(w)) >= needed:
            return start, threshold, err_threshold
    return None, threshold, err_threshold


def classify_phase(row, restabilized_at, threshold, err_threshold, expected_replicas):
    if row["t_rel"] == "" or row["t_rel"] < 0:
        return "baseline"
    if restabilized_at is not None and row["t_rel"] >= restabilized_at:
        return "stable"
    if row["active_pods"] < expected_replicas:
        if row["err_rate"] > err_threshold:
            # A replica is missing *and* requests are failing: the load balancer
            # is still routing traffic to the dead replica, and those requests
            # hang until the gateway gives up on them.
            return "blackhole"
        # Endpoints have dropped the dead replica; the survivors now carry the
        # whole offered load with two thirds of the capacity.
        return "brownout"
    # All replicas are serving again. Errors here are the brownout's backlog
    # draining past the gateway timeout, not a routing black hole.
    return "recovering"


def add_derived(rows, baseline_p95):
    for index, row in enumerate(rows):
        row["baseline_p95_ms"] = baseline_p95 if baseline_p95 else ""
        row["p95_ratio"] = round(row["p95_ms"] / baseline_p95, 3) if baseline_p95 else ""
        prior = rows[index - 5] if index >= 5 else rows[0]
        row["p95_slope_5s"] = round((row["p95_ms"] - prior["p95_ms"]) / 5.0, 2)
        window = rows[max(0, index - 4):index + 1]
        row["err_rate_roll5"] = round(sum(w["err_rate"] for w in window) / len(window), 4)
        row["time_since_fault"] = row["t_rel"] if row["t_rel"] != "" else ""


def print_curve(rows, baseline_p95, restabilized_at, every):
    """A terminal view of the bell curve, so you can sanity-check the run."""
    incident_rows = [r for r in rows if r["t_rel"] != ""]
    if not incident_rows:
        return
    peak = max(r["p95_ms"] for r in incident_rows) or 1.0
    width = 44

    print(f"\n{'t_rel':>6}  {'phase':<10} {'p95ms':>7} {'err':>4} {'pods':>4}  latency")
    print("-" * 78)
    for row in incident_rows:
        if row["t_rel"] % every != 0:
            continue
        bar = "#" * max(1, int(row["p95_ms"] / peak * width)) if row["p95_ms"] else ""
        mark = ""
        if row["t_rel"] == 0:
            mark = "  <-- FAULT"
        elif restabilized_at is not None and row["t_rel"] == restabilized_at:
            mark = "  <-- RESTABILIZED"
        print(f"{row['t_rel']:>6}  {row['phase']:<10} {row['p95_ms']:>7.0f} "
              f"{row['err_count']:>4} {row['active_pods']:>4}  {bar}{mark}")
    if baseline_p95:
        print("-" * 78)
        print(f"baseline p95 = {baseline_p95} ms")


def append_to_pool(rows, run_id, pool_path):
    """Maintain data/recovery_dataset.csv as the union of all runs."""
    os.makedirs(os.path.dirname(pool_path), exist_ok=True)
    existing = []
    if os.path.exists(pool_path):
        with open(pool_path, newline="", encoding="utf-8") as handle:
            existing = [r for r in csv.DictReader(handle) if r.get("run_id") != run_id]
    with open(pool_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(rows)
    return len(existing) + len(rows)


def main():
    parser = argparse.ArgumentParser(description="Flatten one run's traces into a recovery time series.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-replicas", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.25,
                        help="p95 may sit this many times above baseline and still count as stable")
    parser.add_argument("--hold", type=int, default=10,
                        help="length of the window that must look healthy to declare restabilization")
    parser.add_argument("--hold-fraction", type=float, default=0.8,
                        help="fraction of that window that must be healthy (default 0.8)")
    parser.add_argument("--print-every", type=int, default=5,
                        help="print one curve line every N seconds")
    args = parser.parse_args()

    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(ROOT, args.run_dir)
    meta = load_json(os.path.join(run_dir, "run_meta.json"), "run metadata")
    raw = load_json(os.path.join(run_dir, "raw_trace_data.json"),
                    "raw traces - run extract_data.py first")

    chaos_path = os.path.join(run_dir, "chaos_event.json")
    chaos = load_json(chaos_path, "chaos event") if os.path.exists(chaos_path) else {}
    fault_at = chaos.get("fault_at") or meta.get("fault_at")

    traces = raw.get("data", [])
    print(f"Run     : {meta['run_id']}")
    print(f"Traces  : {len(traces)}")

    records = collect_gateway_spans(traces)
    if not records:
        print(f"ERROR: no '{GATEWAY_OP}' spans found in the export.", file=sys.stderr)
        sys.exit(1)
    print(f"Gateway spans: {len(records)}")

    if not fault_at:
        print("WARNING: no fault time recorded - this looks like a baseline run. "
              "The time series will be written without labels.")

    buckets = aggregate_by_second(records)
    rows = build_rows(buckets, fault_at, args.expected_replicas)

    baseline_p95 = compute_baseline(rows, fault_at)
    base_err_rate = baseline_error_rate(rows)
    restabilized_at, threshold, err_threshold = (None, 0.0, 0.02)
    if baseline_p95:
        restabilized_at, threshold, err_threshold = find_restabilization(
            rows, baseline_p95, base_err_rate, args.expected_replicas,
            args.tolerance, args.hold, args.hold_fraction
        )

    add_derived(rows, baseline_p95)

    for row in rows:
        row["run_id"] = meta["run_id"]
        row["chaos_mode"] = chaos.get("mode", "")
        row["phase"] = classify_phase(row, restabilized_at, threshold, err_threshold,
                                      args.expected_replicas)
        if restabilized_at is not None and row["t_rel"] != "" and row["t_rel"] >= 0:
            row["seconds_to_restabilize"] = max(0, restabilized_at - row["t_rel"])
        else:
            row["seconds_to_restabilize"] = ""

    print_curve(rows, baseline_p95, restabilized_at, args.print_every)

    print()
    if baseline_p95:
        print(f"Baseline p95           : {baseline_p95} ms")
        print(f"Baseline error rate    : {base_err_rate * 100:.2f}%  (the healthy-cluster floor)")
        print(f"Stability criteria     : p95 <= {threshold:.0f} ms, err_rate <= "
              f"{err_threshold * 100:.1f}%, {args.expected_replicas} replicas serving,")
        print(f"                         for {int(round(args.hold * args.hold_fraction))} "
              f"of {args.hold} consecutive seconds")
    if restabilized_at is not None:
        incident = [r for r in rows if r["t_rel"] != "" and 0 <= r["t_rel"] <= restabilized_at]
        peak = max((r["p95_ms"] for r in incident), default=0)
        print(f"Peak p95 during incident: {peak:.0f} ms "
              f"({peak / baseline_p95:.1f}x baseline)" if baseline_p95 else "")
        print(f"RESTABILIZED AFTER     : {restabilized_at} seconds")
        print(f"Labelled rows          : {len(incident)}")
    elif fault_at:
        print("WARNING: the system never met the stability criteria inside this run window.")
        print("         Either extend --duration, or relax --tolerance / --hold.")
        print("         Rows are written unlabelled, so this run cannot be used for training.")

    out_path = os.path.join(run_dir, "recovery_timeseries.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    pool_path = os.path.join(ROOT, "data", "recovery_dataset.csv")
    pool_size = append_to_pool(rows, meta["run_id"], pool_path)

    print(f"\nPer-run time series : {out_path}  ({len(rows)} rows)")
    print(f"Pooled dataset      : {pool_path}  ({pool_size} rows across all runs)")
    print("\nNext:\n  python3 scripts/forecast_recovery.py")


if __name__ == "__main__":
    # Every run is also saved to results/ as a numbered, timestamped text, CSV
    # and PDF report, so the terminal output is never the only copy.
    from run_report import RunReport

    with RunReport("scenario-4", "build_dataset"):
        main()
