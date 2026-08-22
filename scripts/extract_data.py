"""Step 5a: pull the raw traces for one run out of Jaeger.

Same idea as `extract_data.py` in the earlier scenarios, with one important
difference: a five-minute run at 32 req/s produces roughly 10,000 traces, and
Jaeger's query API caps how many it will return per call. So instead of one big
request we walk the run window in short time slices and stitch the results
together, de-duplicating by trace ID.

Usage:
    python3 scripts/extract_data.py --run-dir runs/run_20260725_141530_node
"""

import argparse
import json
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_meta(run_dir):
    path = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: {path} not found. Run the load generator first.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fetch_window(base_url, service, start_us, end_us, limit):
    response = requests.get(
        f"{base_url}/api/traces",
        params={"service": service, "start": start_us, "end": end_us, "limit": limit},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("data", []) or []


def main():
    parser = argparse.ArgumentParser(description="Export one run's traces from Jaeger.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--jaeger", default="http://127.0.0.1:30686")
    parser.add_argument("--service", default="api-gateway")
    parser.add_argument("--slice", type=float, default=15.0,
                        help="seconds per query window (default 15)")
    parser.add_argument("--limit", type=int, default=4000,
                        help="max traces per query window")
    parser.add_argument("--pad", type=float, default=15.0,
                        help="seconds of padding around the run window")
    args = parser.parse_args()

    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(ROOT, args.run_dir)
    meta = load_meta(run_dir)

    start_epoch = meta["start_epoch"] - args.pad
    end_epoch = meta["end_epoch"] + args.pad
    total_span = end_epoch - start_epoch

    print(f"Run          : {meta['run_id']}")
    print(f"Window       : {time.strftime('%H:%M:%S', time.localtime(start_epoch))} "
          f"-> {time.strftime('%H:%M:%S', time.localtime(end_epoch))} "
          f"({total_span:.0f}s)")
    print(f"Jaeger       : {args.jaeger}  (service '{args.service}')")
    print(f"Fetching in {args.slice:.0f}s slices...\n")

    traces = {}
    cursor = start_epoch
    slice_index = 0
    while cursor < end_epoch:
        window_end = min(cursor + args.slice, end_epoch)
        start_us = int(cursor * 1_000_000)
        end_us = int(window_end * 1_000_000)

        try:
            batch = fetch_window(args.jaeger, args.service, start_us, end_us, args.limit)
        except requests.exceptions.RequestException as exc:
            print(f"ERROR querying Jaeger: {exc}", file=sys.stderr)
            print("Is the cluster up, and is port 30686 mapped? Try: kubectl -n rca4 get pods",
                  file=sys.stderr)
            sys.exit(1)

        new = 0
        for trace in batch:
            trace_id = trace.get("traceID")
            if trace_id and trace_id not in traces:
                traces[trace_id] = trace
                new += 1

        slice_index += 1
        print(f"  slice {slice_index:>3} "
              f"[{time.strftime('%H:%M:%S', time.localtime(cursor))}] "
              f"returned {len(batch):>5}, new {new:>5}, total {len(traces):>6}"
              + ("   <-- hit the limit, consider a smaller --slice"
                 if len(batch) >= args.limit else ""))

        cursor = window_end

    if not traces:
        print("\nNo traces found. Check that the service name matches what the gateway "
              "reports in Jaeger, and that the run actually generated traffic.",
              file=sys.stderr)
        sys.exit(1)

    payload = {"data": list(traces.values())}
    out_path = os.path.join(run_dir, "raw_trace_data.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nSUCCESS: {len(traces)} unique traces -> {out_path} ({size_mb:.1f} MB)")
    print(f"\nNext:\n  python3 scripts/build_dataset.py --run-dir {args.run_dir}")


if __name__ == "__main__":
    # Every run is also saved to results/ as a numbered, timestamped text, CSV
    # and PDF report, so the terminal output is never the only copy.
    from run_report import RunReport

    with RunReport("scenario-4", "extract_data"):
        main()
