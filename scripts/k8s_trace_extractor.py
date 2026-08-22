"""Extract and clean the Kubernetes traces for Scenario 4.

Pulls traces from Jaeger, flattens the span graph into one row per request, and
aggregates into 1-second buckets to produce the recovery curve.

Usage:
    # From a live cluster (requires scripts/cluster_up.sh to have been run)
    python3 scripts/k8s_trace_extractor.py

    # From a raw export already saved by extract_data.py (no cluster needed)
    python3 scripts/k8s_trace_extractor.py --from-file runs/<run_id>/raw_trace_data.json
"""

import argparse
import datetime
import json

import pandas as pd
import requests

# --- Configuration ---
# Scenario 4 runs Jaeger *inside* the cluster and publishes it on NodePort
# 30686. Port 16686 is the standalone Jaeger used by Scenarios 1-3.
JAEGER_API_URL = "http://localhost:30686/api/traces"

# The instrumented entry point. Every request produces exactly one span with
# this service name, so this is the service to query.
SERVICE_NAME = "api-gateway"

# The two operations that make up each trace.
GATEWAY_OP = "api-gateway.order"      # Level 1: the entry point (end-to-end latency)
BACKEND_OP = "order-backend.process"  # Level 2: the replica that served it

LIMIT = 20000  # A 5-minute run at 42 req/s produces roughly 12,500 traces


def fetch_traces(from_file=None):
    """Return the list of traces, either from Jaeger or from a saved export."""
    if from_file:
        print(f"Loading traces from file: {from_file}")
        with open(from_file, encoding="utf-8") as handle:
            return json.load(handle).get("data", [])

    print(f"Querying Jaeger API for service: {SERVICE_NAME}...")
    params = {"service": SERVICE_NAME, "limit": LIMIT}
    try:
        response = requests.get(JAEGER_API_URL, params=params, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to Jaeger. Is the cluster running, and is "
              "port 30686 published? Try: bash scripts/cluster_up.sh")
        return None
    except requests.exceptions.RequestException as exc:
        print(f"ERROR: Jaeger returned an error: {exc}")
        return None

    return response.json().get("data", [])


def extract_k8s_traces(from_file=None, output="k8s_recovery_data.csv"):
    traces = fetch_traces(from_file)
    if traces is None:
        return
    if not traces:
        print("No traces returned. Check the service name and that a run has happened.")
        return

    print(f"Successfully retrieved {len(traces)} traces. Flattening data...")
    if not from_file and len(traces) >= LIMIT:
        print(f"WARNING: hit the query limit of {LIMIT}, so the export is truncated. "
              "Use scripts/extract_data.py, which walks the run in time slices.")

    flattened_data = []

    # 2. Parse the complex JSON Graph
    for trace in traces:
        # Jaeger groups metadata into 'processes'. We need to map processID to a Pod ID.
        processes = trace.get("processes", {})

        for span in trace.get("spans", []):
            operation = span.get("operationName")
            if operation not in (GATEWAY_OP, BACKEND_OP):
                continue

            # Convert microseconds to standard seconds and milliseconds
            start_time_unix = span["startTime"] / 1000000.0
            timestamp = datetime.datetime.fromtimestamp(start_time_unix)
            duration_ms = span["duration"] / 1000.0

            span_tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}

            # Extract the Pod ID from the process tags. Note this is the pod that
            # *emitted* the span, so for a gateway span it is the gateway pod.
            process_tags = processes.get(span["processID"], {}).get("tags", [])
            pod_id = "unknown-pod"
            for tag in process_tags:
                if tag["key"] in ("k8s.pod.name", "hostname", "container_id"):
                    pod_id = tag["value"]
                    break

            # The backend replica that actually served the request. This is the
            # column that shows the load balancer going 3 replicas -> 2 -> 3.
            # A gateway span records it as a tag; a backend span *is* that pod.
            if operation == BACKEND_OP:
                backend_pod = pod_id
            else:
                backend_pod = span_tags.get("backend.pod")  # absent when the request failed

            # Check for errors (requests routed to the dead node time out)
            is_error = int(bool(span_tags.get("error"))
                           or span_tags.get("otel.status_code") == "ERROR")

            flattened_data.append({
                "timestamp": timestamp,
                "pod_id": pod_id,
                "backend_pod": backend_pod,
                "operation": operation,
                "duration_ms": duration_ms,
                "is_error": is_error,
            })

    # 3. Convert to Pandas DataFrame for aggregation
    df = pd.DataFrame(flattened_data)

    if df.empty:
        print("No relevant spans found. Check your service name or Jaeger UI.")
        return

    # Sort chronologically
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)

    # 4. AGGREGATION: Group by the second to find the cluster's behaviour.
    # This creates the recovery curve showing the time to full restabilization.
    #
    # Latency and errors are taken from the gateway span only. Each request
    # produces two spans (gateway and backend), so averaging across both would
    # count every request twice and dilute the end-to-end latency the user
    # actually experiences.
    gateway = df[df["operation"] == GATEWAY_OP]

    cluster_metrics = pd.DataFrame({
        "avg_latency_ms": gateway["duration_ms"].resample("1s").mean(),
        "total_errors": gateway["is_error"].resample("1s").sum(),
        # nunique ignores missing values, so failed requests (which never reached
        # a replica) correctly do not inflate the count.
        "active_pods": df["backend_pod"].resample("1s").nunique(),
    }).fillna(0)

    cluster_metrics["total_errors"] = cluster_metrics["total_errors"].astype(int)
    cluster_metrics["active_pods"] = cluster_metrics["active_pods"].astype(int)
    cluster_metrics["avg_latency_ms"] = cluster_metrics["avg_latency_ms"].round(1)

    # 5. Export the clean data
    cluster_metrics.to_csv(output)
    print(f"Success. Extracted, aggregated and saved {len(cluster_metrics)} seconds "
          f"of data to '{output}'.")
    print()
    print(cluster_metrics.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and clean Scenario 4 Kubernetes traces from Jaeger.")
    parser.add_argument("--from-file", default=None,
                        help="read a saved raw_trace_data.json instead of querying Jaeger")
    parser.add_argument("--output", default="k8s_recovery_data.csv",
                        help="output CSV path (default: k8s_recovery_data.csv)")
    args = parser.parse_args()

    extract_k8s_traces(from_file=args.from_file, output=args.output)
