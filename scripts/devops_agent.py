"""Generate an AI DevOps incident report from the extracted recovery data.

Reads the per-second CSV produced by k8s_trace_extractor.py, measures what
actually happened during the incident, and asks a locally hosted Llama model to
write the report.

Requires:
    pip install langchain-ollama
    ollama serve            (and: ollama pull llama3.2)

Usage:
    python3 scripts/devops_agent.py --csv k8s_recovery_data.csv
"""

import argparse
import sys

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# How the incident is judged to be over. These mirror the definition used by
# scripts/build_dataset.py so that both tools report the same recovery time.
HOLD_WINDOW = 10          # seconds that must look healthy
HOLD_REQUIRED = 8         # of which at least this many must be healthy
LATENCY_TOLERANCE = 1.25  # latency may sit this far above baseline and still count


# --- 1. Analyze the CSV Data (The "Brain" before the LLM) ---
def analyze_k8s_data(csv_file):
    """Return (summary, None) on success, or (None, problem_description)."""
    print("Analyzing Kubernetes recovery data...")
    df = pd.read_csv(csv_file)

    if df.empty:
        return None, f"'{csv_file}' contains no rows."

    normal_pods = int(df["active_pods"].max())
    min_pods = int(df["active_pods"].min())

    # A capture where no replica ever served a request cannot describe an
    # incident: there is no baseline to compare against and no recovery to
    # measure. Refuse it rather than inventing a narrative from noise.
    if normal_pods == 0:
        return None, (
            f"'{csv_file}' shows 0 active replicas for the entire capture, so no "
            "healthy baseline exists. This usually means the traffic run never "
            "reached the backend. Re-extract from a complete run."
        )

    # The incident begins at the first sign of trouble: either a replica has
    # dropped out of the load balancer, or requests have started failing.
    trouble = (df["active_pods"] < normal_pods) | (df["total_errors"] > 0)
    if not trouble.any():
        return None, "System remained stable: no replica drops and no errors detected."

    incident_start = int(trouble.idxmax())

    # Baseline is the typical latency before anything went wrong. Seconds with no
    # traffic are excluded, since a zero there means "nothing measured", not
    # "instant response".
    pre = df.iloc[:incident_start]
    pre = pre[(pre["avg_latency_ms"] > 0) & (pre["total_errors"] == 0)]
    if pre.empty:
        return None, (
            f"'{csv_file}' starts already degraded, so there is no healthy period to "
            "use as a baseline. Re-extract with more lead-in before the fault."
        )
    baseline_latency = float(pre["avg_latency_ms"].median())
    threshold = max(baseline_latency * LATENCY_TOLERANCE, baseline_latency + 20)

    # The system is restabilized at the first second from which it stays healthy.
    # A strong majority of the window must be healthy rather than all of it, so
    # that a single noisy second does not move the finish line.
    def healthy(row):
        return (row["active_pods"] >= normal_pods
                and row["total_errors"] == 0
                and 0 < row["avg_latency_ms"] <= threshold)

    recovered_at = None
    for i in range(incident_start, len(df) - HOLD_WINDOW + 1):
        window = df.iloc[i:i + HOLD_WINDOW]
        if not healthy(window.iloc[0]):
            continue
        if sum(healthy(row) for _, row in window.iterrows()) >= HOLD_REQUIRED:
            recovered_at = i
            break

    if recovered_at is None:
        return None, (
            "The capture ends before the system returned to its baseline, so the "
            "recovery time cannot be measured. Extract a longer run."
        )

    incident = df.iloc[incident_start:recovered_at + 1]
    recovery_time_seconds = recovered_at - incident_start
    peak_latency = float(incident["avg_latency_ms"].max())
    failed_requests = int(incident["total_errors"].sum())
    degraded_seconds = int((incident["active_pods"] < normal_pods).sum())

    summary = (
        f"1. Architecture: Kubernetes cluster running {normal_pods} replicas of the "
        f"order backend, one per node, behind a load-balancing Service.\n"
        f"2. Incident: One node was lost. Replicas serving traffic dropped from "
        f"{normal_pods} to {min_pods}, and stayed below full strength for "
        f"{degraded_seconds} seconds.\n"
        f"3. Impact: Average request latency rose from a baseline of "
        f"{baseline_latency:.1f} ms to a peak of {peak_latency:.1f} ms "
        f"({peak_latency / baseline_latency:.1f} times baseline) as the surviving "
        f"replicas absorbed the full traffic load. {failed_requests} requests failed "
        f"outright during the incident.\n"
        f"4. Recovery: The orchestrator detected the loss, scheduled a replacement "
        f"replica, and the system returned to its baseline "
        f"{recovery_time_seconds} seconds after the first sign of trouble."
    )
    return summary, None


# --- 2. Setup the LangChain AI Agent ---
def generate_incident_report(data_summary, model="llama3.2"):
    print(f"\nBooting up local DevOps AI (Ollama - {model})...")

    # Connect to the local Ollama instance. Note the modern import: the older
    # langchain_community.llms.Ollama class is deprecated.
    llm = OllamaLLM(model=model)

    template = """
    You are a Senior Site Reliability Engineer (SRE) at a top-tier tech startup.
    A Kubernetes node failure just occurred in production, and the orchestration
    system auto-healed it.

    Here is the measured data extracted from our OpenTelemetry pipeline:
    {data_summary}

    Write a formal, highly professional "Root Cause & Incident Report" for the
    executive team. Do not use generic filler. Use the exact metrics provided
    above, and do not invent any figure that is not listed.

    The report should include:
    - Executive Summary
    - Incident Timeline & Impact (mention the latency spike)
    - Resolution (mention the recovery time and the orchestrator auto-healing)
    """

    prompt = PromptTemplate(
        input_variables=["data_summary"],
        template=template,
    )

    # Combine the prompt and the LLM
    chain = prompt | llm

    print("Generating formal incident report. Please wait...\n")
    print("======================================================")

    try:
        report = chain.invoke({"data_summary": data_summary})
    except Exception as exc:
        print("Could not reach the local model.")
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("Check that Ollama is installed and running:")
        print("  ollama serve")
        print(f"  ollama pull {model}")
        print("======================================================")
        return False

    print(report)
    print("======================================================")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an incident report from the recovery CSV.")
    parser.add_argument("--csv", default="k8s_recovery_data.csv",
                        help="per-second CSV produced by k8s_trace_extractor.py")
    parser.add_argument("--model", default="llama3.2",
                        help="Ollama model name (e.g. llama3.2, llama3, mistral)")
    parser.add_argument("--summary-only", action="store_true",
                        help="print the measured summary and skip the LLM call")
    args = parser.parse_args()

    summary_stats, problem = analyze_k8s_data(args.csv)
    if problem:
        print(f"\n{problem}")
        sys.exit(1)

    print("\nMeasured summary passed to the model:")
    print("------------------------------------------------------")
    print(summary_stats)
    print("------------------------------------------------------")

    if not args.summary_only:
        generate_incident_report(summary_stats, model=args.model)
