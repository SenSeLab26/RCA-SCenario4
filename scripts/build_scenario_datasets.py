"""Build one dataset per scenario, instead of a single pooled dataset.

Each scenario owns its faults, so each scenario gets its own CSV and its own
model. This keeps the four scenarios independent: a result reported for
Scenario 1 rests only on Scenario 1 data.

    Scenario 1  Resource exhaustion   memory_leak, cpu_saturation
    Scenario 2  Dependency failure    dependency_slow, dependency_down, config_error
    Scenario 3  Network degradation   network_partition, packet_loss
    Scenario 4  Orchestrator recovery node_failure, pod_failure

Every row is one second of one run, and every dataset uses the same columns, so
the same evaluation code works on all four. The `data_source` column records
whether a row was measured on the cluster or generated, so the two can never be
confused in a results table.

Usage:
    python3 scripts/build_scenario_datasets.py
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_rca_classifier import load_run  # reuse the per-second feature extraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Which scenario each fault belongs to.
FAULT_TO_SCENARIO = {
    "memory_leak": 1,
    "cpu_saturation": 1,
    "dependency_slow": 2,
    "dependency_down": 2,
    "config_error": 2,
    "network_partition": 3,
    "packet_loss": 3,
    "node_failure": 4,
    "pod_failure": 4,
}

SCENARIO_NAMES = {
    1: "resource_exhaustion",
    2: "dependency_failure",
    3: "network_degradation",
    4: "orchestrator_recovery",
}

COLUMNS = [
    "run_id", "fault_type", "scenario", "data_source", "t_rel",
    "req_count", "err_rate", "p50_ms", "p95_ms", "max_ms",
    "active_pods", "p50_ratio", "tail_ratio",
]


def rows_for_run(run_dir):
    """Return (list_of_output_rows, fault, run_id) for one run directory."""
    rows, fault, run_id = load_run(run_dir)
    if not rows or fault not in FAULT_TO_SCENARIO:
        return None, fault, run_id

    scenario = FAULT_TO_SCENARIO[fault]
    out = []
    for row in rows:
        out.append({
            "run_id": run_id,
            "fault_type": fault,
            "scenario": scenario,
            "data_source": "measured",
            "t_rel": round(row["t_rel"], 1),
            "req_count": row["sent"],
            "err_rate": round(row["err_rate"], 4),
            "p50_ms": row["p50_ms"],
            "p95_ms": row["p95_ms"],
            "max_ms": row["max_ms"],
            "active_pods": row["distinct_pods"],
            "p50_ratio": round(row["p50_ratio"], 3),
            "tail_ratio": round(row["tail_ratio"], 3),
        })
    return out, fault, run_id


def main():
    parser = argparse.ArgumentParser(description="Build one dataset per scenario.")
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "*"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    by_scenario = {}
    skipped = []

    for run_dir in sorted(glob.glob(args.runs)):
        rows, fault, run_id = rows_for_run(run_dir)
        if not rows:
            skipped.append((os.path.basename(run_dir.rstrip("/")), fault))
            continue
        by_scenario.setdefault(FAULT_TO_SCENARIO[fault], []).append((run_id, fault, rows))

    if not by_scenario:
        raise SystemExit("No usable runs found. Each run needs loadgen_metrics.csv "
                         "and chaos_event.json.")

    print("PER-SCENARIO DATASETS")
    print("=" * 78)

    for scenario in sorted(SCENARIO_NAMES):
        entries = by_scenario.get(scenario, [])
        name = SCENARIO_NAMES[scenario]
        path = os.path.join(args.out_dir, f"scenario{scenario}_{name}.csv")

        if not entries:
            print(f"\nScenario {scenario} ({name}): no runs recorded yet.")
            faults = [f for f, s in FAULT_TO_SCENARIO.items() if s == scenario]
            print(f"  Record one with: inject_fault.py --fault {faults[0]}")
            continue

        all_rows = [r for _, _, rows in entries for r in rows]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)

        faults_present = sorted({fault for _, fault, _ in entries})
        print(f"\nScenario {scenario} ({name})")
        print(f"  file      : {os.path.relpath(path, ROOT)}")
        print(f"  runs      : {len(entries)}")
        print(f"  rows      : {len(all_rows)}")
        print(f"  faults    : {', '.join(faults_present)}")
        for fault in faults_present:
            runs_of = [rid for rid, f, _ in entries if f == fault]
            print(f"      {fault:<20} {len(runs_of)} run(s)")

        # A model can only be tested on unseen data if a whole run can be held
        # back, which needs at least two runs of every fault in the scenario.
        thin = [f for f in faults_present
                if len([1 for _, ff, _ in entries if ff == f]) < 2]
        if thin:
            print(f"  WARNING   : only one run for {', '.join(thin)}. "
                  "Record a second before quoting any accuracy.")

    if skipped:
        print(f"\nSkipped {len(skipped)} run(s) with no usable data or an unknown fault:")
        for run_id, fault in skipped:
            print(f"  {run_id:<34} fault={fault}")

    print("\nNext: python3 scripts/evaluate_scenario_models.py")


if __name__ == "__main__":
    main()
