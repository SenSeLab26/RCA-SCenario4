"""Multi-class Root Cause Analysis: given the telemetry, which fault is this?

Reads every run in runs/, labels each one with the fault that was injected, and
trains one classifier to tell the faults apart from the telemetry alone.

It reads `loadgen_metrics.csv`, which the load generator writes directly for
every run. No Jaeger extraction step is needed, and nothing outside this folder
is touched - Scenarios 1, 2 and 3 stay independently runnable in their own repos.

Usage:
    python3 scripts/train_rca_classifier.py
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import LeaveOneGroupOut

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The features an SRE would read off a dashboard. All come straight from the
# load generator, so they exist for every run without any extra processing.
FEATURES = ["p50_ms", "p95_ms", "max_ms", "err_rate", "distinct_pods",
            "sent", "p50_ratio", "tail_ratio"]

# inject_chaos.py (the original) writes "mode"; inject_fault.py writes "fault".
# Normalise both onto one vocabulary.
MODE_TO_FAULT = {"node": "node_failure", "pod": "pod_failure"}


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def load_run(run_dir):
    """Return (rows, fault_type, run_id) for one run, or (None, None, None)."""
    metrics_path = os.path.join(run_dir, "loadgen_metrics.csv")
    chaos_path = os.path.join(run_dir, "chaos_event.json")
    if not (os.path.exists(metrics_path) and os.path.exists(chaos_path)):
        return None, None, None

    with open(chaos_path, encoding="utf-8") as handle:
        chaos = json.load(handle)
    fault = chaos.get("fault") or MODE_TO_FAULT.get(chaos.get("mode"), chaos.get("mode"))
    fault_at = chaos.get("fault_at")
    if not (fault and fault_at):
        return None, None, None

    with open(metrics_path, encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))

    # Time every row relative to the fault ourselves, rather than trusting the
    # t_rel column: it is only filled in when the load generator injected the
    # fault itself, and some runs were injected out of band.
    rows = []
    for record in raw:
        sent = int(record["sent"])
        if sent == 0:
            continue  # a second with no traffic measures nothing
        rows.append({
            "t_rel": float(record["second_epoch"]) - fault_at,
            "p50_ms": float(record["p50_ms"]),
            "p95_ms": float(record["p95_ms"]),
            "max_ms": float(record["max_ms"]),
            "err_rate": int(record["err"]) / sent,
            "distinct_pods": int(record["distinct_pods"]),
            "sent": sent,
        })

    if not rows:
        return None, None, None

    # Two derived features, both cheap and both genuinely discriminating.
    baseline = median([r["p50_ms"] for r in rows if r["t_rel"] < 0]) or rows[0]["p50_ms"]
    for row in rows:
        row["p50_ratio"] = row["p50_ms"] / baseline if baseline else 1.0
        # How much worse the unlucky request is than the typical one. A high
        # ratio means "most requests fine, a few timing out" (a dead endpoint);
        # a low ratio means "everything is uniformly slow" (a queue).
        row["tail_ratio"] = row["p95_ms"] / row["p50_ms"] if row["p50_ms"] else 1.0

    return rows, fault, os.path.basename(run_dir.rstrip("/"))


def build_dataset(runs_glob):
    X, y, groups = [], [], []
    per_run = {}

    for run_dir in sorted(glob.glob(runs_glob)):
        rows, fault, run_id = load_run(run_dir)
        if not rows:
            continue
        # Classify the incident, so only seconds at or after the fault.
        incident = [r for r in rows if r["t_rel"] >= 0]
        if not incident:
            continue
        for row in incident:
            X.append([row[f] for f in FEATURES])
            y.append(fault)
            groups.append(run_id)
        per_run[run_id] = (fault, len(incident))

    return np.array(X, dtype=float), np.array(y), np.array(groups), per_run


def main():
    parser = argparse.ArgumentParser(description="Train the multi-class RCA classifier.")
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "*"))
    parser.add_argument("--out", default=os.path.join(ROOT, "rca_confusion_matrix.png"))
    args = parser.parse_args()

    X, y, groups, per_run = build_dataset(args.runs)

    if len(per_run) == 0:
        raise SystemExit("No usable runs found. Each run needs loadgen_metrics.csv "
                         "and chaos_event.json.")

    print("Runs found:")
    for run_id, (fault, count) in sorted(per_run.items()):
        print(f"  {run_id:<32} {fault:<18} {count:>4} incident seconds")

    faults = sorted(set(y))
    print(f"\nFault types: {len(faults)} -> {', '.join(faults)}")
    print(f"Rows: {len(X)}")

    if len(faults) < 2:
        raise SystemExit("\nOnly one fault type present, so there is nothing to "
                         "classify yet. Record runs for other faults with "
                         "scripts/inject_fault.py.")

    # Group by run so the model is always tested on an incident it never saw.
    # Testing on seconds from a run it trained on would be self-deception: the
    # seconds within one run are highly similar to each other.
    #
    # Leave *one run* out per fold, rather than splitting the runs into k groups.
    # With only two or three runs per fault, a k-way split can put every run of a
    # class in the test fold, leaving the model unable to predict a class it was
    # never shown - which scores it at zero for reasons that are the validation's
    # fault, not the model's.
    runs_per_fault = {f: len({g for g, label in zip(groups, y) if label == f}) for f in faults}
    n_splits = min(runs_per_fault.values())

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")

    if n_splits < 2:
        thin = [f for f, n in runs_per_fault.items() if n < 2]
        print(f"\nWARNING: only one run for: {', '.join(thin)}.")
        print("         Cannot hold a whole run back for those, so the score below")
        print("         is optimistic. Record a second run of each to fix this.")
        model.fit(X, y)
        predictions = model.predict(X)
        print(f"\nIn-sample accuracy (NOT a real score): {accuracy_score(y, predictions) * 100:.1f}%")
    else:
        splitter = LeaveOneGroupOut()
        print(f"\nValidation: leave-one-run-out, {splitter.get_n_splits(groups=groups)} folds")
        all_true, all_pred = [], []
        verdicts = []
        for train_idx, test_idx in splitter.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            predicted = model.predict(X[test_idx])
            all_pred.extend(predicted)
            all_true.extend(y[test_idx])

            # An RCA system gives one answer per incident, not per second, so
            # also record the majority vote across the held-out run.
            held_out = groups[test_idx][0]
            truth = y[test_idx][0]
            values, counts = np.unique(predicted, return_counts=True)
            verdict = values[counts.argmax()]
            share = counts.max() / counts.sum()
            verdicts.append((held_out, truth, verdict, share))

        print(f"\nPer-second accuracy on unseen incidents: "
              f"{accuracy_score(all_true, all_pred) * 100:.1f}%\n")
        print(classification_report(all_true, all_pred, zero_division=0))

        print("Per-incident verdict (the answer an RCA system would actually give):")
        correct = 0
        for run_id, truth, verdict, share in verdicts:
            mark = "correct" if verdict == truth else "WRONG"
            correct += verdict == truth
            print(f"  {run_id:<32} actual={truth:<14} predicted={verdict:<14} "
                  f"({share * 100:.0f}% of seconds)  {mark}")
        print(f"\n  Incidents identified correctly: {correct}/{len(verdicts)}")

        ConfusionMatrixDisplay.from_predictions(
            all_true, all_pred, labels=faults, xticks_rotation=45, cmap="Blues")
        plt.title("Multi-class RCA: which fault is this?")
        plt.tight_layout()
        plt.savefig(args.out, dpi=140)
        print(f"Saved confusion matrix: {args.out}")

    model.fit(X, y)
    print("\nWhat the model looks at:")
    for name, importance in sorted(zip(FEATURES, model.feature_importances_),
                                   key=lambda pair: pair[1], reverse=True):
        print(f"  {name:<16} {importance:.3f}  {'#' * int(importance * 50)}")


if __name__ == "__main__":
    main()
