"""Scenario 4: train and evaluate the models on the Scenario 4 dataset.

Two questions, so two kinds of model, reported with the agreed metrics.

  Classification  Which fault is this - a lost node or a lost pod?
                  Accuracy, precision, recall, F1 and a confusion matrix.

  Regression      How many more seconds until the system is fully stable?
                  MAE, RMSE and R squared.

The recovery label is derived here rather than stored, so the rule that defines
"recovered" is visible in one place: all replicas serving again, the error rate
back to its baseline, and response time back near baseline, sustained for a
number of seconds.

Validation is leave-one-run-out. Each run is one complete incident, so the model
is always scored on an incident it has never seen. Seconds inside one incident
are highly similar to each other, so a random split would let the model see
almost every test row during training and every score would be inflated.

Usage:
    python3 scripts/evaluate_scenario_models.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES = ["req_count", "err_rate", "p50_ms", "p95_ms", "max_ms",
            "active_pods", "p50_ratio", "tail_ratio"]

EXPECTED_REPLICAS = 3
HOLD_WINDOW = 10        # seconds that must look healthy
HOLD_REQUIRED = 8       # of which at least this many must be healthy
LATENCY_TOLERANCE = 1.25


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def add_recovery_label(df):
    """Work out, for every second of every incident, how long recovery has left."""
    labelled = []
    for run_id, group in df.groupby("run_id", sort=True):
        group = group.sort_values("t_rel").copy()

        pre = group[group["t_rel"] < 0]
        if pre.empty:
            continue
        baseline_p50 = float(pre["p50_ms"].median())
        baseline_err = float(pre["err_rate"].mean())
        latency_limit = max(baseline_p50 * LATENCY_TOLERANCE, baseline_p50 + 20)
        error_limit = max(baseline_err * 2.0, 0.02)

        incident = group[group["t_rel"] >= 0].reset_index(drop=True)
        if incident.empty:
            continue

        healthy = ((incident["active_pods"] >= EXPECTED_REPLICAS)
                   & (incident["err_rate"] <= error_limit)
                   & (incident["p50_ms"] <= latency_limit)).to_numpy()

        recovered_at = None
        for i in range(len(healthy) - HOLD_WINDOW + 1):
            window = healthy[i:i + HOLD_WINDOW]
            if window[0] and window.sum() >= HOLD_REQUIRED:
                recovered_at = float(incident["t_rel"].iloc[i])
                break

        group["seconds_to_restabilize"] = np.nan
        if recovered_at is not None:
            mask = (group["t_rel"] >= 0) & (group["t_rel"] <= recovered_at)
            group.loc[mask, "seconds_to_restabilize"] = (
                recovered_at - group.loc[mask, "t_rel"]).clip(lower=0)
        labelled.append(group)

    return pd.concat(labelled, ignore_index=True) if labelled else pd.DataFrame()


def evaluate_classification(df):
    print("\n" + "=" * 74)
    print("CLASSIFICATION - which fault is this?")
    print("=" * 74)

    incident = df[df["t_rel"] >= 0]
    faults = sorted(incident["fault_type"].unique())
    if len(faults) < 2:
        print(f"Only one fault type present ({faults[0]}), so there is nothing to")
        print("classify. Record runs of another Scenario 4 fault first.")
        return

    X = incident[FEATURES].to_numpy(dtype=float)
    y = incident["fault_type"].to_numpy()
    groups = incident["run_id"].to_numpy()

    runs_per_fault = {f: incident[incident["fault_type"] == f]["run_id"].nunique()
                      for f in faults}
    print(f"Classes    : {', '.join(f'{f} ({n} runs)' for f, n in runs_per_fault.items())}")
    print(f"Rows       : {len(X)} incident seconds")
    print(f"Validation : leave-one-run-out, {len(set(groups))} folds")
    print(f"Features   : {', '.join(FEATURES)}\n")

    if min(runs_per_fault.values()) < 2:
        thin = [f for f, n in runs_per_fault.items() if n < 2]
        print(f"WARNING: only one run for {', '.join(thin)}. Holding that run back")
        print("         leaves the model with no example of the class at all, so the")
        print("         scores below understate what it could do with more data.\n")

    # Macro averaging weights every fault equally, so a class with fewer seconds
    # is not drowned out by a class with more.
    print(f"{'model':<20} {'accuracy':>9} {'precision':>10} {'recall':>8} {'F1':>7}")
    print("-" * 74)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42,
                                               class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=4,
                                               class_weight="balanced"),
        "LogisticRegression": LogisticRegression(max_iter=3000,
                                                 class_weight="balanced"),
    }
    splitter = LeaveOneGroupOut()
    results = {}
    for name, model in models.items():
        true_all, pred_all = [], []
        verdicts = []
        for train_idx, test_idx in splitter.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            predicted = model.predict(X[test_idx])
            true_all.extend(y[test_idx])
            pred_all.extend(predicted)
            values, counts = np.unique(predicted, return_counts=True)
            verdicts.append((groups[test_idx][0], y[test_idx][0],
                             values[counts.argmax()]))
        results[name] = {
            "accuracy": accuracy_score(true_all, pred_all),
            "precision": precision_score(true_all, pred_all, average="macro",
                                         zero_division=0),
            "recall": recall_score(true_all, pred_all, average="macro",
                                   zero_division=0),
            "f1": f1_score(true_all, pred_all, average="macro", zero_division=0),
            "true": true_all, "pred": pred_all, "verdicts": verdicts,
        }
        r = results[name]
        print(f"{name:<20} {r['accuracy']:>9.3f} {r['precision']:>10.3f} "
              f"{r['recall']:>8.3f} {r['f1']:>7.3f}")

    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\nBest by F1 (macro): {best}")

    matrix = confusion_matrix(results[best]["true"], results[best]["pred"], labels=faults)
    print(f"\nConfusion matrix for {best} (rows = actual, columns = predicted):")
    header = "".join(f"{f[:14]:>16}" for f in faults)
    print(f"{'':>18}{header}")
    for i, fault in enumerate(faults):
        print(f"{fault:>18}" + "".join(f"{v:>16}" for v in matrix[i]))

    print("\nOne verdict per incident, which is what a diagnosis system actually gives:")
    correct = 0
    for run_id, truth, verdict in results[best]["verdicts"]:
        mark = "correct" if verdict == truth else "WRONG"
        correct += verdict == truth
        print(f"  {run_id:<34} actual={truth:<14} predicted={verdict:<14} {mark}")
    print(f"\n  Incidents identified correctly: {correct}/{len(results[best]['verdicts'])}")

    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=faults).plot(
        cmap="Blues", xticks_rotation=30)
    plt.title(f"Scenario 4 - fault classification ({best})")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "scenario4_confusion_matrix.png"), dpi=140)
    print("\nSaved scenario4_confusion_matrix.png")


def evaluate_regression(df):
    print("\n" + "=" * 74)
    print("REGRESSION - how many seconds until the system is stable again?")
    print("=" * 74)

    labelled = df.dropna(subset=["seconds_to_restabilize"])
    if labelled.empty:
        print("No run reached a stable state inside its recording window, so there is")
        print("no recovery time to predict. Record longer runs.")
        return

    runs = sorted(labelled["run_id"].unique())
    print("Recovery time measured per incident:")
    for run_id in runs:
        of_run = labelled[labelled["run_id"] == run_id]
        print(f"  {run_id:<34} {of_run['seconds_to_restabilize'].max():>5.0f} s "
              f"({of_run['fault_type'].iloc[0]})")

    if len(runs) < 2:
        print("\nOnly one incident recovered, so a model cannot be tested on an unseen")
        print("incident. Record more runs before quoting these numbers.")
        return

    X = labelled[FEATURES + ["t_rel"]].to_numpy(dtype=float)
    y = labelled["seconds_to_restabilize"].to_numpy(dtype=float)
    groups = labelled["run_id"].to_numpy()

    print(f"\nRows       : {len(X)} labelled seconds")
    print(f"Validation : leave-one-run-out, {len(runs)} folds\n")
    print(f"{'model':<22} {'MAE (s)':>9} {'RMSE (s)':>10} {'R2':>8}")
    print("-" * 74)

    splitter = LeaveOneGroupOut()
    for name, model in {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
    }.items():
        true_all, pred_all = [], []
        for train_idx, test_idx in splitter.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            pred_all.extend(model.predict(X[test_idx]))
            true_all.extend(y[test_idx])
        print(f"{name:<22} {mean_absolute_error(true_all, pred_all):>9.2f} "
              f"{rmse(true_all, pred_all):>10.2f} {r2_score(true_all, pred_all):>8.3f}")

    print("\nNote: recovery times differ substantially between incidents, and with a")
    print("small number of incidents the model has few examples of each length. These")
    print("numbers should be read as a first result, not a settled one.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Scenario 4 models.")
    parser.add_argument("--dataset",
                        default=os.path.join(ROOT, "data",
                                             "scenario4_orchestrator_recovery.csv"))
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.dataset)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: {args.dataset} not found. Run "
                         "scripts/build_scenario_datasets.py first.")

    print("SCENARIO 4: ORCHESTRATOR RECOVERY - MODEL EVALUATION")
    print(f"Dataset : {os.path.relpath(args.dataset, ROOT)}")
    print(f"Rows    : {len(df)}   Runs: {df['run_id'].nunique()}   "
          f"Faults: {', '.join(sorted(df['fault_type'].unique()))}")

    df = add_recovery_label(df)
    if df.empty:
        raise SystemExit("No run had both a healthy baseline and an incident. Record "
                         "runs with at least 20 seconds of traffic before the fault.")

    evaluate_classification(df)
    evaluate_regression(df)

    print("\n" + "=" * 74)
    print("Done. Every score came from an incident the model had never seen.")


if __name__ == "__main__":
    main()
