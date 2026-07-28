"""Step 6: train the AI forecaster to predict how long the healing will take.

Scenarios 1-3 trained a *classifier* ("is this trace a crash?"). This one trains
a *regressor* on the recovery curve. Every second of the incident is one training
example:

    features -> what the dashboard looks like right now
    label    -> how many more seconds until the system is fully stable

This is the same framing as remaining-useful-life estimation, and it is what
lets the model answer the question the scenario asks: given only the first few
seconds of a latency spike, how long until we are back to normal?

Two feature sets are evaluated deliberately:

  full        - includes `time_since_fault`, i.e. the model knows how long ago
                the node died. Realistic, since Kubernetes tells you that.
  signal-only - excludes it, so the model must read the *shape* of the latency
                and replica-count curves alone. This is the honest check that
                the model learned the recovery dynamics rather than just reading
                a clock.

Usage:
    python3 scripts/forecast_recovery.py
    python3 scripts/forecast_recovery.py --llm     # add an Ollama narrative
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FULL_FEATURES = [
    "p50_ms", "p95_ms", "max_ms", "mean_ms", "mean_queue_wait_ms",
    "err_rate", "err_rate_roll5", "active_pods", "pods_missing",
    "p95_ratio", "p95_slope_5s", "req_count", "time_since_fault",
]
SIGNAL_FEATURES = [f for f in FULL_FEATURES if f != "time_since_fault"]

TARGET = "seconds_to_restabilize"


def build_models():
    return {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
    }


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def evaluate(df, features, label):
    """Leave-one-run-out when we have several runs; chronological split when we don't."""
    runs = df["run_id"].unique()
    X = df[features].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    groups = df["run_id"].to_numpy()

    print(f"\n--- Feature set: {label} ({len(features)} features) ---")

    results = {}

    if len(runs) >= 2:
        n_splits = min(len(runs), 5)
        splitter = GroupKFold(n_splits=n_splits)
        print(f"Validation: leave-one-run-out across {len(runs)} runs ({n_splits} folds)")
        for name, model in build_models().items():
            maes, rmses = [], []
            for train_idx, test_idx in splitter.split(X, y, groups):
                model.fit(X[train_idx], y[train_idx])
                predicted = model.predict(X[test_idx])
                maes.append(mean_absolute_error(y[test_idx], predicted))
                rmses.append(rmse(y[test_idx], predicted))
            results[name] = (float(np.mean(maes)), float(np.mean(rmses)))
            print(f"  {name:<18} MAE {np.mean(maes):6.2f}s   RMSE {np.mean(rmses):6.2f}s")
    else:
        # Only one incident to learn from. Split it in time so we at least never
        # test on a second we trained on.
        ordered = df.sort_values("time_since_fault")
        cut = int(len(ordered) * 0.6)
        X_train = ordered[features].to_numpy(dtype=float)[:cut]
        y_train = ordered[TARGET].to_numpy(dtype=float)[:cut]
        X_test = ordered[features].to_numpy(dtype=float)[cut:]
        y_test = ordered[TARGET].to_numpy(dtype=float)[cut:]
        print("Validation: chronological 60/40 split within a single run")
        print("  WARNING: one run means one recovery curve. These numbers show the")
        print("           model can fit this incident, not that it generalises to the")
        print("           next one. Collect 3+ runs for a trustworthy MAE.")
        for name, model in build_models().items():
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            results[name] = (
                float(mean_absolute_error(y_test, predicted)),
                float(rmse(y_test, predicted)),
            )
            print(f"  {name:<18} MAE {results[name][0]:6.2f}s   RMSE {results[name][1]:6.2f}s")

    return results


def plot_run(df_run, model, features, run_id, out_path):
    """Latency curve on top, predicted vs actual remaining time underneath."""
    incident = df_run.dropna(subset=[TARGET]).sort_values("time_since_fault")
    if incident.empty:
        print("No labelled rows to plot.")
        return

    predicted = model.predict(incident[features].to_numpy(dtype=float))
    baseline = df_run["baseline_p95_ms"].dropna()
    baseline_p95 = float(baseline.iloc[0]) if not baseline.empty else None

    all_rows = df_run.sort_values("second_epoch")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax_top.plot(all_rows["t_rel"], all_rows["p95_ms"], color="#1f6feb",
                linewidth=1.6, label="p95 latency (ms)")
    ax_top.fill_between(all_rows["t_rel"], 0, all_rows["err_count"] * 0 + all_rows["p95_ms"].max(),
                        where=all_rows["err_count"] > 0, color="#d1242f", alpha=0.10,
                        label="seconds with errors")
    if baseline_p95:
        ax_top.axhline(baseline_p95, color="#2da44e", linestyle="--",
                       linewidth=1.2, label=f"baseline p95 ({baseline_p95:.0f} ms)")
    ax_top.axvline(0, color="#d1242f", linewidth=1.6, label="node killed")
    restabilized_at = float(incident["time_since_fault"].iloc[0] + incident[TARGET].iloc[0])
    ax_top.axvline(restabilized_at, color="#2da44e", linewidth=1.6,
                   label=f"restabilized (T+{restabilized_at:.0f}s)")

    ax_twin = ax_top.twinx()
    ax_twin.plot(all_rows["t_rel"], all_rows["active_pods"], color="#8250df",
                 linewidth=1.2, alpha=0.8, drawstyle="steps-post")
    ax_twin.set_ylabel("replicas serving traffic", color="#8250df")
    ax_twin.set_ylim(0, 4)
    ax_twin.tick_params(axis="y", colors="#8250df")

    ax_top.set_ylabel("p95 latency (ms)")
    ax_top.set_title(f"Kubernetes node failure and recovery - {run_id}")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(alpha=0.2)

    ax_bot.plot(incident["time_since_fault"], incident[TARGET], color="#57606a",
                linewidth=2.0, label="actual seconds to restabilization")
    ax_bot.plot(incident["time_since_fault"], predicted, color="#bf8700",
                linewidth=1.8, linestyle="--", label="AI forecast")
    ax_bot.axvline(0, color="#d1242f", linewidth=1.6)
    ax_bot.set_xlabel("seconds since node failure")
    ax_bot.set_ylabel("seconds remaining")
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved chart: {out_path}")


def early_prediction_demo(df_run, model, features, offsets):
    """The output an on-call engineer would actually see."""
    print("\n" + "=" * 70)
    print("LIVE FORECAST - what the AI says early in the incident")
    print("=" * 70)

    incident = df_run.dropna(subset=[TARGET]).sort_values("time_since_fault")
    if incident.empty:
        print("No labelled incident rows available.")
        return

    for offset in offsets:
        row = incident[incident["time_since_fault"] == offset]
        if row.empty:
            continue
        predicted = float(model.predict(row[features].to_numpy(dtype=float))[0])
        actual = float(row[TARGET].iloc[0])
        p95 = float(row["p95_ms"].iloc[0])
        pods = int(row["active_pods"].iloc[0])
        errors = int(row["err_count"].iloc[0])

        print(f"\n[T+{offset:>3.0f}s]  p95={p95:.0f}ms  replicas_serving={pods}  errors={errors}")
        print(f"          Node failure detected. Orchestrator rerouting traffic.")
        print(f"          Estimated time to full system restabilization: {predicted:.0f} seconds.")
        print(f"          (ground truth: {actual:.0f}s, error {predicted - actual:+.0f}s)")


def llm_narrative(df_run, run_id, restabilized_at):
    """Optional: hand the numbers to a local Ollama model, as in Scenario 1."""
    incident = df_run[(df_run["t_rel"] >= 0) & (df_run["t_rel"] <= restabilized_at)]
    baseline = df_run["baseline_p95_ms"].dropna()
    baseline_p95 = float(baseline.iloc[0]) if not baseline.empty else 0.0

    prompt = f"""
You are an expert Site Reliability Engineer. A Kubernetes cluster running three
replicas of an order backend lost one node under heavy traffic. Here is the
telemetry from the incident:

- Baseline p95 latency before the failure: {baseline_p95:.0f} ms
- Peak p95 latency during the incident: {incident['p95_ms'].max():.0f} ms
- Failed requests during the incident: {int(incident['err_count'].sum())}
- Seconds where fewer than 3 replicas served traffic: {int((incident['active_pods'] < 3).sum())}
- Total time to full restabilization: {restabilized_at:.0f} seconds

Write a short, professional incident report of at most four sentences. Explain
what the orchestrator did, why latency spiked before it recovered, and one
concrete change that would shorten the recovery next time.
"""

    print("\n[AI Assistant] Generating incident narrative via local Ollama...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        print("\nINCIDENT REPORT")
        print("-" * 70)
        print(response.json()["response"].strip())
        print("-" * 70)
    except requests.exceptions.RequestException as exc:
        print(f"[skipped] Could not reach Ollama at localhost:11434 ({exc}).")


def main():
    parser = argparse.ArgumentParser(description="Train a recovery-time forecaster on the pooled dataset.")
    parser.add_argument("--dataset", default=os.path.join(ROOT, "data", "recovery_dataset.csv"))
    parser.add_argument("--run-id", default=None,
                        help="which run to chart and demo (default: the most recent)")
    parser.add_argument("--offsets", default="5,10,15,20,30",
                        help="seconds after the fault to demo the live forecast at")
    parser.add_argument("--llm", action="store_true", help="also ask a local Ollama for a narrative")
    parser.add_argument("--out", default=os.path.join(ROOT, "recovery_forecast.png"))
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"ERROR: {args.dataset} not found. Run build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.dataset)
    labelled = df.dropna(subset=[TARGET]).copy()
    if labelled.empty:
        print("ERROR: the dataset has no labelled rows. Did any run reach restabilization?",
              file=sys.stderr)
        sys.exit(1)

    runs = sorted(df["run_id"].unique())
    print(f"Dataset  : {args.dataset}")
    print(f"Rows     : {len(df)} total, {len(labelled)} labelled incident seconds")
    print(f"Runs     : {len(runs)} -> {', '.join(runs)}")
    print("\nRecovery time per run (the thing we are predicting):")
    for run_id, group in labelled.groupby("run_id"):
        print(f"  {run_id:<34} {group[TARGET].max():>5.0f}s "
              f"({group['chaos_mode'].iloc[0] or 'unknown'} kill, "
              f"{len(group)} labelled seconds)")

    full_results = evaluate(labelled, FULL_FEATURES, "full")
    signal_results = evaluate(labelled, SIGNAL_FEATURES, "signal-only (no clock)")

    best_name = min(full_results, key=lambda name: full_results[name][0])
    print(f"\nBest model on the full feature set: {best_name} "
          f"(MAE {full_results[best_name][0]:.2f}s)")
    print(f"Same model without the clock feature: MAE {signal_results[best_name][0]:.2f}s "
          "- how well it reads the curve itself.")

    # Refit the winner on everything for the chart and the live demo.
    final = build_models()[best_name]
    final.fit(labelled[FULL_FEATURES].to_numpy(dtype=float),
              labelled[TARGET].to_numpy(dtype=float))

    if hasattr(final, "feature_importances_"):
        print("\nWhat the model actually looks at:")
        ranked = sorted(zip(FULL_FEATURES, final.feature_importances_),
                        key=lambda pair: pair[1], reverse=True)
        for name, importance in ranked[:8]:
            print(f"  {name:<22} {importance:.3f}  {'#' * int(importance * 50)}")

    target_run = args.run_id or runs[-1]
    df_run = df[df["run_id"] == target_run].copy()
    print(f"\nCharting and demoing run: {target_run}")

    offsets = [float(x) for x in args.offsets.split(",") if x.strip()]
    early_prediction_demo(df_run, final, FULL_FEATURES, offsets)
    plot_run(df_run, final, FULL_FEATURES, target_run, args.out)

    if args.llm:
        run_labelled = df_run.dropna(subset=[TARGET])
        if not run_labelled.empty:
            restabilized_at = float(
                run_labelled["time_since_fault"].iloc[0] + run_labelled[TARGET].iloc[0]
            )
            llm_narrative(df_run, target_run, restabilized_at)


if __name__ == "__main__":
    main()
