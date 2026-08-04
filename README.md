# Scenario 4: Kubernetes Fault Tolerance, Rerouting and Restabilization

Scenarios 1-3 broke an application. This one breaks the **infrastructure
underneath it**, and then measures how long the orchestrator takes to heal.

We run a real Kubernetes cluster with an order backend replicated across three
nodes behind a load balancer, drive continuous traffic through it, and then
violently kill one of the nodes mid-flight. Kubernetes reroutes traffic to the
survivors, notices the node is gone, and schedules a replacement. Every
millisecond of that is recorded by OpenTelemetry.

The AI question is different from the earlier scenarios too. Instead of
classifying "is this a crash?", we train a **regressor** on the recovery curve
to answer: *given the first few seconds of a latency spike, how long until the
system is fully stable again?*

---

## 1. What actually happens during a run

A run is five minutes of steady traffic with a node killed 60 seconds in. The
telemetry goes through four distinct phases:

| Phase | Roughly | What it looks like | Why |
| --- | --- | --- | --- |
| **baseline** | T-60 → T+0 | p50 ~110 ms, no errors, 3 replicas serving | Healthy. |
| **blackhole** | T+0 → T+45 | ~1 in 3 requests fails after a 5 s hang; p50 stays normal | The node is dead but Kubernetes doesn't know yet. The Service still lists the dead pod as an endpoint, so a third of traffic is routed into a hole. |
| **brownout** | T+45 → T+75 | No errors, but p50 climbs 110 ms → ~1900 ms | The endpoint has been removed. Two replicas now absorb all the traffic with two thirds of the capacity, so a queue builds. |
| **recovering → stable** | T+75 → T+88 | Third replica appears, latency decays back to baseline | The replacement pod is scheduled, warms up, becomes Ready, and the backlog drains. |

In the measured run in this repo, full restabilization took **88 seconds**, and
p95 peaked at **24× baseline**.

The blackhole phase is the part people find surprising, and it is the most
valuable thing this scenario teaches: **Kubernetes does not detect a dead node
instantly.** The node controller waits out a grace period (~40 s by default)
before marking the node's pods unhealthy. Until then, the load balancer keeps
confidently routing traffic to a machine that no longer exists.

---

## 2. The cluster

```text
                    kind cluster "rca4" (5 Docker containers)

  rca4-control-plane        etcd, apiserver, scheduler, controller-manager
                            (the node controller here is what detects the fault)

  rca4-worker   [infra]     Jaeger  <---- OTLP spans -------+
                            api-gateway  :30080             |
                                 |                          |
                                 |  ClusterIP Service       |
                                 |  (the load balancer)     |
                    +------------+------------+             |
                    |            |            |             |
  rca4-worker2 [app] |  rca4-worker3 [app] |  rca4-worker4 [app]
   order-backend     |   order-backend     |   order-backend  --> spans
      replica 1      |      replica 2      |      replica 3
                                                    ^
                                                    |
                                            we kill this node
```

Host access: gateway on `localhost:30080`, Jaeger UI on `localhost:30686`.

A few design choices are load-bearing, and each one is commented in the file it
lives in:

- **The backend has bounded concurrency** (`WORKER_SLOTS` requests at a time,
  each taking `SERVICE_TIME_MS`). Without a capacity limit, losing a replica
  would only reduce throughput, never increase latency, and there would be no
  brownout to observe.
- **Jaeger and the gateway live on their own node**, not on the control plane.
  They must survive the fault, and they must not compete for CPU with the
  components that get busy *because* of the fault.
- **`Connection: close` on every upstream call.** Kubernetes balances per
  connection, so a pooled keep-alive connection would pin to one pod and we'd
  never see traffic reroute.
- **Preferred, not required, pod anti-affinity.** Replicas spread one-per-node
  while healthy, but the replacement is allowed to double up on a survivor -
  otherwise it would sit `Pending` forever and the system could never recover.
- **10-second unreachable toleration.** The Kubernetes default is 300 s, which
  would stall the replacement for five minutes.

---

## 3. Choosing the load: the arithmetic that makes or breaks this

Each replica serves `WORKER_SLOTS / SERVICE_TIME_MS` = **20 req/s**. So three
replicas serve 60 req/s and two serve 40 req/s.

Losing one of three replicas multiplies utilisation by exactly **1.5×**, and
queue wait grows like `ρ/(1-ρ)`. That makes the offered load the single most
important knob in the whole scenario:

| Offered load | Healthy ρ | Wounded ρ | Result |
| --- | --- | --- | --- |
| 36 req/s | 0.60 | 0.90 | Stable even while wounded. Spike is only ~3×: real, but underwhelming. |
| **42 req/s** | **0.70** | **1.05** | **The default.** Survivors are just over capacity, so the backlog grows while the replacement is missing, then drains. This is what produces a curve with a shape. |
| 48 req/s | 0.80 | 1.20 | Queue runs away, latency pins at the gateway timeout. Dramatic, but saturated - no shape left for the model to learn. |

---

## 4. Prerequisites

- Docker Desktop or OrbStack, **running**
- `kind` and `kubectl`: `brew install kind kubectl`
- Python 3.10+

The cluster is five containers plus two Python processes. If your machine is
tight on RAM, lower `--rps` rather than adding nodes.

---

## 5. Running it

```bash
cd RCA-SCenario4

python3 -m venv scene4
./scene4/bin/pip install -r requirements.txt

# Step 1+2: build the cluster, deploy everything, confirm a healthy baseline
bash scripts/cluster_up.sh

# Step 3+4: traffic for 5 minutes, node killed at T+60, keep hammering while it heals
./scene4/bin/python scripts/load_generator.py --duration 300 --rps 42 --chaos-at 60

# Step 5: pull the traces and flatten them into a labelled time series.
# Set RUN to the run id the load generator printed. Never type angle brackets -
# zsh treats < and > as redirection and will fail with "parse error near '\n'".
RUN=runs/run_20260728_141846_node
./scene4/bin/python scripts/extract_data.py  --run-dir "$RUN"
./scene4/bin/python scripts/build_dataset.py --run-dir "$RUN"

# Step 6: train the recovery-time forecaster
./scene4/bin/python scripts/forecast_recovery.py
```

Or let the shell pick the most recent run for you:

```bash
RUN=$(ls -dt runs/*/ | head -1)
./scene4/bin/python scripts/extract_data.py  --run-dir "$RUN"
./scene4/bin/python scripts/build_dataset.py --run-dir "$RUN"
```

`cluster_up.sh` prints the run command with the right flags, and each script
prints the next one. `RUNBOOK.md` has the command reference and troubleshooting.

The load generator prints a live one-line-per-second view, so you can watch the
brownout happen:

```text
   time     rps   ok  err   p50ms   p95ms   maxms  replicas serving
   20:30:15   42   42    0    113.9   172.4   201.3  3
   20:30:16   42   37    5    113.2   172.1  5009.7  3   <== FAULT INJECTED
   20:30:19   42   27   15    157.4  5015.2  5021.1  2   <== errors
   ...
   20:31:04   42   42    0    943.1  1091.4  1123.8  2   <== degraded
   20:31:31   42   42    0    163.2  1756.0  1801.2  3
```

---

## 6. Steps 5 and 6: from traces to a prediction

### The dataset is per-second, not per-trace

Scenarios 1-3 produced one row per trace labelled healthy/crash. A recovery time
cannot be labelled that way. Here every **second** of the incident is one row:

- **Features** — what an SRE would see on a dashboard: request rate, error rate,
  p50/p95/max latency, mean queue wait, how many distinct replicas actually
  served traffic, the 5-second latency slope, and time since the fault.
- **Label** — `seconds_to_restabilize`, i.e. how much longer the healing has
  left to run.

### Restabilization is defined, not eyeballed

A run counts as restabilized at the first second from which, for 8 of the next
10 seconds: p95 is back within 25 % of the pre-fault baseline, the error rate is
back to its baseline floor, and all three replicas are serving again.

Two deliberate details there. It compares against the **measured baseline**
rather than zero, and it requires 8 of 10 seconds rather than all 10. Both exist
because this environment has a small background error rate — kube-proxy
occasionally loses a SYN under a high rate of short-lived connections, and TCP
retransmission turns that request into a slow one. Roughly 1-4 % of seconds
contain one such request even when nothing is wrong. Demanding perfection would
make the finish line hostage to one unlucky packet.

### What the model learns, and an honest caveat

`forecast_recovery.py` trains RandomForest, GradientBoosting and a linear
baseline, then prints the live forecast an on-call engineer would see:

```text
[T+ 10s]  p95=5010ms  replicas_serving=2  errors=16
          Node failure detected. Orchestrator rerouting traffic.
          Estimated time to full system restabilization: 78 seconds.
```

**One run is not a training set.** With a single incident, the model discovers
that `seconds_to_restabilize = 88 - time_since_fault` and simply reads the
clock — feature importance for `time_since_fault` comes out at 0.998 and the
apparent error is ±0 s. That is memorization, not forecasting, and the script
says so out loud rather than quietly reporting a flattering number.

To guard against exactly this, every model is scored on two feature sets:

| Feature set | What it tests | MAE on the single run in this repo |
| --- | --- | --- |
| full | includes `time_since_fault` | 1.81 s (meaningless — it is reading a clock) |
| signal-only | latency and replica-count shape only, no clock | **15.78 s** |

The signal-only number is the one worth believing, and it is the one that
improves as you add runs. **Collect at least three runs before quoting any
accuracy figure:**

```bash
bash scripts/reset_cluster.sh    # restart the dead node, re-spread the replicas
./scene4/bin/python scripts/load_generator.py --duration 300 --rps 42 --chaos-at 60
# ... extract, build, repeat
```

Each run appends to `data/recovery_dataset.csv`. Once there are two or more,
`forecast_recovery.py` automatically switches to **leave-one-run-out**
validation: train on all runs but one, predict the recovery it has never seen.
That is the number that means something.

Varying the runs makes the model better and the evaluation harder: use
`--chaos-mode pod` for a much faster recovery with no detection phase, and vary
`--rps` to change the brownout depth.

---

## 7. Files

```text
RCA-SCenario4/
├── app/
│   ├── order_backend.py       # replicated service, bounded concurrency + warm-up
│   ├── api_gateway.py         # instrumented entry point, records which pod served
│   └── Dockerfile             # one image, both roles
├── k8s/
│   ├── kind-cluster.yaml      # 5 nodes: control-plane, infra, 3x app
│   ├── 00-namespace.yaml
│   ├── 10-jaeger.yaml         # Jaeger in-cluster, NodePort 30686
│   ├── 20-order-backend.yaml  # 3 replicas, anti-affinity, fast-failover tolerations
│   └── 30-api-gateway.yaml    # gateway + NodePort 30080
├── scripts/
│   ├── cluster_up.sh          # build cluster + image, deploy, smoke test
│   ├── cluster_down.sh        # delete the cluster
│   ├── reset_cluster.sh       # restore baseline between runs
│   ├── load_generator.py      # open-loop traffic + timed chaos injection
│   ├── inject_chaos.py        # kill a node (default) or a pod
│   ├── extract_data.py        # Jaeger API -> raw_trace_data.json
│   ├── build_dataset.py       # traces -> labelled per-second time series
│   └── forecast_recovery.py   # train, evaluate, chart, live forecast
├── runs/<run_id>/             # per-run: metrics, chaos event, traces, time series
├── data/recovery_dataset.csv  # all runs pooled - the training set
├── recovery_forecast.png      # generated chart
├── RUNBOOK.md                 # commands and troubleshooting
└── requirements.txt
```

---

## 8. Differences from Scenarios 1-3

| | Scenarios 1-3 | Scenario 4 |
| --- | --- | --- |
| Failure origin | Application (memory leak, queue, network) | Infrastructure (a node dies) |
| Environment | One script, or two Docker containers | A 5-node Kubernetes cluster |
| Jaeger | Standalone container on the host | Deployed inside the cluster |
| Host ports | 16686 / 4317 | 30686 / 30080 (no collision with the others) |
| ML task | Classification: healthy vs crash | Regression: seconds until restabilized |
| Row granularity | One row per trace | One row per second of the incident |
| Label source | Injected error flag | Derived from the recovery curve itself |

Because Jaeger runs inside the cluster on different host ports, this scenario
can run at the same time as the standalone Jaeger the earlier scenarios use.
