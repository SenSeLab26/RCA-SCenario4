# Runbook: commands and step-by-step process

Quick reference. The README explains *why*; this file is just *what to type*.

All commands run from the **`RCA-SCenario4`** folder (note the capital S — a
mistyped `RCA-Scenario4` does not exist).

Docker Desktop or OrbStack must be running before anything else. If `kubectl`
reports `connection refused`, that is almost always the cause.

---

## One-time setup

```bash
brew install kind kubectl                    # if you don't have them
python3 -m venv scene4
./scene4/bin/pip install -r requirements.txt
./scene4/bin/pip install langchain-ollama    # only for the LLM report step
```

---

## Step 1 — Start the cluster

```bash
bash scripts/cluster_up.sh
```

Builds a five-node cluster (control plane, one infra node for Jaeger and the
gateway, three app nodes each running one backend replica), deploys everything
and smoke-tests it. A few minutes the first time. Safe to re-run.

- Gateway: <http://localhost:30080/order>
- Jaeger UI: <http://localhost:30686> — pick service `api-gateway`

---

## Step 2 — Run traffic and inject a fault

Eight faults are available. `inject_fault.py` handles all of them using only
`kubectl` and `docker`; nothing extra is installed.

```bash
./scene4/bin/python scripts/inject_fault.py --list
```

| Fault | Scenario | What breaks |
| --- | --- | --- |
| `memory_leak` | 1 | backend leaks memory until the kernel OOMKills it |
| `cpu_saturation` | 1 | backend burns CPU until the container limit throttles it |
| `dependency_slow` | 2 | backend slows until the gateway times out |
| `dependency_down` | 2 | backend scaled to zero: connection refused |
| `config_error` | 2 | gateway pointed at a hostname that does not resolve |
| `network_partition` | 3 | replica alive and healthy but removed from the load balancer |
| `pod_failure` | 4 | one replica force-deleted |
| `node_failure` | 4 | the node hosting a replica is killed outright |

### The load level matters, and differs by fault

This is the easiest thing to get wrong. Each replica serves about 20 requests
per second, so three replicas serve 60. If the offered load is too close to that
ceiling, the queue collapses and every curve saturates at the gateway timeout
instead of showing its shape.

| Fault | Use | Why |
| --- | --- | --- |
| `node_failure`, `pod_failure` | `--rps 42` | 70% of capacity, so losing a replica pushes it just over and produces the brownout |
| `memory_leak`, `cpu_saturation` | `--rps 6` | the fault itself slows each request, so load must stay well under capacity or the queue hides the degradation |
| `dependency_slow` | `--rps 6` | same reason |
| `dependency_down`, `config_error`, `network_partition` | `--rps 42` | these fail outright rather than slowing down, so there is no queue to protect |

### 2a. Faults the load generator can fire for you

For `node_failure` and `pod_failure`, the load generator injects the fault itself
at the right moment, which keeps the timing perfectly aligned:

```bash
# Node failure: the full scenario, 5 minutes, node killed at T+60s
./scene4/bin/python scripts/load_generator.py --duration 300 --rps 42 --chaos-at 60

# Pod failure: much faster recovery, no ~45s detection phase
./scene4/bin/python scripts/load_generator.py --duration 240 --rps 42 --chaos-at 60 --chaos-mode pod
```

### 2b. Any of the eight faults

Start traffic in one terminal and inject in another. Give the run a name so the
two halves land in the same folder:

```bash
# Terminal 1 — traffic (note the low rps for a slow-degradation fault)
./scene4/bin/python scripts/load_generator.py --duration 160 --rps 6 --run-id run_memleak_01

# Terminal 2 — wait ~20s for a healthy baseline, then inject
./scene4/bin/python scripts/inject_fault.py --fault memory_leak \
    --leak-mb-per-sec 1.5 --run-dir runs/run_memleak_01
```

Every run needs a **healthy baseline in front of the fault** — inject 20 seconds
in at the earliest. Without it there is nothing to compare against and the
analysis scripts will refuse the run.

Useful flags: `--dry-run` shows what would be hit without doing it;
`--target rca4-worker3` picks the victim instead of choosing at random.

---

## Step 3 (optional) — Watch Kubernetes heal itself

Records the orchestrator's own timeline: when the node was marked dead, when the
replacement was scheduled, when it started, when it began taking traffic. Run it
in a third terminal, started just before the traffic:

```bash
./scene4/bin/python scripts/k8s_event_watcher.py --run-dir runs/run_memleak_01 --duration 180
```

It prints where the time actually went, and writes `k8s_timeline.json` and
`k8s_timeline.csv` into the run folder. This is how you show that Kubernetes
often finishes its repair long before users stop feeling the incident.

---

## Step 4 — Turn the run into data

```bash
RUN=$(ls -dt runs/*/ | head -1)     # newest run; set by hand to target another
./scene4/bin/python scripts/extract_data.py  --run-dir "$RUN"
./scene4/bin/python scripts/build_dataset.py --run-dir "$RUN"
```

`extract_data.py` pulls the traces out of Jaeger; `build_dataset.py` flattens
them to one row per second, works out when the system restabilized, and appends
to the single pooled dataset at `data/recovery_dataset.csv`.

There is also a simpler one-shot extractor that aggregates straight to a CSV,
used by the LLM report step:

```bash
./scene4/bin/python scripts/k8s_trace_extractor.py                    # from the live cluster
./scene4/bin/python scripts/k8s_trace_extractor.py --from-file "$RUN/raw_trace_data.json"
```

---

## Step 5 — The models

**Which fault is this?** The multi-class RCA classifier. Reads every run in
`runs/` and needs no cluster and no extraction step, so it works with Docker off:

```bash
./scene4/bin/python scripts/train_rca_classifier.py
```

Prints a per-incident verdict (the answer an RCA system actually gives) and
saves `rca_confusion_matrix.png`.

**How long until it recovers?** The recovery-time forecaster:

```bash
./scene4/bin/python scripts/forecast_recovery.py
```

Saves `recovery_forecast.png`. Needs `build_dataset.py` to have run.

**Write it up as an incident report.** Needs Ollama running locally
(`ollama serve`, `ollama pull llama3.2`):

```bash
./scene4/bin/python scripts/devops_agent.py --csv k8s_recovery_data.csv
./scene4/bin/python scripts/devops_agent.py --csv k8s_recovery_data.csv --summary-only   # skip the LLM
```

---

## Between runs — always reset

```bash
bash scripts/reset_cluster.sh
```

Restarts any node left powered off, re-spreads the replicas one per node, and
undoes `dependency_down`, `config_error` and `network_partition`. Skipping this
is the most common cause of a run that produces nonsense.

Each run writes its own `runs/<run_id>/` and appends to the single pooled
dataset, so runs accumulate rather than overwrite.

**Collect at least two runs per fault type.** With only one run of a fault, no
model can be tested on an incident it has not already seen, and both scripts
will say so rather than quoting a flattering number.

---

## Teardown

```bash
bash scripts/cluster_down.sh     # deletes the cluster, frees ports 30080 / 30686
```

`runs/` and `data/` survive teardown.

---

## Inspection commands

```bash
# Where are the replicas, and are they healthy?
kubectl -n rca4 get pods -l app=order-backend -o wide

# Proof the memory leak caused a real kernel kill
kubectl -n rca4 get pods -l app=order-backend -o custom-columns=\
'POD:.metadata.name,RESTARTS:.status.containerStatuses[0].restartCount,REASON:.status.containerStatuses[0].lastState.terminated.reason,EXIT:.status.containerStatuses[0].lastState.terminated.exitCode'

# What is the load balancer actually routing to?
kubectl -n rca4 get endpointslice -l kubernetes.io/service-name=order-backend

# Node health (a killed node shows Ready=Unknown)
kubectl get nodes -L rca4-role

# Watch the recovery live, in another terminal
kubectl -n rca4 get pods -l app=order-backend -w

# Logs
kubectl -n rca4 logs deploy/api-gateway --tail=20
kubectl -n rca4 logs -l app=order-backend --tail=20 --prefix

# Which node is down?
docker ps -a --filter name=rca4 --format '{{.Names}}\t{{.Status}}'
```

---

## Ports

This scenario does **not** use 16686 / 4317, so it does not collide with the
standalone Jaeger that Scenarios 1-3 use. Both can run at once.

| Port | What |
| --- | --- |
| 30080 | API gateway (`/order`) |
| 30686 | Jaeger UI and query API |

---

## Troubleshooting

- **`connection refused` on port 55559 (or any kubectl error)** — Docker is not
  running. Start OrbStack or Docker Desktop, wait for the nodes to come back
  (`docker ps --filter name=rca4`), then retry.
- **`cannot reach the API gateway`** — the cluster is not up. Run
  `bash scripts/cluster_up.sh`.
- **`OOMKilled` shows as `Unknown` / exit 255** — the pod has terminated again
  since, and Kubernetes only keeps the *most recent* termination. Any cluster or
  Docker restart erases the evidence. Re-run the leak to regenerate it.
- **A fault reports success but nothing happens** — check that
  `inject_fault.py` and `app/order_backend.py` agree on the parameter names. The
  in-app faults (`memory_leak`, `cpu_saturation`, `dependency_slow`) are set
  through the backend's `/control` endpoint, and an unrecognised key is silently
  ignored.
- **`no context exists with the name "kind-rca4"`** — a previous teardown
  half-failed and left orphaned containers. Clean up and recreate with
  `docker ps -a --filter name=rca4 -q | xargs -r docker rm -f` followed by
  `bash scripts/cluster_up.sh`.
- **Replicas not spread one per node** — check the labels with
  `kubectl get nodes -L rca4-role`; there should be three `app` nodes. Then run
  `reset_cluster.sh`. Anti-affinity is *preferred*, not required, so a
  replacement is allowed to double up on a surviving node rather than sitting
  `Pending` forever.
- **`build_dataset.py` says the system never restabilized** — the run ended
  before recovery finished. Use a longer `--duration`, or relax the criteria with
  `--tolerance 1.5 --hold 8`. For `memory_leak` this is expected: the container is
  OOMKilled repeatedly and never returns to baseline.
- **Every curve saturates at 5000 ms** — the offered load is too high for the
  fault. See the load table in Step 2.
- **No traces found by `extract_data.py`** — Jaeger stores traces in memory, so a
  Jaeger restart loses everything collected so far. Confirm it is up with
  `kubectl -n rca4 get pods`.
- **Everything is slow and erratic all the time** — five node containers plus two
  Python processes on a machine with under 4 GB for Docker. Lower `--rps`, or give
  OrbStack more memory.

---

## Running on Windows

Everything here works on Windows, with two adjustments.

**The three `.sh` scripts need a bash shell.** Use **Git Bash**, which comes with
Git for Windows, or a WSL 2 terminal. Docker Desktop already requires WSL 2, so
one of the two is certainly installed. Open Git Bash in the `RCA-SCenario4`
folder and run `bash scripts/cluster_up.sh` exactly as written.

**Paths and variables differ between shells.** In Git Bash use `RUN=runs/...`
and `"$RUN"`. In PowerShell use `$RUN = "runs\..."` and `$RUN`. Everything that
starts with `python scripts\...` works in PowerShell once the virtual
environment is active.

Common Windows-specific failures:

- `docker: invalid reference format` - the command was copied with backslash
  line continuations. Put it on one line, or run it in Git Bash.
- `No matching distribution found` - upgrade pip first with
  `python -m pip install --upgrade pip`.
- `running scripts is disabled on this system` - PowerShell blocks the venv
  activation script. Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- The cluster never becomes Ready - WSL 2 has too little memory. Create
  `%UserProfile%\.wslconfig` with `[wsl2]` and `memory=6GB`, then run
  `wsl --shutdown` and restart Docker Desktop.
- `kind: command not found` - install with `winget install Kubernetes.kind` and
  open a new terminal so PATH is refreshed.
