# Runbook: Scenario 4 commands

Quick reference. The README explains *why*; this file is just *what to type*.

All commands run from the `RCA-SCenario4` folder.

---

## One-time setup

```bash
brew install kind kubectl          # if you don't have them
python3 -m venv scene4
./scene4/bin/pip install -r requirements.txt
```

Docker Desktop or OrbStack must be running before any of the commands below.

---

## The happy path (four commands)

```bash
# 1. Build the six-node cluster, deploy Jaeger + gateway + 3 replicas
bash scripts/cluster_up.sh

# 2. Five minutes of traffic, with a node killed 60 seconds in
./scene4/bin/python scripts/load_generator.py --duration 300 --rps 42 --chaos-at 60

# 3. Pull the traces out of Jaeger and flatten them.
#    This picks the newest run automatically; set RUN by hand to target another.
#    Do not type angle brackets - zsh reads < and > as redirection.
RUN=$(ls -dt runs/*/ | head -1)
./scene4/bin/python scripts/extract_data.py  --run-dir "$RUN"
./scene4/bin/python scripts/build_dataset.py --run-dir "$RUN"

# 4. Train the recovery-time forecaster
./scene4/bin/python scripts/forecast_recovery.py
```

Jaeger UI: <http://localhost:30686> — pick service `api-gateway`.

---

## Between runs

A chaos run leaves one node powered off. Put the cluster back to a healthy
3-replicas-on-3-nodes baseline:

```bash
bash scripts/reset_cluster.sh
```

Then run the load generator again. Each run writes its own `runs/<run_id>/`
directory and appends to `data/recovery_dataset.csv`.

**Collect at least three runs before trusting the model's error numbers.** One
run is one recovery curve; the forecaster can fit it perfectly and still know
nothing about the next incident. `forecast_recovery.py` says so explicitly when
it only finds one run.

---

## Teardown

```bash
bash scripts/cluster_down.sh    # deletes the cluster, frees ports 30080 / 30686
```

Your `runs/` and `data/` directories survive teardown.

---

## Useful inspection commands

```bash
# Where are the replicas right now?
kubectl -n rca4 get pods -l app=order-backend -o wide

# What is the load balancer actually routing to?
kubectl -n rca4 get endpointslice -l kubernetes.io/service-name=order-backend -o yaml

# Node health (the killed node shows Ready=Unknown, then disappears)
kubectl get nodes

# Watch the recovery happen live, in another terminal
kubectl -n rca4 get pods -l app=order-backend -w

# Gateway / backend logs
kubectl -n rca4 logs deploy/api-gateway --tail=20
kubectl -n rca4 logs -l app=order-backend --tail=20 --prefix

# Which node did we kill, and is it still down?
docker ps -a --filter name=rca4 --format '{{.Names}}\t{{.Status}}'
```

---

## Variations worth running

```bash
# Kill just the pod instead of the whole node. Kubernetes notices immediately,
# so there is no ~40 second detection phase and recovery is much faster. Good
# contrast run, and it gives the model a second, different recovery shape.
./scene4/bin/python scripts/load_generator.py --duration 240 --rps 42 --chaos-at 60 --chaos-mode pod

# A clean baseline run with no fault at all, to see what "normal" looks like.
./scene4/bin/python scripts/load_generator.py --duration 120 --rps 42

# Pick the victim yourself instead of at random.
./scene4/bin/python scripts/inject_chaos.py --mode node --target rca4-worker3 --dry-run

# Heavier load: the survivors go well over capacity and latency pins at the
# gateway timeout instead of forming a curve. See the note in
# k8s/20-order-backend.yaml about why 42 is the default.
./scene4/bin/python scripts/load_generator.py --duration 300 --rps 48 --chaos-at 60
```

---

## Ports

Unlike Scenarios 1-3, this scenario does **not** use ports 16686 / 4317 on the
host, so it does not collide with the standalone Jaeger those scenarios use.
Jaeger runs inside the cluster and is published on **30686**; the gateway on
**30080**.

| Port | What |
| --- | --- |
| 30080 | API gateway (`/order`) |
| 30686 | Jaeger UI and query API |

---

## Troubleshooting

- **`cannot reach the API gateway`** — the cluster is not up, or Docker is not
  running. `bash scripts/cluster_up.sh`.
- **`no context exists with the name "kind-rca4"`** — a previous
  `cluster_down.sh` half-failed and left orphaned containers. Clean up and
  recreate:
  ```bash
  docker ps -a --filter name=rca4 -q | xargs -r docker rm -f
  bash scripts/cluster_up.sh
  ```
- **Replicas not spread one-per-node** — check the app nodes are labelled:
  `kubectl get nodes -L rca4-role`. There must be four `app` nodes for three
  replicas plus a spare.
- **The replacement replica sits `Pending`** — no spare app node is available.
  `kubectl -n rca4 describe pod <pod>` will say
  `didn't match pod anti-affinity rules`. Run `reset_cluster.sh`.
- **`build_dataset.py` says the system never restabilized** — the run ended
  before the cluster recovered. Use a longer `--duration`, or relax the criteria
  with `--tolerance 1.5 --hold 8`.
- **No traces found by `extract_data.py`** — confirm the run generated traffic,
  and that Jaeger is up: `kubectl -n rca4 get pods`. Jaeger stores traces in
  memory, so a Jaeger restart loses everything collected so far.
- **Everything is slow and erratic, all the time** — the whole cluster is six
  containers plus two Python processes. On a machine with little free RAM, give
  Docker/OrbStack more memory, or lower `--rps`.



my understanding was flawed so stick with the intitial approach. so don't duplicate them since we need the single dataset.