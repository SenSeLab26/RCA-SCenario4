# Scenario Consolidation Plan

Mapping the nine fault types from the supervisor onto four scenarios, and the
way forward for fault injection tooling.

---

## 1. Where the nine faults land

| Supervisor's fault | Covered today? | Proposed home |
| --- | --- | --- |
| Memory leaks | Scenario 1 (simulated) | **Scenario 1** |
| CPU saturation | Not covered | **Scenario 1** |
| Database failures | Partly, Scenario 2 (slow, never down) | **Scenario 2** |
| Cascading failures | Scenario 2 | **Scenario 2** |
| Configuration errors | Not covered | **Scenario 2** |
| Packet loss | Scenario 3 | **Scenario 3** |
| Network partition | Not covered (loss is not partition) | **Scenario 3** |
| Node failure | Scenario 4 (real, measured) | **Scenario 4** |
| Pod failures | Scenario 4 (implemented, now tested) | **Scenario 4** |

Nine faults, four scenarios, no orphans. The pairing is not arbitrary: each
scenario groups faults that **share a failure signature but differ in root
cause**, which is exactly the discrimination task an RCA model has to solve.

---

## 2. The four consolidated scenarios

### Scenario 1: Resource Exhaustion
**Merges:** memory leak + CPU saturation

Both are "the process is alive but starving", and they separate cleanly in
telemetry:

- **Memory leak** ends in a hard stop. The container hits its memory limit and is
  OOMKilled, so the pod restarts and the error is abrupt.
- **CPU saturation** never stops. Latency climbs and stays high, with no restart
  and often no errors at all.

The model's job is to tell a degradation that will *end in a crash* from one that
will simply *stay slow*. That is a genuinely useful distinction for an on-call
engineer, and it does not exist in the current Scenario 1.

**Upgrade over today:** the present scenario fakes the leak with `time.sleep()`
against a 0.8 s threshold. Replacing it with a real container memory limit and a
real OOMKill makes the failure genuine rather than asserted.

### Scenario 2: Dependency Failure and Cascading Timeout
**Merges:** database failures + cascading failures + configuration errors

This is the strongest RCA scenario in the set, because all three faults produce
the **same symptom** at the front door - an HTTP 504 from Level 1 - while the
cause differs completely:

| Root cause | Backend signature |
| --- | --- |
| Database overloaded | Backend responds, slowly. No errors at the backend. |
| Database down | Backend fails fast with connection refused. |
| Misconfigured endpoint | Backend fails instantly, and fails from the first request, with no healthy period at all. |

A model that only reads the frontend cannot separate these. One that reads the
whole trace can. That is a much sharper research claim than the current
"frontend is red, backend is green" result, and it absorbs configuration errors
without inventing a fifth scenario.

### Scenario 3: Network Degradation and Partition
**Merges:** packet loss + network partition

- **Packet loss** is graded. Latency and error rate rise smoothly with the loss
  percentage.
- **Partition** is binary and asymmetric. Both sides may believe they are
  healthy while being unable to reach each other, which is why it needs its own
  treatment rather than being folded into "loss".

**Upgrade over today:** the current scenario floods the network with `iperf3`,
which is real but uncontrolled - you cannot ask for "8% loss" and reproduce it.
A fault injector sets loss, latency and partition as exact, repeatable
parameters.

### Scenario 4: Orchestrator Fault Tolerance
**Covers:** node failure + pod failure

Already built and measured. Keep as the mature reference implementation.
Both modes now work:

| Mode | Detection | Kubernetes healing | User-visible recovery |
| --- | --- | --- | --- |
| Node kill | ~45 s | replacement Ready ~T+75 s | 88 s and 134 s (two runs) |
| Pod kill | immediate | replacement Ready ~T+18 s | ~27 s |

The pod-kill measurement produced the most quotable finding in the project so
far: **Kubernetes' own repair took about one second; the other 17 seconds were
our application's readiness warm-up.** The orchestrator is not the bottleneck.

---

## 3. Fault injection tooling

### Recommendation: Chaos Mesh, for Scenarios 1-3 only

Chaos Mesh maps almost one-to-one onto the supervisor's list:

| Fault needed | Chaos Mesh resource |
| --- | --- |
| Memory leak, CPU saturation | `StressChaos` (memory and CPU stressors) |
| Pod failure | `PodChaos` (pod-kill, pod-failure, container-kill) |
| Packet loss, latency, partition | `NetworkChaos` (loss, delay, partition) |
| Database failure | `PodChaos` against the database pod |
| Configuration error | plain `kubectl patch` on the Deployment - no tool needed |

Chosen over LitmusChaos because it is lighter. LitmusChaos' ChaosCenter needs
its own database and control plane, which this machine cannot spare (see the
resource warning below). Chaos Mesh experiments are plain Kubernetes YAML, which
fits how Scenario 4 is already driven.

### Do not use it for the node kill

Chaos Mesh runs *inside* the cluster, so it cannot destroy a kind node, because
that node is a Docker container on the host, outside the cluster's reach. The
existing `docker kill` in `inject_chaos.py` is the correct mechanism and should
stay. Adopting a tool is not a reason to replace the part that already works.

### Installation note

On kind, Chaos Mesh needs to be told which container runtime it is talking to,
or NetworkChaos silently fails:

```bash
helm install chaos-mesh chaos-mesh/chaos-mesh \
  --namespace=chaos-mesh --create-namespace \
  --set chaosDaemon.runtime=containerd \
  --set chaosDaemon.socketPath=/run/containerd/containerd.sock
```

---

## 4. The real prize: one dataset instead of four

Today each scenario produces its own CSV with its own columns, and each trains
its own model. Nothing transfers between them.

If all four scenarios run on the **Scenario 4 cluster**, through the **same
gateway and the same OpenTelemetry pipeline**, then every scenario emits the
same per-second feature row, and the only thing that changes is a `fault_type`
label. That single change upgrades the research contribution:

| Today | After consolidation |
| --- | --- |
| Four disconnected datasets | One dataset, one schema |
| Four models, none comparable | One multi-class RCA classifier: given the telemetry, which of the nine faults is this? |
| Recovery prediction only in Scenario 4 | Recovery prediction across every fault type |

That is what makes it a Smart Telemetry **System** rather than four experiments,
and it is the strongest argument for consolidating rather than adding scenarios
five through nine.

---

## 5. Risks and decisions needed

### Resource ceiling - check this first

The Docker VM has **8 CPUs but only 3.9 GiB of RAM**, and the control-plane node
alone is already using 778 MiB. A six-node cluster previously starved the
control plane badly enough that `kube-scheduler` crash-looped and lost its leader
lease, which is why the cluster was reduced to five nodes.

Chaos Mesh adds a controller manager (3 replicas by default), a chaos-daemon on
**every** node, and a dashboard. That is a meaningful addition to a cluster
already near its limit.

**Mitigation, in order of preference:**
1. Raise the OrbStack VM memory allocation to 8 GiB (a settings change, no code).
2. Install Chaos Mesh with `--set controllerManager.replicaCount=1` and the
   dashboard disabled.
3. Drop the cluster to four nodes.

This should be settled before any migration work, because everything else
depends on the cluster staying stable.

### Do not delete the existing scripts

`scenario_1.py`, `scenario_2.py` and `network_test.py` are already written up in
the LaTeX report and produce the results in it. The migrated versions should be
built alongside them, not on top of them. If the migration stalls, the report
still stands.

### Scope check

Migrating Scenarios 1-3 onto the cluster is roughly the work that Scenario 4
took, three times over. A realistic order is Scenario 2 first (highest research
value, and the gateway and backend already exist to build on), then 3, then 1.

---

## 6. Proposed sequence

| Step | Work | Outcome |
| --- | --- | --- |
| 0 | Resolve the memory ceiling; install Chaos Mesh; verify one trivial `PodChaos` runs | Tooling proven on this machine |
| 1 | Add `fault_type` to the dataset schema; confirm Scenario 4 still builds | One schema, no regression |
| 2 | Scenario 2 on the cluster: DB slow, DB down, misconfigured | Three causes, one symptom - the RCA result |
| 3 | Scenario 3 with `NetworkChaos`: graded loss plus partition | Reproducible network faults |
| 4 | Scenario 1 with `StressChaos`: real OOMKill plus CPU saturation | Real resource exhaustion |
| 5 | Train the multi-class RCA classifier across all runs | The Smart Telemetry System result |

Step 0 is the gate. Nothing after it is worth starting until Chaos Mesh is
confirmed to run stably on this cluster.
