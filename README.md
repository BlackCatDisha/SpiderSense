# SpiderSense Lite

**Bio-inspired Software Fault Localisation**  
*A college-level Bio-inspired Engineering project — RVCE, Semester 3*

---

## Overview

SpiderSense Lite demonstrates how the vibration-sensing mechanism of a spider's web can be computationally translated into a fault-localisation algorithm for interconnected software services.

A spider detects and localises prey by integrating vibration signals arriving at different legs from different directions. Each leg receives a signal whose intensity depends on the distance from the disturbance source and the mechanical properties of the silk threads. The spider infers the most probable source from the pattern of signal intensities and arrival times.

SpiderSense Lite applies this exact principle to a simulated e-commerce service mesh:

- A **disturbance signal** originates at a faulty service and propagates through the service-dependency graph.
- Signal intensity **attenuates** at each hop according to a **transmission coefficient** (analogous to silk-thread damping).
- Each service records the signal **intensity** and **arrival time**.
- A **residual-minimisation algorithm** tests every service as a hypothetical root cause, computes the expected propagation pattern, and selects the candidate whose predicted pattern best matches the observed anomalies.

---

## Project Structure

```
spidersense/
├── app.py            — Streamlit dashboard (main entry point)
├── services.py       — Service model and system state container
├── fault_injector.py — Fault injection logic
├── propagation.py    — Spider-web signal propagation engine
├── root_cause.py     — Root-cause localisation algorithm
├── scenarios.py      — Pre-defined scenarios and experiment runner
├── visualizer.py     — Plotly chart and graph builders
├── requirements.txt  — Python dependencies
└── README.md         — This file
```

---

## Setup and Installation

### Prerequisites
- Python 3.9 or later
- pip

### Install dependencies

```bash
cd spidersense
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Service Topology

The application simulates a six-service e-commerce system.  
Edges represent **failure-impact propagation** (not normal request flow):

```
Database ──► Payment ──► Orders ──► API Gateway ──► Frontend
         │                  ▲
         └──► Inventory ────┘
```

| Edge | Transmission Coefficient | Propagation Delay |
|---|---|---|
| Database → Payment | 0.85 | 1.0 t |
| Database → Inventory | 0.80 | 1.2 t |
| Payment → Orders | 0.75 | 1.5 t |
| Inventory → Orders | 0.70 | 1.3 t |
| Orders → API Gateway | 0.78 | 1.0 t |
| API Gateway → Frontend | 0.65 | 0.8 t |

---

## Signal Propagation Model

```
S_next = S_current × T_ij     (signal intensity at next service)
t_next = t_current + d_ij     (signal arrival time)
```

Where:
- `S_current` — signal intensity at the sender
- `T_ij`      — transmission coefficient of the directed edge (0–1)
- `d_ij`      — propagation delay (simulated time units)

---

## Root-Cause Localisation Algorithm

```
R(c) = Σ_i  w_i × [(O_i - P_i(c))² + λ × (TA_i - TP_i(c))²]
```

Where:
- `c`     — candidate originating service
- `O_i`   — observed anomaly intensity at service *i*
- `P_i(c)`— predicted signal intensity at *i* if *c* were the origin
- `TA_i`  — observed signal arrival time at service *i*
- `TP_i(c)`— predicted arrival time at *i* if *c* were the origin
- `w_i`   — importance weight of service *i*
- `λ`     — time-mismatch penalty coefficient (0.05)

The algorithm:
1. Collects observed anomaly metrics from every service.
2. For each candidate service, runs `predict_propagation_from()` — a pure mathematical simulation of propagation from that origin.
3. Computes the weighted residual score.
4. Ranks candidates by ascending residual.
5. Reports the lowest-residual candidate as the predicted root cause.

**The algorithm does not access the injected-fault label.**

---

## Pre-defined Scenarios

| Scenario | Injected Service | Fault Type | Severity |
|---|---|---|---|
| Database Outage | Database | Complete Service Failure | High |
| Payment Latency Spike | Payment | Latency Spike | High |
| Inventory Failure | Inventory | Complete Service Failure | Medium |
| API Gateway Timeout | API Gateway | Service Timeout | High |
| Orders High Error Rate | Orders | High Error Rate | Medium |

---

## Fault Types

| Fault Type | Description |
|---|---|
| Latency Spike | Dramatic increase in response time |
| High Error Rate | High proportion of failed requests |
| Service Timeout | Connections hang before failing |
| Complete Service Failure | Service is entirely unavailable |

---

## Design

The dashboard uses a **pastel engineering-analytics** visual style:
- Soft cream/off-white background with an engineering grid paper texture
- Sage green for healthy services, pale amber for degraded, dusty rose for failed
- Lavender/blue for signal propagation and analysis highlights
- Clean typography, thin borders, and generous spacing

---

## Limitations

This is a **simulation** for educational purposes:
- Services are logical objects inside one Python process, not real microservices.
- Metrics are deterministic computations, not measurements from real systems.
- The propagation model is a simplified analogy; it does not simulate the physics of spider silk.
- One originating fault at a time is supported.

---

## Biological Analogy Summary

| Spider Web | SpiderSense |
|---|---|
| Spider silk threads | Service dependency edges |
| Disturbance at a point | Fault injected at a service |
| Vibration propagation | Signal propagation along edges |
| Silk damping coefficient | Transmission coefficient T_ij |
| Vibration arrival time at leg | Signal arrival time at service |
| Spider localises prey | Algorithm localises root cause |
