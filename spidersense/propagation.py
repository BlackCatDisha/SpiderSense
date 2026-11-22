"""
propagation.py — Spider-web signal propagation engine.

Implements the bio-inspired disturbance signal that travels through
the directed service-dependency graph, analogous to vibrations
propagating across a spider's web.

Propagation model
-----------------
  S_next = S_current × T_ij          (signal attenuation)
  t_next = t_current + d_ij          (propagation delay)

where:
  S_current  — signal intensity at the sender
  T_ij       — transmission coefficient of the directed edge (0, 1]
  d_ij       — propagation delay (simulated time units)
  t_current  — signal arrival time at the sender
"""

from __future__ import annotations
import networkx as nx
from typing import Dict, List, Optional, Tuple
from services import ServiceSystem, Service, ANOMALY_THRESHOLD

# ---------------------------------------------------------------------------
# Directed failure-impact propagation graph definition
# Edges: (source, target) — failure at source impacts target
# ---------------------------------------------------------------------------

EDGES: List[Tuple[str, str]] = [
    ("Database",    "Payment"),
    ("Database",    "Inventory"),
    ("Payment",     "Orders"),
    ("Inventory",   "Orders"),
    ("Orders",      "API Gateway"),
    ("API Gateway", "Frontend"),
]

# ---------------------------------------------------------------------------
# Edge transmission coefficients — how well a failure signal crosses the edge
# A higher value means downstream services are more heavily impacted.
# ---------------------------------------------------------------------------
TRANSMISSION: Dict[Tuple[str, str], float] = {
    ("Database",    "Payment"):     0.85,
    ("Database",    "Inventory"):   0.80,
    ("Payment",     "Orders"):      0.75,
    ("Inventory",   "Orders"):      0.70,
    ("Orders",      "API Gateway"): 0.78,
    ("API Gateway", "Frontend"):    0.65,
}

# ---------------------------------------------------------------------------
# Edge propagation delays (simulated time units, not wall-clock seconds)
# ---------------------------------------------------------------------------
DELAYS: Dict[Tuple[str, str], float] = {
    ("Database",    "Payment"):     1.0,
    ("Database",    "Inventory"):   1.2,
    ("Payment",     "Orders"):      1.5,
    ("Inventory",   "Orders"):      1.3,
    ("Orders",      "API Gateway"): 1.0,
    ("API Gateway", "Frontend"):    0.8,
}


def build_graph() -> nx.DiGraph:
    """Build and return the NetworkX directed graph for the service topology."""
    G = nx.DiGraph()
    G.add_nodes_from([
        "Frontend", "API Gateway", "Orders",
        "Payment", "Inventory", "Database",
    ])
    for src, dst in EDGES:
        G.add_edge(
            src, dst,
            transmission=TRANSMISSION[(src, dst)],
            delay=DELAYS[(src, dst)],
        )
    return G


def propagate(
    system: ServiceSystem,
    graph: nx.DiGraph,
    origin_service: str,
    *,
    min_signal: float = 1.0,
) -> List[dict]:
    """
    Run the signal propagation simulation from *origin_service*.

    The origin service must already have signal_intensity and signal_arrival_t
    set (done by fault_injector.inject_fault).

    Returns a list of propagation events in chronological order:
        [{"service": str, "signal": float, "arrival_t": float,
          "parent": str, "edge": (src, dst), "transmission": float}, ...]
    """
    events: List[dict] = []

    # BFS / priority-queue traversal so we always process the node with
    # the earliest current arrival time first (Dijkstra-like).
    import heapq

    # heap entries: (arrival_time, service_name, signal_intensity, parent_name)
    origin_svc = system.get(origin_service)
    heap = [(
        origin_svc.signal_arrival_t,
        origin_service,
        origin_svc.signal_intensity,
        None,
    )]
    # Track best signal received per service (we keep the strongest)
    best_signal: Dict[str, float] = {origin_service: origin_svc.signal_intensity}

    while heap:
        arrival_t, current_name, signal, parent_name = heapq.heappop(heap)

        # Skip if we already found a stronger signal path to this node
        if best_signal.get(current_name, -1) > signal and current_name != origin_service:
            continue

        current_svc = system.get(current_name)

        # Update service state if this signal is stronger than what we have
        if signal >= current_svc.signal_intensity:
            current_svc.signal_intensity  = signal
            current_svc.signal_arrival_t  = arrival_t
            if parent_name:
                current_svc.signal_parent = parent_name

        # Mark threshold exceeded
        if signal >= ANOMALY_THRESHOLD:
            current_svc.threshold_exceeded = True

        # Update anomaly intensity on affected (non-origin) services
        if current_name != origin_service:
            # Anomaly grows proportionally to incoming signal
            added_anomaly = signal * 0.6
            current_svc.anomaly    = min(100.0, current_svc.anomaly + added_anomaly)
            # Secondary latency / error impact
            current_svc.latency_ms += signal * 3.5
            current_svc.error_rate  = min(99.0, current_svc.error_rate + signal * 0.15)
            current_svc.update_status()

            events.append({
                "service":      current_name,
                "signal":       signal,
                "arrival_t":    arrival_t,
                "parent":       parent_name,
                "edge":         (parent_name, current_name),
                "transmission": TRANSMISSION.get((parent_name, current_name), 0.0),
            })

        # Propagate to successors
        if signal < min_signal:
            continue

        for _, neighbor in graph.out_edges(current_name):
            t_coeff = TRANSMISSION.get((current_name, neighbor), 0.5)
            delay   = DELAYS.get((current_name, neighbor), 1.0)
            next_signal  = signal * t_coeff
            next_arrival = arrival_t + delay

            if next_signal < min_signal:
                continue

            # Only push if this is a stronger signal than previously seen
            if next_signal > best_signal.get(neighbor, -1):
                best_signal[neighbor] = next_signal
                heapq.heappush(heap, (next_arrival, neighbor, next_signal, current_name))

    return sorted(events, key=lambda e: e["arrival_t"])


def predict_propagation_from(
    graph: nx.DiGraph,
    candidate_origin: str,
    initial_signal: float = 100.0,
) -> Dict[str, dict]:
    """
    Pure-math prediction: given a hypothetical origin, compute the expected
    signal intensity and arrival time at every reachable node.

    Returns:
        {service_name: {"signal": float, "arrival_t": float}}
    """
    import heapq

    result: Dict[str, dict] = {
        candidate_origin: {"signal": initial_signal, "arrival_t": 0.0}
    }
    heap = [(0.0, candidate_origin, initial_signal)]

    while heap:
        t, name, sig = heapq.heappop(heap)
        if result.get(name, {}).get("signal", -1) > sig:
            continue

        for _, neighbor in graph.out_edges(name):
            t_coeff = TRANSMISSION.get((name, neighbor), 0.5)
            delay   = DELAYS.get((name, neighbor), 1.0)
            nsig    = sig * t_coeff
            nt      = t + delay

            if nsig > result.get(neighbor, {}).get("signal", -1):
                result[neighbor] = {"signal": nsig, "arrival_t": nt}
                heapq.heappush(heap, (nt, neighbor, nsig))

    # Services unreachable from the candidate get zero signal
    all_nodes = list(graph.nodes)
    for node in all_nodes:
        if node not in result:
            result[node] = {"signal": 0.0, "arrival_t": float("inf")}

    return result

