"""
root_cause.py — Independent root-cause localisation engine.

Implements candidate-based residual minimisation inspired by the
principle that a spider can localise a disturbance from the pattern
of vibration intensities across its web.

Algorithm
---------
For each candidate service c:

  R(c) = Σ_i  w_i × [(O_i - P_i(c))²  +  λ × (TA_i - TP_i(c))²]

where:
  O_i    — observed anomaly intensity at service i
  P_i(c) — predicted intensity if c were the origin
  TA_i   — observed signal arrival time at service i (None → penalised)
  TP_i(c)— predicted arrival time if c were origin
  w_i    — importance weight of service i
  λ      — time-mismatch weight (small)

The candidate with the lowest R is the predicted root cause.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math
import networkx as nx

from services import ServiceSystem, Service
from propagation import predict_propagation_from

# ---------------------------------------------------------------------------
# Service importance weights — services closer to the user have more
# observability, but their anomaly is usually secondary.
# Deeper infrastructure services are weighted slightly higher because
# a primary fault there impacts many downstream nodes simultaneously.
# ---------------------------------------------------------------------------
IMPORTANCE_WEIGHTS: Dict[str, float] = {
    "Frontend":    0.6,
    "API Gateway": 0.7,
    "Orders":      0.8,
    "Payment":     0.9,
    "Inventory":   0.9,
    "Database":    1.0,
}

# Time-mismatch penalty coefficient
_LAMBDA = 0.05

# Confidence gap: if the two lowest residuals are within this fraction
# of the best score, report uncertainty.
_UNCERTAINTY_GAP = 0.15


def run_analysis(
    system: ServiceSystem,
    graph: nx.DiGraph,
) -> dict:
    """
    Independently infer the most likely root-cause service.

    Does NOT look at the is_fault_origin flag on any service.

    Returns a results dict:
    {
        "predicted":      str,
        "actual":         str | None,   # filled by caller if desired
        "correct":        bool | None,
        "uncertain":      bool,
        "confidence":     float,        # 0–1
        "candidates":     [
            {"service": str, "residual": float, "rank": int}, ...
        ],
        "explanation":    str,
        "propagation_path": [str, ...], # predicted path from winner
    }
    """
    # ---- 1. Collect observations ----------------------------------------
    # We observe anomaly intensity and signal arrival time per service.
    observations: Dict[str, dict] = {}
    for svc in system.all():
        observations[svc.name] = {
            "anomaly":    svc.anomaly,
            "arrival_t":  svc.signal_arrival_t,
            "threshold":  svc.threshold_exceeded,
        }

    # Normalise observed anomaly to [0, 100] scale
    max_obs = max((v["anomaly"] for v in observations.values()), default=1.0)
    if max_obs == 0:
        max_obs = 1.0

    # Determine a representative "initial signal" to test from each candidate.
    # Use the max observed anomaly as a proxy.
    init_sig = max(100.0, max_obs)

    # ---- 2. Evaluate each candidate --------------------------------------
    scores: List[Tuple[float, str]] = []

    for candidate in system.SERVICE_NAMES:
        predicted = predict_propagation_from(graph, candidate, initial_signal=init_sig)

        residual = 0.0
        for svc_name, obs in observations.items():
            w      = IMPORTANCE_WEIGHTS.get(svc_name, 0.5)
            O_i    = obs["anomaly"]
            P_i    = predicted.get(svc_name, {}).get("signal", 0.0)
            TA_i   = obs["arrival_t"]
            TP_i   = predicted.get(svc_name, {}).get("arrival_t", float("inf"))

            # Anomaly residual (primary term)
            intensity_res = (O_i - P_i) ** 2

            # Arrival-time residual (secondary term) — only for affected services
            if TA_i is not None and TP_i != float("inf"):
                time_res = (TA_i - TP_i) ** 2
            elif TA_i is None and O_i < 5.0:
                time_res = 0.0   # unaffected and correctly predicted silent
            else:
                time_res = 1000.0  # penalty for missing/mismatched timing

            residual += w * (intensity_res + _LAMBDA * time_res)

        scores.append((residual, candidate))

    scores.sort(key=lambda x: x[0])

    # ---- 3. Build ranked candidate list ----------------------------------
    best_residual = scores[0][0]
    second_residual = scores[1][0] if len(scores) > 1 else best_residual * 2

    candidates_ranked = [
        {
            "service":  name,
            "residual": round(res, 2),
            "rank":     rank + 1,
        }
        for rank, (res, name) in enumerate(scores)
    ]

    predicted_service = scores[0][1]

    # ---- 4. Confidence and uncertainty -----------------------------------
    # Confidence: how much better the best candidate is vs. the second best.
    if second_residual > 0:
        gap_ratio = (second_residual - best_residual) / second_residual
    else:
        gap_ratio = 0.0

    uncertain  = gap_ratio < _UNCERTAINTY_GAP
    # Confidence mapped to [0, 1]: gap_ratio of 0 → 0.5, 1 → 1.0
    confidence = 0.5 + 0.5 * min(gap_ratio / 0.5, 1.0)
    if uncertain:
        confidence = min(confidence, 0.70)

    # ---- 5. Predicted propagation path -----------------------------------
    # Reconstruct a likely propagation path from the predicted root cause
    # through the graph, following edges whose predicted signal exceeds threshold.
    predicted_path = _build_propagation_path(
        graph, predicted_service, init_sig
    )

    # ---- 6. Explanation --------------------------------------------------
    explanation = _generate_explanation(
        predicted_service, candidates_ranked, observations,
        uncertain, confidence, predicted_path
    )

    return {
        "predicted":         predicted_service,
        "actual":            None,
        "correct":           None,
        "uncertain":         uncertain,
        "confidence":        round(confidence, 3),
        "candidates":        candidates_ranked,
        "explanation":       explanation,
        "propagation_path":  predicted_path,
        "best_residual":     round(best_residual, 2),
    }


def _build_propagation_path(
    graph: nx.DiGraph,
    origin: str,
    init_sig: float,
    threshold: float = 15.0,
) -> List[str]:
    """Return ordered list of services that would receive signal above threshold."""
    from propagation import TRANSMISSION, DELAYS
    import heapq

    path_order: List[Tuple[float, str]] = [(0.0, origin)]
    heap = [(0.0, origin, init_sig)]
    visited = {origin}

    while heap:
        t, name, sig = heapq.heappop(heap)
        for _, nbr in graph.out_edges(name):
            t_c = TRANSMISSION.get((name, nbr), 0.5)
            d   = DELAYS.get((name, nbr), 1.0)
            nsig = sig * t_c
            nt   = t + d
            if nsig >= threshold and nbr not in visited:
                visited.add(nbr)
                path_order.append((nt, nbr))
                heapq.heappush(heap, (nt, nbr, nsig))

    path_order.sort(key=lambda x: x[0])
    return [name for _, name in path_order]


def _generate_explanation(
    predicted: str,
    candidates: list,
    observations: dict,
    uncertain: bool,
    confidence: float,
    path: list,
) -> str:
    """Generate a human-readable explanation from computed values."""
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None

    pct = round(confidence * 100, 1)
    path_str = " → ".join(path) if path else predicted

    obs_summary = []
    for name, obs in observations.items():
        if obs["anomaly"] > 5:
            obs_summary.append(f"{name} (anomaly={obs['anomaly']:.1f})")

    obs_text = ", ".join(obs_summary) if obs_summary else "no affected services"

    lines = [
        f"**Predicted root cause: {predicted}** (confidence {pct}%)",
        "",
        f"The residual-minimisation algorithm evaluated all six services as "
        f"potential fault origins. {predicted} produced the lowest weighted "
        f"residual score ({best['residual']:.1f}), meaning its predicted "
        f"propagation pattern best matched the observed anomaly distribution.",
    ]

    if second:
        lines.append(
            f"The next-ranked candidate, {second['service']}, scored "
            f"{second['residual']:.1f}."
        )

    if obs_text:
        lines.append(f"Observed anomalies: {obs_text}.")

    if path:
        lines.append(f"Predicted failure cascade: {path_str}.")

    if uncertain:
        lines.append(
            "\u26a0\ufe0f **Uncertainty warning:** Multiple candidates have similar "
            "residual scores. The prediction may not be reliable. Consider "
            "examining the candidate ranking table for alternative hypotheses."
        )

    return "\n".join(lines)

