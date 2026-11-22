"""
scenarios.py — Pre-defined fault scenarios and experimental test suite.

Each scenario specifies a service, fault type, and severity so that
clicking a scenario button instantly sets up a reproducible experiment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import time

from services import ServiceSystem
from fault_injector import (
    inject_fault,
    FAULT_COMPLETE_FAILURE, FAULT_LATENCY_SPIKE,
    FAULT_HIGH_ERROR_RATE, FAULT_SERVICE_TIMEOUT,
    SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW,
)
from propagation import build_graph, propagate
from root_cause import run_analysis


@dataclass
class Scenario:
    """Describes one pre-defined fault scenario."""
    name:         str
    description:  str
    service:      str
    fault_type:   str
    severity:     str
    # Expected root cause for evaluation (used only after prediction is made)
    expected_root_cause: str = ""


# ---------------------------------------------------------------------------
# Pre-defined scenarios
# ---------------------------------------------------------------------------
SCENARIOS: List[Scenario] = [
    Scenario(
        name="Database Outage",
        description=(
            "The Database service experiences a complete failure. "
            "This is the deepest infrastructure fault: failures propagate "
            "simultaneously to both Payment and Inventory, then converge "
            "on Orders, and cascade all the way to the Frontend."
        ),
        service="Database",
        fault_type=FAULT_COMPLETE_FAILURE,
        severity=SEVERITY_HIGH,
        expected_root_cause="Database",
    ),
    Scenario(
        name="Payment Latency Spike",
        description=(
            "The Payment service develops extreme latency. "
            "Orders service starts timing out waiting for payment confirmations, "
            "causing API Gateway degradation and partial frontend errors."
        ),
        service="Payment",
        fault_type=FAULT_LATENCY_SPIKE,
        severity=SEVERITY_HIGH,
        expected_root_cause="Payment",
    ),
    Scenario(
        name="Inventory Failure",
        description=(
            "The Inventory service crashes completely. "
            "Orders can no longer verify stock, so it fails; "
            "this then propagates to API Gateway and the Frontend."
        ),
        service="Inventory",
        fault_type=FAULT_COMPLETE_FAILURE,
        severity=SEVERITY_MEDIUM,
        expected_root_cause="Inventory",
    ),
    Scenario(
        name="API Gateway Timeout",
        description=(
            "The API Gateway experiences repeated timeouts. "
            "Because it sits between Orders and Frontend, the Frontend "
            "immediately degrades while upstream services remain healthy."
        ),
        service="API Gateway",
        fault_type=FAULT_SERVICE_TIMEOUT,
        severity=SEVERITY_HIGH,
        expected_root_cause="API Gateway",
    ),
    Scenario(
        name="Orders High Error Rate",
        description=(
            "The Orders service starts returning high error rates. "
            "API Gateway begins reporting failures, and the Frontend "
            "shows shopping-cart errors to end users."
        ),
        service="Orders",
        fault_type=FAULT_HIGH_ERROR_RATE,
        severity=SEVERITY_MEDIUM,
        expected_root_cause="Orders",
    ),
]

# Display-only scenarios (shown in sidebar)
DISPLAY_SCENARIOS = SCENARIOS[:3]  # first three for the sidebar quick-select


# ---------------------------------------------------------------------------
# Experimental evaluation runner
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    scenario_name:    str
    actual_fault:     str
    predicted_fault:  str
    correct:          bool
    affected_count:   int
    analysis_time_ms: float
    residual_score:   float


def run_all_experiments(graph=None) -> List[TestResult]:
    """
    Execute every scenario, run SpiderSense analysis, and collect results.
    Returns a list of TestResult objects.
    """
    if graph is None:
        graph = build_graph()

    results: List[TestResult] = []

    for scenario in SCENARIOS:
        # Fresh system for each test
        sys_under_test = ServiceSystem()

        # Inject fault
        inject_fault(
            sys_under_test,
            scenario.service,
            scenario.fault_type,
            scenario.severity,
        )

        # Propagate signal
        propagate(sys_under_test, graph, scenario.service)

        # Run analysis (independent — does not read is_fault_origin)
        t_start = time.perf_counter()
        analysis = run_analysis(sys_under_test, graph)
        t_end   = time.perf_counter()

        predicted   = analysis["predicted"]
        correct     = predicted == scenario.expected_root_cause
        affected    = len(sys_under_test.affected_services())
        elapsed_ms  = round((t_end - t_start) * 1000, 2)
        best_res    = analysis.get("best_residual", 0.0)

        results.append(TestResult(
            scenario_name=    scenario.name,
            actual_fault=     scenario.expected_root_cause,
            predicted_fault=  predicted,
            correct=          correct,
            affected_count=   affected,
            analysis_time_ms= elapsed_ms,
            residual_score=   best_res,
        ))

    return results


def compute_metrics(results: List[TestResult]) -> dict:
    """
    Compute precision, recall, accuracy, and F1 from test results.
    For single-label root-cause prediction:
      - True Positive  (TP): prediction == actual
      - False Positive (FP): prediction != actual (wrong label predicted)
      - False Negative (FN): actual service not predicted
    Accuracy = TP / total
    Precision = TP / (TP + FP)   (same as accuracy for single-label)
    Recall    = TP / (TP + FN)   (same as accuracy for single-label)
    F1        = 2 * P * R / (P + R)
    """
    total = len(results)
    if total == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    tp = sum(1 for r in results if r.correct)
    fp = total - tp
    fn = total - tp   # for single-label task FN == FP

    accuracy  = tp / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "accuracy":  round(accuracy,  3),
        "precision": round(precision, 3),
        "recall":    round(recall,    3),
        "f1":        round(f1,        3),
        "total":     total,
        "correct":   tp,
        "incorrect": fp,
    }

