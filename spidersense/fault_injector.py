"""
fault_injector.py — Fault injection logic.

Applies faults to a selected service and updates its simulated metrics.
The effects on dependent services are handled by the propagation engine.
"""

from __future__ import annotations
from typing import Dict
from services import Service, ServiceSystem, STATUS_DEGRADED, STATUS_FAILED

# ---------------------------------------------------------------------------
# Fault type constants
# ---------------------------------------------------------------------------
FAULT_LATENCY_SPIKE        = "Latency Spike"
FAULT_HIGH_ERROR_RATE      = "High Error Rate"
FAULT_SERVICE_TIMEOUT      = "Service Timeout"
FAULT_COMPLETE_FAILURE     = "Complete Service Failure"

FAULT_TYPES = [
    FAULT_LATENCY_SPIKE,
    FAULT_HIGH_ERROR_RATE,
    FAULT_SERVICE_TIMEOUT,
    FAULT_COMPLETE_FAILURE,
]

# Severity level descriptions
SEVERITY_LOW    = "Low"
SEVERITY_MEDIUM = "Medium"
SEVERITY_HIGH   = "High"

SEVERITY_LEVELS = [SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH]

# ---------------------------------------------------------------------------
# Severity multipliers (applied to the base fault parameters)
# ---------------------------------------------------------------------------
_SEVERITY_MULTIPLIER: Dict[str, float] = {
    SEVERITY_LOW:    0.5,
    SEVERITY_MEDIUM: 1.0,
    SEVERITY_HIGH:   1.8,
}

# ---------------------------------------------------------------------------
# Base fault parameters per fault type
# (latency_add_ms, error_rate_add, anomaly_add, initial_signal)
# ---------------------------------------------------------------------------
_BASE_PARAMS: Dict[str, tuple] = {
    # (Δlatency ms, Δerror %, Δanomaly intensity, initial signal strength)
    FAULT_LATENCY_SPIKE:    (1200.0, 2.0,  45.0, 80.0),
    FAULT_HIGH_ERROR_RATE:  (200.0,  25.0, 55.0, 85.0),
    FAULT_SERVICE_TIMEOUT:  (3000.0, 15.0, 70.0, 90.0),
    FAULT_COMPLETE_FAILURE: (5000.0, 95.0, 100.0, 100.0),
}


def inject_fault(
    system: ServiceSystem,
    service_name: str,
    fault_type: str,
    severity: str = SEVERITY_MEDIUM,
) -> dict:
    """
    Apply a fault to *service_name* inside *system*.

    Returns a summary dict describing what was changed.
    """
    svc: Service = system.get(service_name)
    mult = _SEVERITY_MULTIPLIER[severity]
    lat_add, err_add, anom_add, init_signal = _BASE_PARAMS[fault_type]

    # Apply the fault
    svc.latency_ms   += lat_add  * mult
    svc.error_rate   += err_add  * mult
    svc.anomaly       = min(100.0, svc.anomaly + anom_add * mult)

    # Mark this service as the fault origin
    svc.is_fault_origin      = True
    svc.signal_intensity     = init_signal * mult
    svc.signal_intensity     = min(100.0, svc.signal_intensity)
    svc.signal_arrival_t     = 0.0        # time 0 — the origin
    svc.threshold_exceeded   = True

    # Update status from new metrics
    svc.update_status()

    return {
        "service":        service_name,
        "fault_type":     fault_type,
        "severity":       severity,
        "delta_latency":  lat_add * mult,
        "delta_error":    err_add * mult,
        "initial_signal": svc.signal_intensity,
        "status":         svc.status,
    }

