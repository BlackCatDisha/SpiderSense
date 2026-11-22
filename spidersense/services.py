"""
services.py — Service model and system state management.

Defines the six simulated services of the e-commerce system,
their attributes, and baseline/current metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import copy

# ---------------------------------------------------------------------------
# Service status constants
# ---------------------------------------------------------------------------
STATUS_HEALTHY   = "healthy"
STATUS_DEGRADED  = "degraded"
STATUS_FAILED    = "failed"

# ---------------------------------------------------------------------------
# Baseline (healthy-state) metrics — these never change
# ---------------------------------------------------------------------------
BASELINES: Dict[str, dict] = {
    "Frontend": {
        "latency_ms":    120.0,
        "error_rate":    0.5,    # percentage
        "anomaly":       0.0,
    },
    "API Gateway": {
        "latency_ms":    80.0,
        "error_rate":    0.3,
        "anomaly":       0.0,
    },
    "Orders": {
        "latency_ms":    200.0,
        "error_rate":    0.8,
        "anomaly":       0.0,
    },
    "Payment": {
        "latency_ms":    350.0,
        "error_rate":    0.5,
        "anomaly":       0.0,
    },
    "Inventory": {
        "latency_ms":    180.0,
        "error_rate":    0.6,
        "anomaly":       0.0,
    },
    "Database": {
        "latency_ms":    50.0,
        "error_rate":    0.2,
        "anomaly":       0.0,
    },
}

# Thresholds for the conventional baseline monitor
LATENCY_THRESHOLD  = 500.0   # ms — flag if latency exceeds this
ERROR_RATE_THRESHOLD = 5.0   # %  — flag if error rate exceeds this
ANOMALY_THRESHOLD  = 30.0    # signal units

# ---------------------------------------------------------------------------
# Service node
# ---------------------------------------------------------------------------
@dataclass
class Service:
    """Represents one simulated service in the e-commerce system."""

    name: str

    # Current operational metrics (mutable during fault injection)
    latency_ms:   float = 0.0
    error_rate:   float = 0.0   # percentage 0–100
    anomaly:      float = 0.0   # anomaly intensity 0–100
    status:       str   = STATUS_HEALTHY

    # Signal propagation fields
    signal_intensity:  float          = 0.0
    signal_arrival_t:  Optional[float] = None   # simulated time units
    signal_parent:     Optional[str]  = None    # name of originating neighbour
    threshold_exceeded: bool          = False
    is_fault_origin:   bool           = False   # was this the injected source?

    # -----------------------------------------------------------------------
    def reset_to_baseline(self) -> None:
        """Restore metrics to healthy baseline and clear propagation data."""
        base = BASELINES[self.name]
        self.latency_ms          = base["latency_ms"]
        self.error_rate          = base["error_rate"]
        self.anomaly             = base["anomaly"]
        self.status              = STATUS_HEALTHY
        self.signal_intensity    = 0.0
        self.signal_arrival_t    = None
        self.signal_parent       = None
        self.threshold_exceeded  = False
        self.is_fault_origin     = False

    # -----------------------------------------------------------------------
    def update_status(self) -> None:
        """Derive status from current metrics."""
        if self.anomaly >= 60 or self.error_rate >= 20 or self.latency_ms >= 2000:
            self.status = STATUS_FAILED
        elif self.anomaly >= 25 or self.error_rate >= 5 or self.latency_ms >= 500:
            self.status = STATUS_DEGRADED
        else:
            self.status = STATUS_HEALTHY


# ---------------------------------------------------------------------------
# System — container for all services
# ---------------------------------------------------------------------------
class ServiceSystem:
    """
    Holds the current simulation state for all six services.
    Thread-safety is not needed; Streamlit runs single-threaded per session.
    """

    # Canonical service ordering (determines display order)
    SERVICE_NAMES = [
        "Frontend",
        "API Gateway",
        "Orders",
        "Payment",
        "Inventory",
        "Database",
    ]

    def __init__(self) -> None:
        self._services: Dict[str, Service] = {}
        self._initialize()

    # -----------------------------------------------------------------------
    def _initialize(self) -> None:
        for name in self.SERVICE_NAMES:
            svc = Service(name=name)
            svc.reset_to_baseline()
            self._services[name] = svc

    # -----------------------------------------------------------------------
    def reset(self) -> None:
        """Return every service to its healthy baseline state."""
        for svc in self._services.values():
            svc.reset_to_baseline()

    # -----------------------------------------------------------------------
    def get(self, name: str) -> Service:
        return self._services[name]

    # -----------------------------------------------------------------------
    def all(self) -> list[Service]:
        return [self._services[n] for n in self.SERVICE_NAMES]

    # -----------------------------------------------------------------------
    def counts(self) -> dict:
        total    = len(self._services)
        healthy  = sum(1 for s in self._services.values() if s.status == STATUS_HEALTHY)
        degraded = sum(1 for s in self._services.values() if s.status == STATUS_DEGRADED)
        failed   = sum(1 for s in self._services.values() if s.status == STATUS_FAILED)
        return {
            "total": total,
            "healthy": healthy,
            "degraded": degraded,
            "failed": failed,
        }

    # -----------------------------------------------------------------------
    def affected_services(self) -> list[Service]:
        """Return services whose anomaly threshold was exceeded."""
        return [s for s in self._services.values() if s.threshold_exceeded]

    # -----------------------------------------------------------------------
    def snapshot(self) -> Dict[str, dict]:
        """Return a plain-dict snapshot of current service metrics."""
        snap = {}
        for name, svc in self._services.items():
            snap[name] = {
                "latency_ms":        svc.latency_ms,
                "error_rate":        svc.error_rate,
                "anomaly":           svc.anomaly,
                "status":            svc.status,
                "signal_intensity":  svc.signal_intensity,
                "signal_arrival_t":  svc.signal_arrival_t,
                "signal_parent":     svc.signal_parent,
                "threshold_exceeded": svc.threshold_exceeded,
                "is_fault_origin":   svc.is_fault_origin,
            }
        return snap

