import sys
sys.path.insert(0, '.')
from services import ServiceSystem
from fault_injector import inject_fault, FAULT_COMPLETE_FAILURE, SEVERITY_HIGH
from propagation import build_graph, propagate
from root_cause import run_analysis
from scenarios import run_all_experiments, compute_metrics

# Test 1: Database Outage
print('=== Test 1: Database Outage ===')
system = ServiceSystem()
G = build_graph()
inject_fault(system, 'Database', FAULT_COMPLETE_FAILURE, SEVERITY_HIGH)
events = propagate(system, G, 'Database')
print(f'Propagation events: {len(events)}')
for e in events:
    at = e["arrival_t"]
    sn = e["signal"]
    sv = e["service"]
    print(f'  t={at:.2f} -> {sv} signal={sn:.1f}')

result = run_analysis(system, G)
pred = result["predicted"]
conf = result["confidence"]
top3 = [(c["service"], c["residual"]) for c in result["candidates"][:3]]
print(f'Predicted root cause: {pred}')
print(f'Confidence: {conf*100:.1f}%')
print(f'Top candidates: {top3}')
print()

# Test 2: Run all experiments
print('=== Experiment Suite ===')
all_results = run_all_experiments(G)
metrics = compute_metrics(all_results)
for r in all_results:
    verdict = 'CORRECT' if r.correct else 'WRONG'
    print(f'  [{verdict}] {r.scenario_name}: actual={r.actual_fault}, predicted={r.predicted_fault}')
acc = metrics["accuracy"]
prec = metrics["precision"]
f1 = metrics["f1"]
print(f'Accuracy={acc:.1%} Precision={prec:.1%} F1={f1:.1%}')
print()

# Test 3: Verify signal attenuation
print('=== Attenuation Check ===')
sys2 = ServiceSystem()
inject_fault(sys2, 'Database', FAULT_COMPLETE_FAILURE, SEVERITY_HIGH)
evts = propagate(sys2, G, 'Database')
db_signal = sys2.get('Database').signal_intensity
pay_signal = sys2.get('Payment').signal_intensity
ord_signal = sys2.get('Orders').signal_intensity
print(f'Database signal: {db_signal:.1f}')
print(f'Payment signal:  {pay_signal:.1f} (expected < Database)')
print(f'Orders signal:   {ord_signal:.1f} (expected < Payment)')
assert pay_signal < db_signal, 'Signal should attenuate!'
assert ord_signal < pay_signal, 'Signal should attenuate further!'
print('Attenuation: PASS')

# Test 4: Reset
print()
print('=== Reset Test ===')
system.reset()
for svc in system.all():
    assert svc.status == 'healthy', f'{svc.name} should be healthy after reset'
    assert svc.anomaly == 0.0, f'{svc.name} anomaly should be 0 after reset'
print('Reset: PASS')

print()
print('=== All tests passed ===')

