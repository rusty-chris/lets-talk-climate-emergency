"""Issue #4 spike: flagship 10,000-year CO2 + temperature chart, end-to-end.

Spike code (IMPLEMENTATION.md section 2 item 6): exploratory de-risking, not
production. Production modules (`charts/pack.py`, `charts/transforms.py`,
`charts/spec.py`, `charts/render.py`) start from red tests in issues #15-#17
and may reuse or rewrite what is here. The characterisation tests under
`tests/unit/test_spike_*.py` pin this code's behaviour as a canary, not as a
frozen contract.

Findings, spot-checks and the vocabulary gaps fed into #15 are recorded in
`reviews/spike-04-chart-findings.md`.
"""
