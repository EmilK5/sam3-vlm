# PROGRESS

Status legend: `[ ]` todo · `[~]` implemented, awaiting human verification ·
`[x]` verified by human (ONLY the human flips to `[x]`).

Full step specifications live in `docs/implementation_plan.md`. Do not start a
step whose predecessor in the same phase is still `[ ]`. Cross-phase order may
follow the "Suggested order of the first week" in the plan.

## Phase 0 — Scaffold & harness
- [ ] 0.1 Package layout + config.py
- [ ] 0.2 CLI runner for the existing cascade (scripts/run_image.py, graph.to_dict)
- [ ] 0.3 Dataset loader (eval/datasets.py)

## Phase 1 — FM+V-IP verifier
- [ ] 1.1 Query sets + generation script (verifier/queries.py, queries/green_citrus.json)
- [ ] 1.2 Oracles (MockOracle, QwenOracle, Sam3Oracle)
- [ ] 1.3 Training-free V-IP core (verifier/vip.py, pure numpy)
- [ ] 1.4 Verify API + demo (verifier/verify.py, scripts/demo_verify.py)  ← GO/NO-GO checkpoint
- [ ] 1.5 Pipeline integration behind verifier="vip" flag

## Phase 2 — Belief state
- [ ] 2.1 Support, jitter, signatures on nodes (graph.py, pipeline dedup branch)
- [ ] 2.2 belief.py: w_i, U, discovery curve, phi summary

## Phase 3 — Actions & cost
- [ ] 3.1 Action layer (agent/actions.py)
- [ ] 3.2 Cost meter (agent/budget.py)

## Phase 4 — Heuristic controller
- [ ] 4.1 VoI heuristic policy + episode runner

## Phase 5 — VLM orchestrator
- [ ] 5.1 Scene inspection z_t (agent/inspect.py)
- [ ] 5.2 VLM policy with strict validation (agent/policy_vlm.py)

## Phase 6 — Evaluation
- [ ] 6.1 Matching & detection metrics (eval/matching.py)
- [ ] 6.2 Sweep runner (eval/run_eval.py)
- [ ] 6.3 Results table & accuracy-vs-cost plot (eval/report.py)

## Notes / decisions log
<!-- Append dated one-liners here when a step deviates from the plan. -->
