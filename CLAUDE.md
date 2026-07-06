# CLAUDE.md — Orchard Active-Perception Counter

Zero-shot active-perception counting system. SAM3 is a controllable *sensor*
(prompts, exemplars, thresholds, tiles, regions); a candidate graph is the
belief state; an FM+V-IP verifier classifies candidates interpretably
(fruit / leaf / spurious); a policy (heuristic or Qwen-3-VL) chooses the next
sensing action. Research code for a paper — correctness and auditability beat
elegance and performance.

## The one rule that governs everything

**Work proceeds step by step from `docs/implementation_plan.md`, tracked in
`PROGRESS.md`. One step per session. Never implement ahead of the current
step, never batch steps, never "improve" things outside the step's named
files.** The human verifies each step by hand before the next one starts.
Use `/next-step` to begin work and `/verify-step` to prepare handoff.

## Repo map

- `app.py` — Gradio UI (existing, working; don't restructure)
- `pipeline.py` — pass execution: propose → NMS → register → verify → feedback
- `inference.py` — SAM3 loading, raw inference, NMS variants, plotting
- `graph.py` — candidate graph (`OrchardGraph`, `OrchardNode`)
- `config.py` — ALL tunables (created in step 0.1; no magic numbers elsewhere)
- `verifier/` — FM+V-IP: `queries.py`, `oracle.py`, `vip.py`, `verify.py`
- `agent/` — `belief.py`, `actions.py`, `budget.py`, `inspect.py`, policies, `runner.py`
- `eval/` — `datasets.py`, `matching.py`, `metrics.py`, `run_eval.py`
- `scripts/` — CLI entry points; `queries/` — reviewed query-set JSONs
- `docs/implementation_plan.md` — the authoritative plan (read the current step fully before coding)
- `docs/proposal.md` — the research proposal with all math. Read only the
  section relevant to the current step; it is long.

## Hard constraints (violating these ruins validated baselines)

1. NEVER change the behavior of `apply_nms`, `apply_nms_dualgate`,
   `tiled_engine`, or `run_raw_inference`. Additive, backward-compatible
   parameters with safe defaults are allowed (existing patterns:
   `return_indices=False`, `return_masks=False`).
2. New behavior goes behind a config flag; the old path stays the default
   (e.g. `verifier="ioc"` default, `"vip"` opt-in).
3. The VLM must never inject detections. Candidates originate from SAM3 only,
   and no VLM-supplied box may ever become a candidate/node. VLM boxes are
   allowed solely as sensing ROIs that parameterize a SAM3 query (agent
   `LookROIA`): such an ROI only steers where SAM3 looks and is never added to
   the graph.
4. Model objects (SAM3 processor, oracles) are passed as arguments, never
   imported and called globally. Everything must run with `MockOracle` and a
   stubbed processor.
5. No new dependencies without asking. Allowed: numpy, PIL, cv2, matplotlib,
   scipy, pytest, openai (client only), torch/transformers (already present).

## Code style

Plain functions and small dataclasses. No abstract base classes, no async, no
clever metaprogramming. Short docstrings that state units and coordinate
frames (global-frame xyxy pixels unless stated). Log with `logging`, never
`print`, except in CLI `main()`s. Every step ships exactly one new pytest
file under `tests/`.

## Testing & commands

- `pytest -q` — must stay green, CPU-only, no network, no SAM3/Qwen loading,
  < 10 s total. Use `MockOracle` and stub processors; never download weights
  in tests.
- `python scripts/run_image.py --image X --prompt "green fruit" --passes N`
  — GPU smoke test (human runs this; do not run it yourself unless asked —
  model load takes minutes).
- `python scripts/demo_verify.py --image X --boxes x1,y1,x2,y2 --oracle mock`
  — verifier demo.

## Environment

- SAM3 repo path and BPE path are machine-specific (currently hardcoded in
  `app.py` as `/home/ekielar/sam3`; step 0.1 moves them to `config.py`).
- Qwen-3-VL is reached through an OpenAI-compatible endpoint. Read
  `QWEN_BASE_URL`, `QWEN_API_KEY`, `QWEN_MODEL` from environment variables.
  Never hardcode keys or URLs; never commit them; never print `QWEN_API_KEY`.
- GPU box only for real runs; all development/tests must work on CPU.

## Definition of done for a step

1. Only the step's named files were touched.
2. `pytest -q` green.
3. `PROGRESS.md` entry moved from `[ ]` to `[~]` (awaiting human
   verification) with a one-line note of what to check. Only the human ever
   marks `[x]`.
4. A 3–6 line summary: what was built, what was NOT built, exact commands
   for the human's manual verification (from the step's "You verify" block).
