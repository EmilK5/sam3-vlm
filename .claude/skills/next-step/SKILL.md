---
name: next-step
description: Implement the next unfinished step from the implementation plan, exactly one step, following all repo rules.
---

Implement the next step of the project. Follow this procedure exactly:

1. Read `PROGRESS.md`. Find the first step marked `[~]`. If one exists, STOP
   and tell the human: that step is still awaiting their manual verification;
   offer to run `/verify-step` instead. Never start a new step while another
   is `[~]`, unless the human explicitly says to proceed anyway.
2. Otherwise find the first step marked `[ ]` (respecting the ordering note at
   the top of PROGRESS.md). If the human passed an argument like `/next-step
   1.3`, use that step instead — but warn if its prerequisites are not `[x]`.
3. Open `docs/implementation_plan.md` and read the FULL specification of that
   step: Goal, Files, Prompt, and the "You verify" block. Also re-read the
   "Ground rules" and "Design decisions" sections at the top of the plan. If
   the step references formulas, read the matching section of
   `docs/proposal.md` before writing code.
4. Restate the step in 3–5 bullet points (files to touch, what gets built,
   what stays untouched) and list any ambiguity. If something in the plan
   contradicts the current code, STOP and ask the human — do not improvise.
5. Implement. Touch ONLY the files named by the step. New behavior behind
   config flags with old defaults. Models/oracles passed as arguments. No new
   dependencies without asking.
6. Write the step's pytest file. Run `pytest -q`. Iterate until green. Tests
   must be CPU-only, no network, no model weights, and fast.
7. Update `PROGRESS.md`: flip the step to `[~]` and append a one-line note.
   If you deviated from the plan in any way, add a dated line to the
   "Notes / decisions log".
8. Finish with a handoff summary: what was built, what was deliberately NOT
   built, the exact manual verification commands from the step's "You verify"
   block, and anything you want the human to look at closely.

Never mark a step `[x]`. Never continue to the following step.
