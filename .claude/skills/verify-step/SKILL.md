---
name: verify-step
description: Prepare and assist human verification of a step that is in the [~] state; run automated checks and print the manual checklist.
---

Help the human verify a completed step. Procedure:

1. Determine the step: the argument if given (e.g. `/verify-step 1.4`),
   otherwise the first `[~]` entry in `PROGRESS.md`. If none is `[~]`, say so
   and stop.
2. Read that step's specification in `docs/implementation_plan.md`, especially
   its "You verify" block.
3. Run the automated part yourself: `pytest -q` (full suite, not just the new
   file), plus any cheap CPU-only checks from the step (schema validation,
   imports, `--help` of new CLIs). Report results honestly — if anything is
   red or flaky, say so first.
4. Show a diff-level summary of what the step changed: `git diff --stat` (or
   list changed files if git is not initialized) and call out any file
   touched that the step did NOT name. That is a rule violation and must be
   flagged prominently, not smoothed over.
5. Print the manual verification checklist as numbered, copy-pasteable
   commands with expected outcomes, adapted to this machine's paths (check
   `config.py` for dataset/image paths). GPU commands are for the human to
   run — do not run model-loading commands yourself unless explicitly asked.
6. Remind the human: if satisfied, they flip the step to `[x]` in
   `PROGRESS.md` themselves (or ask you to); if not, describe what failed and
   run `/next-step <same id>` guidance to fix within the same step's scope.

Do not fix problems silently during verification. Report first, fix only when
asked.
