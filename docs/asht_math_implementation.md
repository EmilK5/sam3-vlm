# Phase 4: ASHT Mathematical Core

The implementation lives under `agent/asht/` and has no model dependencies.

## Modules

- `belief.py`: normalized patch priors/posteriors and Bayesian updates.
- `kernels.py`: finite sensor profiles and the surrogate mixture kernel.
- `information.py`: entropy, predictive distributions, EIG, RIG/KL, and action
  ranking, optionally adjusted by expected cost.
- `observations.py`: finite targeted observations (`not_found`, `weak_match`,
  `strong_match`) using overlap and confidence.
- `stopping.py`: posterior-confidence and budget stopping.
- `counting.py`: hard count, posterior mean count, variance, and unresolved count.
- `update.py`: one complete update that creates canonical kernel, EIG,
  observation, belief-update, and stopping records.

## Current boundary

Phase 4 is deterministic and separately testable. It is not yet attached to
`OrchardNode`; that graph migration remains Phase 5. Qwen action generation and
live ASHT orchestration remain later phases.
