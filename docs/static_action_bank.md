# Static ASHT Action Bank

The static action bank is a temporary controller input used to validate the
mathematical loop independently of Qwen.

Each template specifies:

- target, negative, or exploration family;
- text prompt and semantic key;
- class-conditional descriptor probabilities;
- SAM3 threshold;
- ROI padding;
- expected sensing cost;
- whether trusted positive or negative exemplars may be attached;
- a concise rationale and expected visual distinction.

At runtime a template is bound to one graph node, clipped to the image, assigned
a stable action ID, and converted into a canonical `SensingActionRecord` and a
surrogate observation kernel.  Semantic keys are consumed at most once per node
by default so repeated equivalent prompts do not receive independent Bayesian
weight.

`green_citrus_static_action_bank()` provides a small initial bank for system
validation.  Its probabilities are configured assumptions, not calibrated
experimental values, and should be replaced by dataset configuration once the
calibration workflow is implemented.
