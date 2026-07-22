# Phase 5 and Phase 6 Completion

## Phase 5: graph-level posterior state

The candidate graph now keeps two independent decision systems:

- legacy verifier scores and `classification`, preserved for old policies;
- an optional `PatchBelief`, used by the ASHT controller.

An intermediate posterior leader is exposed as `temporary_map_class`.  A class
becomes `final_declaration` only after the belief is stopped or a budget forces
a declaration.  Legacy `classification` is never overwritten by ASHT.

Each node records complete lineage:

- raw detections and passes that supported it;
- registration and deduplication decisions;
- verification actions;
- observations and belief updates;
- semantic coordinates already queried;
- positive and negative exemplar uses.

Graph serialization uses schema version `2.0`.  It preserves belief histories,
lineage, and boolean masks using a compact deterministic RLE.  The loader also
accepts the former `{ "nodes": [...] }` graph format without inventing a
posterior for legacy nodes.

Posterior-based exemplars are eligible only when the node has stopped with the
requested class and exceeds the configured confidence threshold.  Unresolved
and budget-exhausted nodes are not trusted exemplars.

## Phase 6: static-action ASHT runner

`StaticAshtRunner` validates the end-to-end controller before Qwen action
generation is introduced.  It can start from an existing graph or bootstrap an
empty graph with one configured discovery query.

The controller performs the following sequence:

1. initialize a posterior on every candidate;
2. select the unresolved node with maximum entropy;
3. bind a finite static action bank to the node and ROI;
4. exclude semantic actions already used on that node;
5. construct a surrogate kernel for every action;
6. rank actions by EIG or EIG divided by expected cost;
7. execute the selected targeted SAM3 query;
8. suppress duplicate outputs without registering verification detections;
9. encode the response as not-found, weak-match, or strong-match;
10. apply the Bayesian update and compute realized information;
11. stop by confidence, node budget, global budget, or exhausted action bank;
12. continue until every node is resolved;
13. return hard count, soft count, variance, and unresolved count.

Every controller pass produces a canonical `PassRecord` with candidate actions,
all action scores, the selected action, SAM3 output, suppression records,
observation, kernel, posterior update, stopping decision, graph snapshots, and
cumulative cost.  All nested records are also emitted individually to the
provenance sink.

## Compatibility

- Existing graph construction and legacy verdict methods remain callable.
- Existing `execute_pass` behavior is unchanged.
- Staged registration now assigns run-scoped node IDs when a provenance sink is
  present and records detection lineage on both node creation and reinforcement.
- The new runner does not call Qwen and does not perform adaptive tiling.
