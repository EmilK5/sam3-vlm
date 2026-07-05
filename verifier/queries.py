"""
verifier/queries.py

Load and strictly validate FM+V-IP query sets from JSON.

A query set specifies, for a target concept, a list of binary visual queries
and, for each query, the expected per-class answer template tau_{k,m} in
{-1, 0, +1} (no / uninformative / yes). These templates and a fixed answer
noise rate epsilon fully determine the closed-form V-IP posterior in vip.py.
The set is generated once by an LLM (scripts/gen_queries.py), hand-reviewed,
and checked into queries/ as JSON. Runtime never calls an LLM to make queries.
"""

import dataclasses
import json
import logging

logger = logging.getLogger(__name__)

# Allowed values for a per-class answer template tau_{k,m}.
VALID_TEMPLATE_VALUES = {-1, 0, 1}


@dataclasses.dataclass
class Query:
    """A single binary visual query and its per-class expected answers.

    templates maps each class name to tau in {-1, 0, +1}:
    +1 = a "yes" is expected for that class, -1 = a "no" is expected,
    0 = the query is uninformative for that class.

    sam3_phrase (optional): a segmentable noun phrase (e.g. "stem") for the
    SAM3-presence oracle channel; None means only VLM oracles answer this query.
    """
    id: str
    text: str
    templates: dict
    sam3_phrase: str = None


@dataclasses.dataclass
class QuerySet:
    """A validated set of queries for one target concept.

    concept:      free-text name of the target concept, e.g. "green citrus".
    classes:      the class labels, e.g. ["target", "distractor", "spurious"].
    class_names:  human-readable name per class, e.g. {"target": "fruit", ...}.
    epsilon:      answer-noise rate in (0, 0.5) used by the V-IP likelihood.
    queries:      list of Query, in file order.
    """
    concept: str
    classes: list
    epsilon: float
    queries: list
    class_names: dict = dataclasses.field(default_factory=dict)


def _validate(data: dict) -> QuerySet:
    """Validate a raw query-set dict and build a QuerySet, or raise ValueError."""
    # --- required top-level keys ---
    for key in ("concept", "classes", "epsilon", "queries"):
        if key not in data:
            raise ValueError(f"Query set missing required top-level key '{key}'.")

    concept = data["concept"]
    classes = data["classes"]
    epsilon = data["epsilon"]
    raw_queries = data["queries"]
    class_names = data.get("class_names", {})

    if not isinstance(classes, list) or len(classes) == 0:
        raise ValueError("'classes' must be a non-empty list of class labels.")
    if len(set(classes)) != len(classes):
        raise ValueError(f"'classes' contains duplicate labels: {classes}.")

    if not isinstance(epsilon, (int, float)) or not (0.0 < epsilon < 0.5):
        raise ValueError(f"'epsilon' must be a float in the open interval (0, 0.5); got {epsilon!r}.")

    if class_names:
        missing_names = set(classes) - set(class_names)
        if missing_names:
            raise ValueError(f"'class_names' is missing entries for classes: {sorted(missing_names)}.")

    if not isinstance(raw_queries, list) or len(raw_queries) == 0:
        raise ValueError("'queries' must be a non-empty list.")

    class_set = set(classes)
    seen_ids = set()
    queries = []
    for i, q in enumerate(raw_queries):
        for key in ("id", "text", "templates"):
            if key not in q:
                raise ValueError(f"Query at index {i} is missing required key '{key}'.")

        qid = q["id"]
        if qid in seen_ids:
            raise ValueError(f"Duplicate query id '{qid}'.")
        seen_ids.add(qid)

        templates = q["templates"]
        if not isinstance(templates, dict):
            raise ValueError(f"Query '{qid}' has non-dict 'templates'.")

        template_classes = set(templates)
        if template_classes != class_set:
            missing = class_set - template_classes
            extra = template_classes - class_set
            raise ValueError(
                f"Query '{qid}' templates must cover exactly the classes {sorted(class_set)}; "
                f"missing={sorted(missing)}, unexpected={sorted(extra)}."
            )

        for cls, val in templates.items():
            if val not in VALID_TEMPLATE_VALUES:
                raise ValueError(
                    f"Query '{qid}' template for class '{cls}' is {val!r}; "
                    f"must be one of {sorted(VALID_TEMPLATE_VALUES)}."
                )

        sam3_phrase = q.get("sam3_phrase")
        if sam3_phrase is not None and (not isinstance(sam3_phrase, str) or not sam3_phrase.strip()):
            raise ValueError(f"Query '{qid}' has a non-string/empty 'sam3_phrase': {sam3_phrase!r}.")

        queries.append(Query(id=qid, text=q["text"], templates=dict(templates),
                             sam3_phrase=sam3_phrase))

    return QuerySet(
        concept=concept,
        classes=list(classes),
        epsilon=float(epsilon),
        queries=queries,
        class_names=dict(class_names),
    )


def load_query_set(path: str) -> QuerySet:
    """Load a query set from a JSON file, validating it strictly.

    Raises ValueError (with a helpful message) if the schema is malformed.
    """
    with open(path, "r") as f:
        data = json.load(f)
    query_set = _validate(data)
    logger.info(
        "Loaded query set '%s' with %d queries over classes %s.",
        query_set.concept, len(query_set.queries), query_set.classes,
    )
    return query_set
