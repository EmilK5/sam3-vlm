"""
scripts/gen_queries.py

One-time, human-run generation of an FM+V-IP query set for a target concept
via an OpenAI-compatible chat endpoint (e.g. Qwen-3-VL served with vLLM /
DashScope). The output JSON must be hand-reviewed before it is trusted; the
checked-in queries/green_citrus.json was produced and edited this way.

Runtime NEVER calls this. It is a developer tool. Endpoint config is read
from the environment: QWEN_BASE_URL, QWEN_API_KEY, QWEN_MODEL.
"""

import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifier.queries import load_query_set

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert visual-attribute engineer building an interpretable "
    "image classifier in the Information Pursuit (FM+V-IP) style. You design "
    "short, binary, visually-answerable questions about a single fixed-size "
    "image crop, and you specify the expected answer of each question for each "
    "class. You output ONLY strict JSON."
)

# Modeled on the FM+V-IP query-generation prompt ("list the useful visual
# attributes and their values of ..."), but asking directly for our schema.
USER_PROMPT_TEMPLATE = """\
Target concept: "{concept}".

We verify candidate image crops into three classes:
  - "target"     (the object itself, e.g. the fruit / {concept}),
  - "distractor" (a plausible look-alike, e.g. a leaf),
  - "spurious"   (clutter: bark, sky, soil, blur, or no clear object).

List {n} useful visual attributes for telling these classes apart, each phrased
as a single yes/no question answerable by looking at ONE cropped image.
For every question, give the expected answer for each class as a template value:
  +1  = "yes" is expected for that class,
  -1  = "no"  is expected for that class,
   0  = the question is uninformative for that class.

Return ONLY a JSON object with EXACTLY this schema (no prose, no code fences):

{{
  "concept": "{concept}",
  "classes": ["target", "distractor", "spurious"],
  "class_names": {{"target": "fruit", "distractor": "leaf", "spurious": "clutter"}},
  "epsilon": 0.15,
  "queries": [
    {{"id": "q01", "text": "...", "templates": {{"target": 1, "distractor": -1, "spurious": 0}}}}
  ]
}}

Use zero-padded ids q01, q02, .... Templates must use only -1, 0, or +1.
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate an FM+V-IP query set JSON via an LLM.")
    parser.add_argument("--concept", required=True, help="Target concept, e.g. 'green citrus'.")
    parser.add_argument("--n", type=int, default=22, help="Approximate number of queries to request.")
    parser.add_argument("--out", required=True, help="Path to write the generated JSON.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    base_url = os.environ.get("QWEN_BASE_URL")
    api_key = os.environ.get("QWEN_API_KEY")
    model = os.environ.get("QWEN_MODEL")
    if not base_url or not model:
        raise SystemExit("Set QWEN_BASE_URL and QWEN_MODEL "
                         "(QWEN_API_KEY is optional for keyless local servers like vLLM).")

    # Imported lazily so the rest of the repo does not depend on `openai`.
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    user_prompt = USER_PROMPT_TEMPLATE.format(concept=args.concept, n=args.n)

    logger.info("Requesting query set for concept '%s' from model '%s'...", args.concept, model)
    response = client.chat.completions.create(
        model=model,
        temperature=args.temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content

    # Parse to fail fast on malformed JSON before writing.
    data = json.loads(content)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote %s. Validating...", args.out)

    # Validate with the same strict loader the runtime uses; surfaces problems now.
    query_set = load_query_set(args.out)
    logger.info(
        "Validation OK: %d queries. REVIEW THIS FILE BY HAND before trusting it.",
        len(query_set.queries),
    )


if __name__ == "__main__":
    main()
