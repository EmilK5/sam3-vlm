"""
verifier/oracle.py

Query-answering oracles for the FM+V-IP verifier. Each oracle exposes a single
shared method:

    answer_batch(crop_pil, query_set) -> np.ndarray[int8] of length M in {-1,0,1}

where +1 = yes, -1 = no, 0 = unsure / undecidable. The returned vector is the
answer to each of the M queries for one candidate crop; vip.run_ip consumes it.

Three implementations:
  - MockOracle  : deterministic template-based answers with optional noise (tests).
  - QwenOracle  : one structured VQA call to an OpenAI-compatible endpoint (primary).
  - Sam3Oracle  : optional SAM3-presence channel for segmentable noun-phrase queries.

Model clients (`openai`, `torch`) are imported lazily inside the methods that
need them, so this module imports cleanly on a CPU-only box with neither
installed, and MockOracle/QwenOracle-parsing stay fully testable offline.
"""

import base64
import io
import json
import logging

import numpy as np

from verifier.queries import QuerySet

logger = logging.getLogger(__name__)

# yes/no/unsure -> +1/-1/0. Anything else (incl. missing) maps to 0.
_ANSWER_MAP = {"yes": 1, "no": -1, "unsure": 0}


class MockOracle:
    """Answers each query with its template value for `true_class`, with optional
    seeded flips. Deterministic given (true_class, noise, seed). For tests only.

    A "flip" (probability `noise` per query) replaces the clean template answer
    with a uniformly-chosen one of the other two answer values.
    """

    def __init__(self, true_class: str, noise: float = 0.0, seed: int = 0):
        self.true_class = true_class
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)

    def answer_batch(self, crop_pil, query_set) -> np.ndarray:
        out = np.empty(len(query_set.queries), dtype=np.int8)
        for m, query in enumerate(query_set.queries):
            a = query.templates[self.true_class]
            if self.noise > 0.0 and self.rng.random() < self.noise:
                alternatives = [v for v in (-1, 0, 1) if v != a]
                a = int(self.rng.choice(alternatives))
            out[m] = a
        return out


class QwenOracle:
    """Answers the whole query set for a crop in one structured VQA call to an
    OpenAI-compatible chat endpoint (e.g. Qwen-3-VL). temperature=0, strict JSON.

    On parse failure it retries up to `cfg.oracle_max_retries` times, then logs a
    warning and returns all-zeros (which makes V-IP fall back to the prior, i.e.
    "unresolved" — never a wrong confident verdict).

    `client` may be injected for offline testing; otherwise an `openai` client is
    built lazily from cfg.oracle_base_url. QWEN_API_KEY is optional: it defaults
    to a placeholder that keyless local servers (e.g. vLLM) ignore.
    """

    SYSTEM_PROMPT = (
        "You are a careful visual inspector. You are shown one cropped image and a "
        "numbered list of yes/no questions about the single central object in it. "
        "Answer every question with 'yes', 'no', or 'unsure'. Reply with ONLY a "
        "JSON object of the form {\"answers\": {\"q01\": \"yes\", \"q02\": \"no\", ...}} "
        "and nothing else."
    )

    def __init__(self, cfg, client=None):
        self.cfg = cfg
        self._injected_client = client

    # --- public API ---

    def answer_batch(self, crop_pil, query_set) -> np.ndarray:
        M = len(query_set.queries)
        client = self._client()
        messages = self._build_messages(crop_pil, query_set)

        for attempt in range(self.cfg.oracle_max_retries + 1):
            try:
                content = self._request(client, messages)
            except Exception as exc:  # network/client error: retry like a parse failure
                logger.warning("QwenOracle: request failed (%s) (attempt %d).", exc, attempt + 1)
                continue
            answers = self._parse_answers(content, query_set)
            if answers is not None:
                return answers
            logger.warning("QwenOracle: unparseable response (attempt %d).", attempt + 1)

        logger.warning("QwenOracle: giving up after %d attempts; returning all-zeros.",
                       self.cfg.oracle_max_retries + 1)
        return np.zeros(M, dtype=np.int8)

    # --- helpers (split out so parsing is unit-testable without a network) ---

    def _client(self):
        if self._injected_client is not None:
            return self._injected_client
        import os
        from openai import OpenAI  # lazy: not needed for tests / other oracles
        return OpenAI(
            base_url=self.cfg.oracle_base_url,
            api_key=os.environ.get("QWEN_API_KEY", "EMPTY"),
        )

    def _build_messages(self, crop_pil, query_set):
        buf = io.BytesIO()
        crop_pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        image_url = f"data:image/png;base64,{b64}"

        lines = [f"{i + 1}. [{q.id}] {q.text}" for i, q in enumerate(query_set.queries)]
        question_block = "\n".join(lines)
        text = (
            "Answer these yes/no questions about the central object in the image.\n"
            f"{question_block}\n\n"
            "Reply with ONLY JSON: {\"answers\": {\"q01\": \"yes|no|unsure\", ...}}."
        )
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]

    def _request(self, client, messages) -> str:
        response = client.chat.completions.create(
            model=self.cfg.oracle_model_name,
            temperature=self.cfg.oracle_temperature,
            messages=messages,
        )
        return response.choices[0].message.content

    def _parse_answers(self, content, query_set):
        """Map a raw response string to an int8 answer vector, or None if unparseable."""
        try:
            data = json.loads(content)
            answers = data["answers"]
            if not isinstance(answers, dict):
                return None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

        out = np.zeros(len(query_set.queries), dtype=np.int8)
        for m, query in enumerate(query_set.queries):
            raw = answers.get(query.id)  # missing id -> None -> 0
            out[m] = _ANSWER_MAP.get(str(raw).strip().lower(), 0)
        return out


class Sam3Oracle:
    """Optional SAM3-presence channel (off by default, retained for ablation).

    Only answers queries that carry a `sam3_phrase` (a segmentable noun phrase
    such as "stem" or "leaf blade"); all other queries get 0. For a query with a
    phrase, the answer is +1 if the SAM3 presence score on the crop is >= tau,
    else -1. Mirrors the presence-scoring pattern in
    inference.verify_box_semantics. Returns 0 for every query when no query in
    the set carries a phrase (e.g. the default green_citrus set), calling no
    model.
    """

    def __init__(self, processor, tau: float):
        self.processor = processor
        self.tau = float(tau)

    def answer_batch(self, crop_pil, query_set) -> np.ndarray:
        out = np.zeros(len(query_set.queries), dtype=np.int8)
        for m, query in enumerate(query_set.queries):
            phrase = getattr(query, "sam3_phrase", None)
            if not phrase:
                continue
            score = self._presence_score(crop_pil, phrase)
            out[m] = 1 if score >= self.tau else -1
        return out

    def _presence_score(self, crop_pil, phrase: str) -> float:
        import torch  # lazy: only the SAM3 channel needs it
        model = self.processor.model
        inputs = self.processor(images=crop_pil, text=phrase, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.001, target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        scores = result["scores"]
        return float(scores[0]) if len(scores) > 0 else 0.0


def _subset_view(query_set, idxs) -> QuerySet:
    """A QuerySet holding only the queries at `idxs`, preserving
    classes/epsilon/names (the Query objects are shared, so route/cv_check/
    sam3_phrase come along)."""
    return QuerySet(
        concept=query_set.concept,
        classes=query_set.classes,
        epsilon=query_set.epsilon,
        queries=[query_set.queries[m] for m in idxs],
        class_names=query_set.class_names,
    )


class RouterOracle:
    """Route each query to the cheapest channel that can answer it, so the VLM is
    asked as little as possible (docs/active_perception_formulation.md; v2 step
    8.4). Each query's `route` (verifier/queries.py) selects the channel:

        "cv"   -> deterministic crop features (cv_answers), NO model call
        "sam3" -> SAM3 presence score (needs a processor; 0 for every sam3 query
                  when processor is None)
        "vlm"  -> the residual, answered by ONE vlm_oracle.answer_batch call on a
                  subset view of just the vlm-routed queries (0-fill when
                  vlm_oracle is None)

    Same interface as the other oracles: answer_batch(crop_pil, query_set) ->
    int8 vector of length M in {-1, 0, +1}. After each call, n_vlm_calls (0 or 1)
    and n_sam_calls hold how many real model calls the LAST batch made, so a
    caller can meter only real model usage. A missing channel yields 0 (unsure) --
    never a wrong confident answer (V-IP then falls back to the prior for it).

    Models are injected: `processor` (SAM3) and `vlm_oracle` (e.g. a QwenOracle)
    are passed in, never imported-and-called globally, so RouterOracle stays
    testable offline with both absent.
    """

    def __init__(self, cfg, processor=None, vlm_oracle=None):
        self.cfg = cfg
        self.processor = processor
        self.vlm_oracle = vlm_oracle
        # sam3_presence_tau is a getattr default here; step 8.5 promotes it to config.
        tau = getattr(cfg, "sam3_presence_tau", 0.5)
        self._sam3 = Sam3Oracle(processor, tau) if processor is not None else None
        self.n_vlm_calls = 0
        self.n_sam_calls = 0

    def answer_batch(self, crop_pil, query_set) -> np.ndarray:
        M = len(query_set.queries)
        out = np.zeros(M, dtype=np.int8)
        self.n_vlm_calls = 0
        self.n_sam_calls = 0

        routes = [getattr(q, "route", "vlm") for q in query_set.queries]
        cv_idx = [m for m in range(M) if routes[m] == "cv"]
        sam_idx = [m for m in range(M) if routes[m] == "sam3"]
        vlm_idx = [m for m in range(M) if routes[m] == "vlm"]

        # cv: answered locally, no model call.
        if cv_idx:
            from verifier import cv_answers  # lazy: only cv-routed queries need cv2
            for m in cv_idx:
                out[m] = cv_answers.answer(crop_pil, query_set.queries[m].cv_check)

        # sam3: presence per phrase (one presence call per sam3 query). 0 when no
        # processor -- an absent channel is unsure, never a wrong confident answer.
        if sam_idx and self._sam3 is not None:
            sam_answers = self._sam3.answer_batch(crop_pil, _subset_view(query_set, sam_idx))
            self.n_sam_calls = len(sam_idx)
            for j, m in enumerate(sam_idx):
                out[m] = sam_answers[j]

        # vlm: ONE batched call over just the residual queries. 0 when no oracle.
        if vlm_idx and self.vlm_oracle is not None:
            vlm_answers = self.vlm_oracle.answer_batch(crop_pil, _subset_view(query_set, vlm_idx))
            self.n_vlm_calls = 1
            for j, m in enumerate(vlm_idx):
                out[m] = vlm_answers[j]

        return out
