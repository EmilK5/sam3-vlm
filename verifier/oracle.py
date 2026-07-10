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
  - Sam3Oracle  : optional SAM3-presence channel for segmentable noun-phrase queries
"""

import base64
import io
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# yes/no/unsure -> +1/-1/0. Anything else (incl. missing) maps to 0.
_ANSWER_MAP = {"yes": 1, "no": -1, "unsure": 0}


class MockOracle:
    """
    Answers each query with its template value for `true_class`, with optional
    seeded flips. Deterministic given (true_class, noise, seed). For tests only.
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
    """
    Answers the whole query set for a crop in one structured VQA call to an
    OpenAI-compatible chat endpoint (e.g. Qwen-3-VL). temperature=0, strict JSON.

    On parse failure it retries up to `cfg.oracle_max_retries` times, then logs a
    warning and returns all-zeros

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
        from openai import OpenAI
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
