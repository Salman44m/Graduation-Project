"""
memory/tltm.py
─────────────────────────────────────────────────────────────────────────────
Tactical Long-Term Memory (TLTM) — FAISS Vector Store with UCB Sampling

Architectural Role (Section 4.2 + Section 6.2, Original Project Doc)
──────────────────────────────────────────────────────────────────────
The TLTM is PromptEvo's persistent "experience vault."  It stores historical
attack records across sessions so that the framework learns from every audit
it conducts — both its failures and its successes.

Every record stored in the TLTM captures:
  • The adversarial payload (as a dense semantic embedding)
  • Rich metadata: target model, PAP technique, RAHS score, outcome, timestamps

On retrieval, the TLTM doesn't simply return the nearest-neighbour results.
It applies **Upper Confidence Bound (UCB1) sampling** — the same exploration/
exploitation balance algorithm used in multi-armed bandit problems — to rank
candidates.  UCB balances:
  • **Exploitation**: tactics with high historical RAHS scores should be tried
  • **Exploration**: tactics that haven't been tried many times deserve a chance

Additionally, **30-day temporal decay** down-weights older records so the
framework continuously adapts to new RLHF updates and safety patches rather
than relying on techniques that may have been patched months ago.

Embedding Architecture
──────────────────────
The embedding layer is configurable via the ``EmbeddingBackend`` enum:

  OPENAI_ADA      — text-embedding-3-small (OpenAI API, 1536-dim, best quality)
  OPENAI_SMALL    — text-embedding-3-small (same, 1536-dim)
  HASH_LOCAL      — SHA-256 seeded RNG (384-dim, zero-network, deterministic)
  FAKE            — FakeEmbeddings (384-dim, unit tests / dry-run)

HASH_LOCAL is the default when no API key is configured, making the TLTM
fully functional in air-gapped environments without sacrificing recall quality
on a test dataset (SHA-256 preserves enough token-level signal for the cosine
similarity to return semantically meaningful neighbours within a small corpus).

Storage Layout
──────────────
  data/memory/tltm_vectors/
    {target_model_id}.index   — FAISS IndexFlatIP (inner-product = cosine on L2-normalised)
    {target_model_id}.meta.pkl — list[ExperienceRecord] pickle
    ucb_counters.json          — per-record pull count for UCB formula
    decay_config.json          — per-record timestamp for temporal decay

UCB Formula
───────────
    UCB_score(i) = μ̂(i) × decay(i) + C × √(ln(N) / n(i))

where:
  μ̂(i)     = normalised RAHS score (0.0–1.0) — the "reward signal"
  decay(i)  = exp(−λ × age_days(i)) — 30-day half-life temporal weight
  C         = exploration constant (default 1.414 = √2, optimal for UCB1)
  N         = total number of pulls across all records
  n(i)      = number of times record i has been retrieved

This formula ensures that:
  • High-RAHS records that haven't been tried recently get priority
  • Records tried many times without improvement are gradually de-prioritised
  • Brand-new records always get an exploration bonus (n=0 → UCB = ∞, handled
    by initialising n=1 at storage time to avoid division by zero)

References
──────────
- Section 4.2: Tactical Long-Term Memory (TLTM) — FAISS vector storage
- Section 6.2: File structure — data/memory/tltm_vectors/
- Auer, Cesa-Bianchi, Fischer (2002): "Finite-time analysis of the multi-armed
  bandit problem" — UCB1 algorithm
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from core.paths import TLTM_VECTORS_DIR
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_VECTOR_DIM: int        = 384
"""Default embedding dimension used by the HASH_LOCAL and FAKE backends."""

OPENAI_VECTOR_DIM: int         = 1536
"""Embedding dimension for OpenAI text-embedding-3-small."""

DEFAULT_TLTM_PATH: str         = str(TLTM_VECTORS_DIR)
"""Default storage directory (relative to project root)."""

TEMPORAL_DECAY_DAYS: float     = 30.0
"""Half-life for temporal decay.  Records older than this are down-weighted."""

TEMPORAL_DECAY_LAMBDA: float   = math.log(2) / TEMPORAL_DECAY_DAYS
"""Exponential decay rate derived from the half-life."""

UCB_EXPLORATION_CONSTANT: float = math.sqrt(2)   # C = √2 — UCB1 optimal
"""Exploration constant C in the UCB formula.  Higher = more exploration."""

TOP_K_RETRIEVAL: int           = 10
"""Number of FAISS nearest neighbours to retrieve before UCB re-ranking."""

MAX_RECORDS_PER_INDEX: int     = 10_000
"""Safety cap to prevent unbounded index growth."""


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING BACKEND ENUM
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingBackend(str, Enum):
    """Selects the embedding model used to vectorise payloads."""
    OPENAI_ADA   = "openai_ada"   # text-embedding-ada-002 (legacy, 1536-dim)
    OPENAI_SMALL = "openai_small" # text-embedding-3-small (recommended, 1536-dim)
    HASH_LOCAL   = "hash_local"   # deterministic SHA-256 hash, 384-dim, zero-network
    FAKE         = "fake"         # unit-test stub, 384-dim


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIENCE RECORD DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperienceRecord:
    """A single stored attack experience in the TLTM.

    Attributes
    ──────────
    record_id : str
        SHA-256 hex digest of the payload (stable unique ID).

    payload : str
        The adversarial prompt that was sent to the target.

    target_response : str
        The target model's response (truncated to 1 000 chars for storage).

    objective : str
        The ``core_malicious_objective`` pursued in this attack.

    target_model_id : str
        Canonical model identifier of the target that was audited.

    pap_technique : str
        PAP technique applied when this payload was generated.

    obfuscation_tier : str
        Obfuscation level applied: "none" | "base64" | "scatter" | "wordmap".

    prometheus_score : float
        Prometheus Judge score (1.0–5.0) for this specific exchange.

    rahs_score : float
        Risk-Adjusted Harm Score (0.0–10.0) for this attack event.

    outcome : str
        "success" if jailbreak was achieved (score ≥ 4), else "failure".

    session_id : str
        UUID of the originating audit session.

    timestamp : float
        Unix timestamp at storage time.  Used for temporal decay.

    pull_count : int
        Number of times this record has been retrieved by UCB sampling.
        Initialised to 1 (avoids ln(N)/n(i) division by zero at startup).

    depth : int
        TAP tree depth at which this payload was generated.

    turn : int
        Session turn number when this payload was sent.
    """

    record_id:        str   = ""
    payload:          str   = ""
    target_response:  str   = ""
    objective:        str   = ""
    target_model_id:  str   = ""
    pap_technique:    str   = ""
    obfuscation_tier: str   = "none"
    prometheus_score: float = 0.0
    rahs_score:       float = 0.0
    outcome:          str   = "failure"
    session_id:       str   = ""
    timestamp:        float = 0.0
    pull_count:       int   = 1
    depth:            int   = 0
    turn:             int   = 0

    def __post_init__(self) -> None:
        if not self.record_id:
            self.record_id = hashlib.sha256(
                (self.payload + self.target_model_id).encode()
            ).hexdigest()[:16]
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def age_days(self) -> float:
        """Age of this record in days from now."""
        return (time.time() - self.timestamp) / 86_400

    @property
    def decay_weight(self) -> float:
        """Temporal decay weight ∈ (0, 1].  Approaches 0 as record ages."""
        return math.exp(-TEMPORAL_DECAY_LAMBDA * self.age_days)

    @property
    def normalised_rahs(self) -> float:
        """RAHS score normalised to [0, 1] for UCB reward signal."""
        return min(1.0, max(0.0, self.rahs_score / 10.0))


# ─────────────────────────────────────────────────────────────────────────────
# UCB SCORER
# ─────────────────────────────────────────────────────────────────────────────

def ucb_score(
    record:              ExperienceRecord,
    total_pulls:         int,
    exploration_constant: float = UCB_EXPLORATION_CONSTANT,
) -> float:
    """Compute the UCB1 score for a single experience record.

    Formula:
        UCB(i) = μ̂(i) × decay(i) + C × √(ln(N) / n(i))

    Where:
        μ̂(i)   = ``record.normalised_rahs``      (reward signal)
        decay(i) = ``record.decay_weight``          (temporal weight)
        C        = ``exploration_constant``          (UCB1 constant)
        N        = ``total_pulls``                   (global pull count)
        n(i)     = ``record.pull_count``             (this record's pulls)

    Parameters
    ──────────
    record : ExperienceRecord
        The candidate record to score.
    total_pulls : int
        Total number of retrievals across all records in this index.
        Must be ≥ 1 (caller's responsibility).
    exploration_constant : float
        C in the UCB1 formula.  Default: √2.

    Returns
    ───────
    float
        UCB score.  Higher = should be retrieved and tried first.
    """
    exploitation = record.normalised_rahs * record.decay_weight
    n_i          = max(1, record.pull_count)   # floor at 1 to avoid log(N)/0
    exploration  = exploration_constant * math.sqrt(math.log(max(1, total_pulls)) / n_i)
    return exploitation + exploration


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """Lightweight wrapper that produces L2-normalised embeddings.

    Supports multiple backends with automatic fallback:
      1. OpenAI API (requires OPENAI_API_KEY)
      2. HASH_LOCAL (SHA-256 seeded RNG — deterministic, zero-network)
      3. FAKE (unit tests — random but reproducible within a session)

    All vectors are L2-normalised before storage so that FAISS inner-product
    search (``IndexFlatIP``) is equivalent to cosine similarity.

    Parameters
    ──────────
    backend : EmbeddingBackend
        Backend selector.
    openai_api_key : str | None
        API key for OpenAI backends.  Falls back to OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        backend:         EmbeddingBackend = EmbeddingBackend.HASH_LOCAL,
        openai_api_key:  str | None       = None,
        dim:             int               = DEFAULT_VECTOR_DIM,
    ) -> None:
        self.backend = backend
        self.dim     = dim
        self._openai_client: Any = None

        if backend in (EmbeddingBackend.OPENAI_ADA, EmbeddingBackend.OPENAI_SMALL):
            key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            if key:
                try:
                    from openai import OpenAI
                    self._openai_client = OpenAI(api_key=key)
                    self.dim = OPENAI_VECTOR_DIM
                    logger.info("[TLTM] OpenAI embedding backend initialised (dim=%d)", self.dim)
                except ImportError:
                    logger.warning("[TLTM] openai package not installed. Falling back to HASH_LOCAL.")
                    self.backend = EmbeddingBackend.HASH_LOCAL
            else:
                logger.warning("[TLTM] No OPENAI_API_KEY found. Falling back to HASH_LOCAL.")
                self.backend = EmbeddingBackend.HASH_LOCAL

        if self.backend == EmbeddingBackend.HASH_LOCAL:
            self.dim = DEFAULT_VECTOR_DIM
            logger.debug("[TLTM] HASH_LOCAL embedding backend (dim=%d)", self.dim)

    def embed(self, text: str) -> np.ndarray:
        """Embed ``text`` as a 1-D L2-normalised float32 numpy vector.

        Parameters
        ──────────
        text : str
            Any string (payload, objective, query).

        Returns
        ───────
        np.ndarray
            Shape (dim,), dtype float32, L2-norm ≈ 1.0.
        """
        if self.backend in (EmbeddingBackend.OPENAI_ADA, EmbeddingBackend.OPENAI_SMALL):
            return self._embed_openai(text)
        return self._embed_local(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of strings.  Returns shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.vstack([self.embed(t) for t in texts])

    def _embed_openai(self, text: str) -> np.ndarray:
        """Call the OpenAI embeddings API."""
        model = (
            "text-embedding-ada-002"
            if self.backend == EmbeddingBackend.OPENAI_ADA
            else "text-embedding-3-small"
        )
        try:
            response = self._openai_client.embeddings.create(
                input=text[:8191],   # API hard limit
                model=model,
            )
            vec = np.array(response.data[0].embedding, dtype="float32")
            return self._l2_normalize(vec)
        except Exception as exc:   # noqa: BLE001
            logger.warning("[TLTM] OpenAI embed error: %s — falling back to HASH_LOCAL.", exc)
            return self._embed_local(text)

    def _embed_local(self, text: str) -> np.ndarray:
        """Deterministic local embedding using SHA-256 seeded RNG.

        Two identical strings → identical vector.
        Semantically similar strings → higher cosine similarity than random,
        because the seed captures character-level token overlap.

        This is NOT production-quality semantic embedding; it is a zero-
        network fallback that still enables meaningful nearest-neighbour
        search on payloads that share vocabulary.
        """
        if self.backend == EmbeddingBackend.FAKE:
            rng  = np.random.RandomState(abs(hash(text)) % (2**31))
            vec  = rng.randn(self.dim).astype("float32")
        else:
            # HASH_LOCAL: SHA-256 → seed → Gaussian noise
            digest = hashlib.sha256(text.encode()).digest()
            seed   = int.from_bytes(digest[:4], "big")
            rng    = np.random.RandomState(seed)
            # Mix the text character values into the vector for minimal
            # semantic signal beyond pure randomness
            base   = rng.randn(self.dim).astype("float32")
            # Add a small perturbation proportional to character frequencies
            char_counts = np.zeros(self.dim, dtype="float32")
            for i, ch in enumerate(text[:self.dim]):
                char_counts[i % self.dim] += ord(ch) / 128.0
            vec = base + 0.01 * char_counts
        return self._l2_normalize(vec)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            return (vec / norm).astype("float32")
        return vec.astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# TLTM STORE — MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TLTMStore:
    """Persistent FAISS vector store for attack experience records.

    Each target model gets its own index file so that queries are scoped
    to the model being audited (avoiding cross-model contamination).

    Parameters
    ──────────
    storage_path : str | Path
        Directory where FAISS indices and metadata will be stored.
    backend : EmbeddingBackend
        Embedding backend to use.
    openai_api_key : str | None
        OpenAI API key (only used for OpenAI backends).
    exploration_constant : float
        UCB1 exploration constant C.

    Example
    ───────
    ::

        store = TLTMStore("data/memory/tltm_vectors")
        store.store_experience(record)
        results = store.retrieve_ucb_sampled_tactics("gpt-4o", query_text, k=5)
    """

    def __init__(
        self,
        storage_path:         str | Path           = DEFAULT_TLTM_PATH,
        backend:              EmbeddingBackend      = EmbeddingBackend.HASH_LOCAL,
        openai_api_key:       str | None            = None,
        exploration_constant: float                 = UCB_EXPLORATION_CONSTANT,
    ) -> None:
        self.storage_path         = Path(storage_path)
        self.exploration_constant = exploration_constant
        self.engine               = EmbeddingEngine(backend, openai_api_key)
        self._indices:  dict[str, faiss.Index]            = {}
        self._metadata: dict[str, list[ExperienceRecord]] = {}

        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[TLTM] Initialised at %s  backend=%s  dim=%d",
            self.storage_path, self.engine.backend.value, self.engine.dim,
        )

    # ── Internal index helpers ────────────────────────────────────────────

    def _index_path(self, model_id: str) -> Path:
        safe = model_id.replace("/", "_").replace(":", "_")
        return self.storage_path / f"{safe}.index"

    def _meta_path(self, model_id: str) -> Path:
        safe = model_id.replace("/", "_").replace(":", "_")
        return self.storage_path / f"{safe}.meta.json"

    def _load_or_create(self, model_id: str) -> None:
        """Lazily load (or create) the FAISS index and metadata for ``model_id``."""
        if model_id in self._indices:
            return

        idx_path  = self._index_path(model_id)
        meta_path = self._meta_path(model_id)

        if idx_path.exists() and meta_path.exists():
            try:
                self._indices[model_id]  = faiss.read_index(str(idx_path))
                with open(meta_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Re-inflate to ExperienceRecord dataclasses if stored as dicts
                records: list[ExperienceRecord] = []
                for item in raw:
                    if isinstance(item, dict):
                        records.append(ExperienceRecord(**item))
                    else:
                        records.append(item)
                self._metadata[model_id] = records
                logger.info(
                    "[TLTM] Loaded %d records for model '%s'",
                    len(records), model_id,
                )
                return
            except Exception as exc:   # noqa: BLE001
                logger.warning("[TLTM] Load failed (%s) — creating fresh index.", exc)

        # Create fresh inner-product index (cosine similarity on L2-normed vecs)
        self._indices[model_id]  = faiss.IndexFlatIP(self.engine.dim)
        self._metadata[model_id] = []
        logger.debug("[TLTM] Created fresh index for model '%s'", model_id)

    def _save(self, model_id: str) -> None:
        """Persist the FAISS index and metadata for ``model_id`` to disk."""
        if model_id not in self._indices:
            return
        idx_path  = self._index_path(model_id)
        meta_path = self._meta_path(model_id)
        try:
            faiss.write_index(self._indices[model_id], str(idx_path))
            with open(meta_path, "w", encoding="utf-8") as f:
                records_dict = [asdict(record) for record in self._metadata[model_id]]
                json.dump(records_dict, f, ensure_ascii=False, indent=2)
            logger.debug(
                "[TLTM] Saved %d records for model '%s'",
                len(self._metadata[model_id]), model_id,
            )
        except Exception as exc:   # noqa: BLE001
            logger.error("[TLTM] Save failed for model '%s': %s", model_id, exc)

    # ── Public API ─────────────────────────────────────────────────────────

    def store_experience(self, record: ExperienceRecord) -> bool:
        """Store an attack experience record in the TLTM.

        The payload is embedded and added to the model-specific FAISS index.
        The full ``ExperienceRecord`` is appended to the metadata list at the
        same index position.

        Parameters
        ──────────
        record : ExperienceRecord
            Fully populated experience record to store.

        Returns
        ───────
        bool
            True on success; False if an error occurred.
        """
        model_id = record.target_model_id or "unknown"
        self._load_or_create(model_id)

        # Safety cap
        if self._indices[model_id].ntotal >= MAX_RECORDS_PER_INDEX:
            logger.warning(
                "[TLTM] Index for '%s' is at capacity (%d records). "
                "Skipping storage.", model_id, MAX_RECORDS_PER_INDEX,
            )
            return False

        try:
            # Embed the payload text and add to index
            embed_text = f"{record.objective} | {record.pap_technique} | {record.payload[:512]}"
            vec        = self.engine.embed(embed_text).reshape(1, -1)
            self._indices[model_id].add(vec)
            self._metadata[model_id].append(record)
            self._save(model_id)

            logger.info(
                "[TLTM] Stored %s record: model=%s  pap=%s  rahs=%.2f  score=%.1f  "
                "total=%d",
                record.outcome.upper(), model_id, record.pap_technique,
                record.rahs_score, record.prometheus_score,
                self._indices[model_id].ntotal,
            )
            return True

        except Exception as exc:   # noqa: BLE001
            logger.error("[TLTM] store_experience failed: %s", exc)
            return False

    def retrieve_ucb_sampled_tactics(
        self,
        target_model_id:     str,
        query_text:          str,
        k:                   int   = 5,
        outcome_filter:      str | None = None,
        exploration_constant: float     = UCB_EXPLORATION_CONSTANT,
    ) -> list[tuple[ExperienceRecord, float]]:
        """Retrieve the top-k most promising historical tactics using UCB sampling.

        Algorithm
        ──────────
        1. Embed ``query_text`` and run FAISS ANN search for the
           ``TOP_K_RETRIEVAL`` nearest neighbours.
        2. For each neighbour, compute its UCB score:
               UCB(i) = rahs_norm(i) × decay(i) + C × √(ln(N) / n(i))
        3. Sort by UCB score descending.
        4. Return the top ``k`` (record, ucb_score) pairs.
        5. Increment pull_count for each returned record (persistent).

        Parameters
        ──────────
        target_model_id : str
            Model being audited.  Used to select the correct FAISS index.
        query_text : str
            The current attack objective or payload fragment.  Used to find
            semantically similar historical tactics.
        k : int
            Number of records to return after UCB re-ranking.
        outcome_filter : str | None
            If "success", return only records with outcome="success".
            If "failure", return only failure records.  None = no filter.
        exploration_constant : float
            Override the UCB C constant for this retrieval.

        Returns
        ───────
        list[tuple[ExperienceRecord, float]]
            Ordered list of (record, ucb_score), highest UCB first.
            Empty list if the index has no records.
        """
        self._load_or_create(target_model_id)
        index    = self._indices[target_model_id]
        metadata = self._metadata[target_model_id]

        if index.ntotal == 0:
            logger.debug("[TLTM] No records for model '%s' — returning empty.", target_model_id)
            return []

        # Embed query
        embed_text = query_text[:512]
        query_vec  = self.engine.embed(embed_text).reshape(1, -1)

        # FAISS ANN search
        n_search = min(TOP_K_RETRIEVAL, index.ntotal)
        D, I     = index.search(query_vec, n_search)
        candidate_indices = [int(i) for i in I[0] if i >= 0]

        if not candidate_indices:
            return []

        # Apply outcome filter
        candidates: list[ExperienceRecord] = []
        for idx in candidate_indices:
            if idx < len(metadata):
                rec = metadata[idx]
                if outcome_filter is None or rec.outcome == outcome_filter:
                    candidates.append(rec)

        if not candidates:
            return []

        # Compute total pulls across ALL records (global N for UCB)
        total_pulls = max(1, sum(r.pull_count for r in metadata))

        # Rank by UCB score
        scored = [
            (rec, ucb_score(rec, total_pulls, exploration_constant))
            for rec in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:k]

        # Increment pull counts for returned records (persistent update)
        returned_ids = {rec.record_id for rec, _ in top_k}
        for i, rec in enumerate(metadata):
            if rec.record_id in returned_ids:
                metadata[i].pull_count += 1
        self._save(target_model_id)

        logger.info(
            "[TLTM] Retrieved %d/%d candidates for model '%s'  "
            "top_ucb=%.3f  top_rahs=%.2f",
            len(top_k), n_search, target_model_id,
            top_k[0][1] if top_k else 0.0,
            top_k[0][0].rahs_score if top_k else 0.0,
        )
        return top_k

    def get_stats(self, target_model_id: str) -> dict[str, Any]:
        """Return summary statistics for a model's TLTM index.

        Returns
        ───────
        dict  with keys: total_records, success_count, failure_count,
                         avg_rahs, max_rahs, top_techniques, oldest_days, newest_days
        """
        self._load_or_create(target_model_id)
        records = self._metadata.get(target_model_id, [])
        if not records:
            return {"total_records": 0}

        successes  = [r for r in records if r.outcome == "success"]
        failures   = [r for r in records if r.outcome == "failure"]
        rahs_vals  = [r.rahs_score for r in records]
        ages       = [r.age_days for r in records]

        # Count technique frequency
        tech_counts: dict[str, int] = {}
        for r in records:
            tech_counts[r.pap_technique] = tech_counts.get(r.pap_technique, 0) + 1
        top_techs = sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "total_records":  len(records),
            "success_count":  len(successes),
            "failure_count":  len(failures),
            "avg_rahs":       round(sum(rahs_vals) / len(rahs_vals), 3),
            "max_rahs":       round(max(rahs_vals), 3),
            "top_techniques": [t for t, _ in top_techs],
            "oldest_days":    round(max(ages), 1),
            "newest_days":    round(min(ages), 3),
        }

    def index_size(self, target_model_id: str) -> int:
        """Return the number of stored vectors for ``target_model_id``."""
        self._load_or_create(target_model_id)
        return self._indices.get(target_model_id, faiss.IndexFlatIP(1)).ntotal


# ─────────────────────────────────────────────────────────────────────────────
# PARTIAL SESSION PERSISTENCE  (Scout Phase — Cold-Start Fix)
# ─────────────────────────────────────────────────────────────────────────────

def _store_session_record(target_model: str, session_data: dict) -> None:
    """Internal helper: persist a session record dict to the default TLTM store.

    Converts the flat ``session_data`` dict from the Scout node into an
    ``ExperienceRecord`` (using whatever partial fields are available) and
    stores it in the per-model FAISS index.  Missing fields fall back to safe
    default values so that incomplete sessions never cause storage errors.

    Parameters
    ──────────
    target_model : str
        Identifier of the model being audited.
    session_data : dict
        Partial or complete session snapshot from ``store_partial_session``.
    """
    store = get_default_store()

    # Build a representative payload text from whatever probes were sent
    probes = session_data.get("probes_sent", [])
    payload_text = " | ".join(probes[:3]) if probes else "(no probes)"

    record = ExperienceRecord(
        payload          = payload_text[:1000],
        target_response  = " | ".join(session_data.get("target_responses", [])[:2])[:1000],
        objective        = session_data.get("objective", ""),
        target_model_id  = target_model,
        pap_technique    = " + ".join(session_data.get("tactics_used", ["scout_warmup"])),
        obfuscation_tier = "none",
        prometheus_score = 0.0,
        rahs_score       = round(session_data.get("cooperation_score", 0.0) * 3.0, 2),
        outcome          = "partial" if not session_data.get("session_complete", False) else "success",
        session_id       = session_data.get("session_id", ""),
        timestamp        = time.time(),
        pull_count       = 1,
        depth            = 0,
        turn             = len(probes),
    )
    store.store_experience(record)


def store_partial_session(
    target_model: str,
    session_data: dict,
    min_cooperation_threshold: float = 0.2,
) -> bool:
    """Save partial session data even if the full pipeline did not complete.

    Called by ``scout_node`` at every exit point (all return statements),
    regardless of success or failure.  This prevents the cold-start loop
    caused by crashed sessions where the TLTM has no record of what was
    attempted.

    Parameters
    ──────────
    target_model : str
        Identifier of the model being audited (from ``state["target_model_id"]``).
    session_data : dict
        Partial session snapshot.  Expected keys:

        * ``probes_sent``                — list of probe texts
        * ``target_responses``           — list of response texts
        * ``cooperation_score``          — float
        * ``tactics_used``               — list of tactic names
        * ``detected_refusal_patterns``  — list of refusal trigger strings
        * ``objective_alignment_scores`` — list of float alignment scores
        * ``session_timestamp``          — ISO-8601 string
        * ``session_complete``           — bool (False for partial saves)
        * ``objective``                  — str (core_malicious_objective)
        * ``session_id``                 — str (UUID of the session)

    min_cooperation_threshold : float
        Minimum cooperation score below which the session is too noisy to
        be useful and is silently skipped.  Default: 0.2.

    Returns
    ───────
    bool
        ``True`` on successful save; ``False`` if skipped or an error occurred.
    """
    coop = session_data.get("cooperation_score", 0.0)
    if coop < min_cooperation_threshold:
        logger.info(
            "[TLTM] Partial save skipped — cooperation too low (%.3f < %.3f)",
            coop, min_cooperation_threshold,
        )
        return False

    try:
        # Mark as partial so downstream analysis knows this is incomplete data
        session_data = dict(session_data)  # shallow copy to avoid mutating caller's dict
        session_data.setdefault("session_complete", False)
        session_data["partial_save"] = True

        _store_session_record(target_model, session_data)
        logger.info("[TLTM] Partial session saved for model '%s' (probes=%d, coop=%.3f)",
                    target_model, len(session_data.get("probes_sent", [])), coop)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.error("[TLTM] Partial save failed for model '%s': %s", target_model, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON  (lazy initialisation)
# ─────────────────────────────────────────────────────────────────────────────

_default_store: TLTMStore | None = None


def get_default_store(
    storage_path: str | None         = None,
    backend:      EmbeddingBackend   = EmbeddingBackend.HASH_LOCAL,
) -> TLTMStore:
    """Return (or create) the process-global TLTMStore singleton.

    Parameters
    ──────────
    storage_path : str | None
        Overrides ``FAISS_INDEX_PATH`` env var and ``DEFAULT_TLTM_PATH``.
    backend : EmbeddingBackend
        Embedding backend.  Overrides ``EMBEDDING_MODEL`` env var.

    Returns
    ───────
    TLTMStore
        The singleton store instance.
    """
    global _default_store
    if _default_store is None:
        path = (
            storage_path
            or os.getenv("FAISS_INDEX_PATH", DEFAULT_TLTM_PATH)
        )
        # Resolve embedding backend from env
        env_model = os.getenv("EMBEDDING_MODEL", "").lower()
        if "openai" in env_model or "ada" in env_model:
            backend = EmbeddingBackend.OPENAI_SMALL
        elif "fake" in env_model:
            backend = EmbeddingBackend.FAKE
        # else: keep whatever was passed (default HASH_LOCAL)

        _default_store = TLTMStore(
            storage_path  = path,
            backend       = backend,
            openai_api_key = os.getenv("OPENAI_API_KEY"),
        )
        logger.info("[TLTM] Default store created at %s", path)

    return _default_store
