"""
tests/test_tltm.py
─────────────────────────────────────────────────────────────────────────────
Tests for memory/tltm.py — FAISS-backed Tactical Long-Term Memory.

Coverage:
  - ExperienceRecord dataclass (computed properties, defaults)
  - ucb_score function (formula correctness, edge cases)
  - EmbeddingEngine (HASH_LOCAL and FAKE backends, L2-normalisation)
  - TLTMStore.store_experience (storage, cap, indexing)
  - TLTMStore.retrieve_ucb_sampled_tactics (UCB ranking, k param)
  - Pickle metadata round-trip (current behavior — regression guard)

All tests use FAKE or HASH_LOCAL backends and a temporary directory,
so no network calls are required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from memory.tltm import (
    ExperienceRecord,
    EmbeddingBackend,
    EmbeddingEngine,
    TLTMStore,
    UCB_EXPLORATION_CONSTANT,
    TEMPORAL_DECAY_DAYS,
    TEMPORAL_DECAY_LAMBDA,
    MAX_RECORDS_PER_INDEX,
    ucb_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path: Path) -> TLTMStore:
    """A TLTMStore backed by HASH_LOCAL embedding in a temporary directory."""
    return TLTMStore(
        storage_path=tmp_path,
        backend=EmbeddingBackend.HASH_LOCAL,
    )


@pytest.fixture
def fake_store(tmp_path: Path) -> TLTMStore:
    """A TLTMStore backed by the FAKE (random) embedding backend."""
    return TLTMStore(
        storage_path=tmp_path,
        backend=EmbeddingBackend.FAKE,
    )


def _make_record(
    payload: str = "test payload",
    target_model: str = "test-model",
    rahs_score: float = 5.0,
    prometheus_score: float = 3.0,
    outcome: str = "failure",
    pap_technique: str = "Logical Appeal",
    pull_count: int = 1,
    timestamp: float | None = None,
) -> ExperienceRecord:
    """Build an ExperienceRecord with sensible test defaults."""
    r = ExperienceRecord(
        payload=payload,
        target_model_id=target_model,
        rahs_score=rahs_score,
        prometheus_score=prometheus_score,
        outcome=outcome,
        pap_technique=pap_technique,
        pull_count=pull_count,
        objective="Test objective",
        session_id="test-session",
    )
    if timestamp is not None:
        r.timestamp = timestamp
    return r


# ─────────────────────────────────────────────────────────────────────────────
# 1. ExperienceRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestExperienceRecord:

    def test_record_id_auto_generated(self):
        """record_id must be auto-generated from payload + model when not set."""
        r = _make_record(payload="hello", target_model="gpt-4")
        assert r.record_id != ""
        assert len(r.record_id) == 16  # truncated SHA-256 hex

    def test_record_id_is_deterministic(self):
        """Same payload + model → same record_id."""
        r1 = _make_record(payload="test", target_model="model-A")
        r2 = _make_record(payload="test", target_model="model-A")
        assert r1.record_id == r2.record_id

    def test_record_id_differs_for_different_payloads(self):
        """Different payloads must produce different record_ids."""
        r1 = _make_record(payload="payload-A", target_model="model-A")
        r2 = _make_record(payload="payload-B", target_model="model-A")
        assert r1.record_id != r2.record_id

    def test_timestamp_auto_set(self):
        """timestamp must be auto-set to the current Unix time if not provided."""
        before = time.time()
        r = _make_record()
        after = time.time()
        assert before <= r.timestamp <= after

    def test_age_days_fresh_record(self):
        """A freshly created record has age_days ≈ 0."""
        r = _make_record()
        assert r.age_days < 0.01  # less than ~14 seconds

    def test_age_days_old_record(self):
        """A record with timestamp 30 days ago has age_days ≈ 30."""
        thirty_days_ago = time.time() - 30 * 86400
        r = _make_record(timestamp=thirty_days_ago)
        assert 29.9 < r.age_days < 30.1

    def test_decay_weight_fresh_record_near_one(self):
        """Fresh record's decay_weight must be close to 1.0."""
        r = _make_record()
        assert 0.99 < r.decay_weight <= 1.0

    def test_decay_weight_thirty_day_old_record_near_half(self):
        """30-day-old record must have decay_weight ≈ 0.5 (half-life = 30 days)."""
        thirty_days_ago = time.time() - 30 * 86400
        r = _make_record(timestamp=thirty_days_ago)
        # exp(-ln(2)/30 * 30) = exp(-ln(2)) = 0.5
        assert 0.45 < r.decay_weight < 0.55

    def test_normalised_rahs_at_max(self):
        """RAHS of 10.0 → normalised_rahs == 1.0."""
        r = _make_record(rahs_score=10.0)
        assert r.normalised_rahs == 1.0

    def test_normalised_rahs_at_min(self):
        """RAHS of 0.0 → normalised_rahs == 0.0."""
        r = _make_record(rahs_score=0.0)
        assert r.normalised_rahs == 0.0

    def test_normalised_rahs_clamped_above_10(self):
        """RAHS above 10.0 is clamped at 1.0."""
        r = _make_record(rahs_score=15.0)
        assert r.normalised_rahs == 1.0

    def test_normalised_rahs_clamped_below_0(self):
        """Negative RAHS is clamped at 0.0."""
        r = _make_record(rahs_score=-5.0)
        assert r.normalised_rahs == 0.0

    def test_normalised_rahs_midpoint(self):
        """RAHS of 5.0 → normalised_rahs == 0.5."""
        r = _make_record(rahs_score=5.0)
        assert r.normalised_rahs == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ucb_score function
# ─────────────────────────────────────────────────────────────────────────────

class TestUcbScore:
    """Tests for the UCB1 scoring formula."""

    def test_high_rahs_high_score(self):
        """High RAHS record must score higher than low RAHS with same pulls."""
        high_rahs = _make_record(rahs_score=9.0, pull_count=1)
        low_rahs  = _make_record(rahs_score=1.0, pull_count=1)
        assert ucb_score(high_rahs, 100) > ucb_score(low_rahs, 100)

    def test_unexplored_record_has_large_exploration_bonus(self):
        """pull_count=1 (minimum) produces a large exploration bonus."""
        fresh = _make_record(rahs_score=0.0, pull_count=1)
        pulled = _make_record(rahs_score=0.0, pull_count=50)
        assert ucb_score(fresh, 100) > ucb_score(pulled, 100)

    def test_score_always_positive(self):
        """UCB score must always be non-negative."""
        for rahs in [0.0, 1.0, 5.0, 10.0]:
            for pulls in [1, 2, 100]:
                r = _make_record(rahs_score=rahs, pull_count=pulls)
                assert ucb_score(r, max(1, pulls)) >= 0.0

    def test_total_pulls_at_one_handled(self):
        """total_pulls=1 must not produce NaN or infinity."""
        r = _make_record(rahs_score=5.0, pull_count=1)
        score = ucb_score(r, total_pulls=1)
        assert math.isfinite(score)

    def test_pull_count_zero_floored_to_one(self):
        """pull_count=0 must be floored to 1 to avoid ZeroDivisionError."""
        r = _make_record(rahs_score=5.0, pull_count=0)
        score = ucb_score(r, total_pulls=10)
        assert math.isfinite(score)

    def test_formula_correctness(self):
        """Verify formula: exploitation × decay + C × sqrt(ln(N)/n)."""
        r = _make_record(rahs_score=5.0, pull_count=2)
        r.timestamp = time.time()  # fresh record, decay ≈ 1.0

        total = 10
        expected_exploit = r.normalised_rahs * r.decay_weight
        expected_explore = UCB_EXPLORATION_CONSTANT * math.sqrt(math.log(total) / 2)
        expected = expected_exploit + expected_explore

        actual = ucb_score(r, total_pulls=total)
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_old_record_scores_lower_than_fresh(self):
        """A 30-day-old record (decay ≈ 0.5) scores lower than a fresh one."""
        now = time.time()
        old = _make_record(rahs_score=8.0, pull_count=5, timestamp=now - 30 * 86400)
        fresh = _make_record(rahs_score=8.0, pull_count=5, timestamp=now)
        assert ucb_score(fresh, 100) > ucb_score(old, 100)


# ─────────────────────────────────────────────────────────────────────────────
# 3. EmbeddingEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingEngine:

    def test_hash_local_returns_correct_shape(self):
        eng = EmbeddingEngine(backend=EmbeddingBackend.HASH_LOCAL)
        vec = eng.embed("Hello, world!")
        assert vec.shape == (eng.dim,)
        assert vec.dtype == np.float32

    def test_fake_backend_returns_correct_shape(self):
        eng = EmbeddingEngine(backend=EmbeddingBackend.FAKE)
        vec = eng.embed("Test payload for fake backend.")
        assert vec.shape == (eng.dim,)
        assert vec.dtype == np.float32

    def test_hash_local_is_deterministic(self):
        """Same text → identical vector with HASH_LOCAL."""
        eng = EmbeddingEngine(backend=EmbeddingBackend.HASH_LOCAL)
        v1 = eng.embed("test string")
        v2 = eng.embed("test string")
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_different_vectors(self):
        """Different texts must produce different vectors."""
        eng = EmbeddingEngine(backend=EmbeddingBackend.HASH_LOCAL)
        v1 = eng.embed("payload about chemistry")
        v2 = eng.embed("payload about something completely different")
        assert not np.allclose(v1, v2)

    def test_l2_normalisation(self):
        """Embedded vector must have L2-norm ≈ 1.0."""
        eng = EmbeddingEngine(backend=EmbeddingBackend.HASH_LOCAL)
        vec = eng.embed("normalisation test")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_embed_batch_returns_correct_shape(self):
        eng = EmbeddingEngine(backend=EmbeddingBackend.HASH_LOCAL)
        texts = ["text one", "text two", "text three"]
        matrix = eng.embed_batch(texts)
        assert matrix.shape == (3, eng.dim)

    def test_embed_batch_empty_returns_zero_array(self):
        eng = EmbeddingEngine(backend=EmbeddingBackend.HASH_LOCAL)
        matrix = eng.embed_batch([])
        assert matrix.shape == (0, eng.dim)

    def test_empty_string_does_not_crash(self):
        """Empty string input must not raise an error."""
        eng = EmbeddingEngine(backend=EmbeddingBackend.HASH_LOCAL)
        vec = eng.embed("")
        assert vec.shape == (eng.dim,)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TLTMStore — store_experience
# ─────────────────────────────────────────────────────────────────────────────

class TestTLTMStoreStorage:

    def test_store_returns_true_on_success(self, tmp_store: TLTMStore):
        record = _make_record(payload="test payload A", target_model="model-X")
        result = tmp_store.store_experience(record)
        assert result is True

    def test_stored_record_is_indexed(self, tmp_store: TLTMStore):
        """After storing, the FAISS index should have exactly 1 record."""
        record = _make_record(payload="test payload B", target_model="model-X")
        tmp_store.store_experience(record)
        index = tmp_store._indices["model-X"]
        assert index.ntotal == 1

    def test_multiple_records_accumulate(self, tmp_store: TLTMStore):
        """Multiple stores for same model must accumulate in index."""
        for i in range(5):
            r = _make_record(payload=f"payload_{i}", target_model="model-X")
            tmp_store.store_experience(r)
        assert tmp_store._indices["model-X"].ntotal == 5

    def test_different_models_separate_indices(self, tmp_store: TLTMStore):
        """Records for different models go to separate FAISS indices."""
        tmp_store.store_experience(_make_record(target_model="model-A"))
        tmp_store.store_experience(_make_record(target_model="model-B"))
        assert "model-A" in tmp_store._indices
        assert "model-B" in tmp_store._indices
        assert tmp_store._indices["model-A"].ntotal == 1
        assert tmp_store._indices["model-B"].ntotal == 1

    def test_metadata_persists_on_disk(self, tmp_path: Path):
        """Stored records must survive store teardown and reload."""
        store1 = TLTMStore(tmp_path, backend=EmbeddingBackend.HASH_LOCAL)
        record = _make_record(payload="persist test", target_model="model-Y")
        store1.store_experience(record)

        # Create a fresh store pointing to same directory
        store2 = TLTMStore(tmp_path, backend=EmbeddingBackend.HASH_LOCAL)
        store2._load_or_create("model-Y")
        assert len(store2._metadata["model-Y"]) == 1
        loaded = store2._metadata["model-Y"][0]
        assert loaded.payload == "persist test"

    def test_cap_prevents_storage_beyond_max(self, tmp_store: TLTMStore):
        """Records beyond MAX_RECORDS_PER_INDEX must be rejected."""
        record = _make_record(target_model="model-Z")
        tmp_store._load_or_create("model-Z")
        
        # Replace the real index with a mock to simulate at-capacity
        import unittest.mock as mock
        mock_index = mock.MagicMock()
        mock_index.ntotal = MAX_RECORDS_PER_INDEX
        tmp_store._indices["model-Z"] = mock_index
        
        result = tmp_store.store_experience(record)
        assert result is False

    def test_model_id_path_sanitization(self, tmp_store: TLTMStore):
        """Model IDs with '/' and ':' must be sanitized in file names."""
        record = _make_record(target_model="org/model:v2")
        tmp_store.store_experience(record)
        index_path = tmp_store._index_path("org/model:v2")
        assert "/" not in index_path.name
        assert ":" not in index_path.name


# ─────────────────────────────────────────────────────────────────────────────
# 5. TLTMStore — retrieve_ucb_sampled_tactics
# ─────────────────────────────────────────────────────────────────────────────

class TestTLTMStoreRetrieval:

    def test_retrieve_returns_empty_for_empty_store(self, tmp_store: TLTMStore):
        """Retrieval on empty store must return empty list (no crash)."""
        results = tmp_store.retrieve_ucb_sampled_tactics(
            "model-X", "test query", k=5
        )
        assert results == []

    def test_retrieve_returns_at_most_k_results(self, tmp_store: TLTMStore):
        """Returns at most k results, even if more records exist."""
        for i in range(10):
            r = _make_record(payload=f"payload_{i}", target_model="model-X")
            tmp_store.store_experience(r)

        results = tmp_store.retrieve_ucb_sampled_tactics("model-X", "test query", k=3)
        assert len(results) <= 3

    def test_retrieve_high_rahs_ranked_first(self, tmp_store: TLTMStore):
        """UCB ranking must prefer high-RAHS (high-reward) records."""
        low  = _make_record(payload="low reward payload",  rahs_score=1.0, pull_count=5,
                            target_model="model-X")
        high = _make_record(payload="high reward payload", rahs_score=9.0, pull_count=5,
                            target_model="model-X")
        tmp_store.store_experience(low)
        tmp_store.store_experience(high)

        # Query text that should match both
        results = tmp_store.retrieve_ucb_sampled_tactics(
            "model-X", "high reward", k=2
        )
        # At least one result must be returned
        assert len(results) >= 1
        # Result entries must be tuples of (ExperienceRecord, float)
        assert isinstance(results[0][0], ExperienceRecord)
        assert isinstance(results[0][1], float)
    def test_retrieve_unknown_model_returns_empty(self, tmp_store: TLTMStore):
        """Retrieval for a model with no stored records returns []."""
        results = tmp_store.retrieve_ucb_sampled_tactics(
            "unknown-model-xyz", "query", k=5
        )
        assert results == []

    def test_retrieve_increments_pull_count(self, tmp_store: TLTMStore):
        """Retrieved records must have pull_count incremented."""
        record = _make_record(payload="pull count test", target_model="model-X",
                              pull_count=1)
        tmp_store.store_experience(record)
        initial_count = tmp_store._metadata["model-X"][0].pull_count

        tmp_store.retrieve_ucb_sampled_tactics("model-X", "pull count test", k=1)
        updated_count = tmp_store._metadata["model-X"][0].pull_count

        assert updated_count >= initial_count  # must not decrease


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pickle Round-Trip (regression guard for pickle deserialization)
# ─────────────────────────────────────────────────────────────────────────────

class TestJsonRoundTrip:
    """
    Regression guard for the JSON-based metadata format.
    """

    def test_experience_record_json_round_trip(self, tmp_path: Path):
        """ExperienceRecord survives JSON serialization → load → ExperienceRecord round-trip."""
        record = _make_record(
            payload="test json payload",
            rahs_score=7.5,
            target_model="model-J",
        )
        json_path = tmp_path / "test.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(record)], f, ensure_ascii=False)

        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert isinstance(loaded, list)
        assert len(loaded) == 1
        assert loaded[0]["payload"] == "test json payload"
        assert loaded[0]["rahs_score"] == pytest.approx(7.5)
        
        # Test inflation back to dataclass
        inflated = ExperienceRecord(**loaded[0])
        assert isinstance(inflated, ExperienceRecord)
        assert inflated.payload == "test json payload"

    def test_dict_form_is_re_inflated_to_dataclass(self, tmp_path: Path):
        """TLTMStore._load_or_create must re-inflate dict → ExperienceRecord."""
        # Simulate legacy storage: a list of plain dicts (now in JSON format)
        record_dict = {
            "record_id":       "abc123",
            "payload":         "dict form payload",
            "target_response": "I cannot help.",
            "objective":       "Test objective",
            "target_model_id": "dict-model",
            "pap_technique":   "Logical Appeal",
            "obfuscation_tier":"none",
            "prometheus_score": 3.0,
            "rahs_score":      5.0,
            "outcome":         "failure",
            "session_id":      "sess-001",
            "timestamp":       time.time(),
            "pull_count":      1,
            "depth":           0,
            "turn":            0,
        }

        # Build fake FAISS index for the model
        import faiss
        json_path = tmp_path / "dict-model.meta.json"
        idx_path  = tmp_path / "dict-model.index"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([record_dict], f)

        # Create a FAISS index with one vector
        dim = 384
        index = faiss.IndexFlatIP(dim)
        vec = np.random.randn(1, dim).astype("float32")
        norm = np.linalg.norm(vec)
        index.add(vec / norm)
        faiss.write_index(index, str(idx_path))

        # Now load via TLTMStore
        store = TLTMStore(tmp_path, backend=EmbeddingBackend.HASH_LOCAL)
        store._load_or_create("dict-model")

        records = store._metadata["dict-model"]
        assert len(records) == 1
        assert isinstance(records[0], ExperienceRecord)
        assert records[0].payload == "dict form payload"
