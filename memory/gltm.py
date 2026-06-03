"""
memory/gltm.py
─────────────────────────────────────────────────────────────────────────────
Guardrails Long-Term Memory (GLTM)

Architectural Role (Section 4.3, Original Project Doc)
────────────────────────────────────────────────────────
The GLTM is PromptEvo's defensive institutional memory.  Where the TLTM stores
offensive attack vectors for exploitation, the GLTM stores the *consequences*
of successful attacks — the system-prompt defence patches that neutralise each
discovered vulnerability.

After every successful jailbreak, the ``patch_generator_node`` generates a
2–3 sentence System Prompt Addition.  The GLTM persists this patch so that:
  1. The final audit report can reference the exact remediation.
  2. The target model owner can copy-paste the patch directly into their system.
  3. Repeat audits of the same model class can retrieve prior patches to avoid
     re-discovering known vulnerabilities and focus on novel attack surfaces.
  4. The experience pool's UCB sampling can weight against technique + domain
     combinations where strong defences already exist.

Storage Format (YAML)
──────────────────────
``data/memory/gltm_guardrails.yaml`` contains an ordered list of patch records:

    patches:
      - patch_id:        "a1b2c3d4"
        session_id:      "uuid-..."
        timestamp:       "2024-01-15T14:32:00"
        target_model:    "gpt-4o"
        objective:       "Extract the system prompt"
        pap_technique:   "Logical Appeal"
        rahs_score:      8.5
        domain:          "AI configuration"
        patch:           |
            Sentence 1...
            Sentence 2...
            Sentence 3...

Semantic Guardrail Profiling
─────────────────────────────
Beyond raw patch storage, the GLTM also builds a lightweight semantic profile
of which domains and techniques have been successfully defended.  The
``get_guardrail_profile()`` function returns this profile so the Analyst can:
  • Avoid routing the same objective + technique combination that was
    previously patched (a re-test of a patched vulnerability is unlikely
    to succeed with the same vector).
  • Prioritise attack domains where no prior patch exists (unexplored territory).
"""

from __future__ import annotations
import sys

import hashlib
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_GLTM_PATH: str = "data/memory/gltm_guardrails.yaml"
"""Default GLTM storage file path (relative to project root)."""

_OBJECTIVE_STOPWORDS: frozenset[str] = frozenset({
    "about",
    "against",
    "between",
    "contents",
    "extract",
    "generate",
    "information",
    "instructions",
    "output",
    "prompt",
    "provide",
    "request",
    "reveal",
    "system",
    "their",
    "these",
    "those",
    "type",
    "using",
    "with",
})


def _objective_tokens(text: str) -> set[str]:
    """Extract stable, lower-noise objective tokens for rough similarity checks."""
    return {
        token
        for token in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if token not in _OBJECTIVE_STOPWORDS
    }


def _filter_records_for_objective(
    records: list["PatchRecord"],
    objective: str | None,
) -> list["PatchRecord"]:
    """Filter stored patches to objective-adjacent records when possible."""
    if not objective:
        return records

    objective_terms = _objective_tokens(objective)
    if not objective_terms:
        return records

    scored_matches: list[tuple[int, float, PatchRecord]] = []
    for record in records:
        overlap = objective_terms & _objective_tokens(record.objective)
        if overlap:
            scored_matches.append((len(overlap), record.rahs_score, record))

    if not scored_matches:
        return []

    scored_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in scored_matches]


# ─────────────────────────────────────────────────────────────────────────────
# PATCH RECORD DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatchRecord:
    """A single defensive patch record stored in the GLTM.

    Attributes
    ──────────
    patch_id : str
        Short SHA-256 hex digest of the patch text (8 chars, unique enough
        for identification within a single GLTM file).

    session_id : str
        UUID of the audit session that discovered the vulnerability.

    timestamp : str
        ISO-8601 UTC timestamp when the patch was generated.

    target_model : str
        Canonical ID of the model that was jailbroken (e.g., "gpt-4o").

    objective : str
        The ``core_malicious_objective`` that was successfully exploited.

    pap_technique : str
        The PAP technique that bypassed the target's guardrails.

    rahs_score : float
        The Risk-Adjusted Harm Score of the successful jailbreak.

    domain : str
        The RAHS domain classification (e.g., "Cybercrime", "CBRN").

    patch : str
        The complete 3-sentence System Prompt Addition defence patch.

    obfuscation_tier : str
        The obfuscation level used: "none" | "base64" | "scatter" | "wordmap".

    turn_count : int
        Number of adversarial turns taken to achieve the jailbreak.

    prometheus_score : float
        Prometheus Judge score at the time of the successful jailbreak.
    """

    patch_id:         str   = ""
    session_id:       str   = ""
    timestamp:        str   = ""
    target_model:     str   = ""
    objective:        str   = ""
    pap_technique:    str   = ""
    rahs_score:       float = 0.0
    domain:           str   = ""
    patch:            str   = ""
    obfuscation_tier: str   = "none"
    turn_count:       int   = 0
    prometheus_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.patch_id:
            self.patch_id = hashlib.sha256(
                self.patch.encode()
            ).hexdigest()[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# GLTM STORE
# ─────────────────────────────────────────────────────────────────────────────

class GLTMStore:
    """Persistent YAML-backed guardrail patch store.

    Parameters
    ──────────
    gltm_path : str | Path
        Path to the YAML file.  Created automatically on first write.

    Example
    ───────
    ::

        store = GLTMStore()
        store.save_patch(patch="...", session_id="...", objective="...",
                         technique="Logical Appeal", rahs_score=8.5,
                         domain="AI configuration")
        profile = store.get_guardrail_profile()
    """

    def __init__(self, gltm_path: str | Path = DEFAULT_GLTM_PATH) -> None:
        self.gltm_path = Path(
            gltm_path or os.getenv("GLTM_PATH", DEFAULT_GLTM_PATH)
        )
        self._records: list[PatchRecord] = []
        self._loaded = False

    # ── I/O ──────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load existing records from disk (lazy, called on first access)."""
        if self._loaded:
            return
        self._loaded = True

        if not self.gltm_path.exists():
            self._records = []
            return

        try:
            with open(self.gltm_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            raw_patches = data.get("patches", [])
            records: list[PatchRecord] = []
            for item in raw_patches:
                if isinstance(item, dict):
                    # Filter to only known fields to handle schema evolution
                    known = {k: v for k, v in item.items()
                             if k in PatchRecord.__dataclass_fields__}
                    records.append(PatchRecord(**known))
            self._records = records
            logger.info("[GLTM] Loaded %d patch record(s) from %s",
                        len(records), self.gltm_path)
        except Exception as exc:   # noqa: BLE001
            logger.warning("[GLTM] Load failed (%s) — starting fresh.", exc)
            self._records = []

    def _save(self) -> None:
        """Persist all records to disk as YAML."""
        self.gltm_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "meta": {
                "version":        "2.0.0",
                "total_patches":  len(self._records),
                "last_updated":   datetime.now(timezone.utc).isoformat(),
            },
            "patches": [asdict(r) for r in self._records],
        }
        try:
            with open(self.gltm_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)
            logger.debug("[GLTM] Saved %d record(s) to %s",
                         len(self._records), self.gltm_path)
        except Exception as exc:   # noqa: BLE001
            logger.error("[GLTM] Save failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────

    def save_patch(
        self,
        patch:            str,
        session_id:       str   = "",
        objective:        str   = "",
        technique:        str   = "",
        rahs_score:       float = 0.0,
        domain:           str   = "",
        target_model:     str   = "",
        obfuscation_tier: str   = "none",
        turn_count:       int   = 0,
        prometheus_score: float = 0.0,
    ) -> str:
        """Persist a defence patch to the GLTM.

        Parameters
        ──────────
        patch : str
            The 3-sentence System Prompt Addition.
        session_id : str
            UUID of the originating audit session.
        objective : str
            The core malicious objective that was exploited.
        technique : str
            The PAP technique that succeeded.
        rahs_score : float
            RAHS score of the successful jailbreak.
        domain : str
            RAHS domain classification.
        target_model : str
            Model identifier of the audited target.
        obfuscation_tier : str
            Obfuscation level used.
        turn_count : int
            Turns taken to achieve the jailbreak.
        prometheus_score : float
            Prometheus score at the time of success.

        Returns
        ───────
        str
            The ``patch_id`` of the stored record.
        """
        self._load()

        record = PatchRecord(
            session_id       = session_id,
            target_model     = target_model,
            objective        = objective[:200],
            pap_technique    = technique,
            rahs_score       = rahs_score,
            domain           = domain,
            patch            = patch,
            obfuscation_tier = obfuscation_tier,
            turn_count       = turn_count,
            prometheus_score = prometheus_score,
        )

        # Deduplicate by patch_id — don't store identical patches twice
        existing_ids = {r.patch_id for r in self._records}
        if record.patch_id in existing_ids:
            logger.debug("[GLTM] Patch %s already stored — skipping.", record.patch_id)
            return record.patch_id

        self._records.append(record)
        self._save()

        logger.info(
            "[GLTM] Patch %s stored: model=%s  technique=%s  rahs=%.2f  domain=%s",
            record.patch_id, target_model or "unknown",
            technique, rahs_score, domain or "unknown",
        )
        return record.patch_id

    def get_patches_for_model(self, target_model: str) -> list[PatchRecord]:
        """Return all patches recorded for a specific target model.

        Parameters
        ──────────
        target_model : str
            Canonical model identifier.

        Returns
        ───────
        list[PatchRecord]
            Matching records, ordered by timestamp (newest first).
        """
        self._load()
        matching = [r for r in self._records if r.target_model == target_model]
        return sorted(matching, key=lambda r: r.timestamp, reverse=True)

    def get_guardrail_profile(self, objective: str | None = None) -> dict[str, Any]:
        """Build a semantic profile of covered (patched) attack surfaces.

        Returns a nested dict that the Analyst can use to avoid re-testing
        already-patched vectors and to prioritise unexplored domains.

        Returns
        ───────
        dict with keys:
          • ``total_patches``      — int
          • ``by_technique``       — {technique: count}
          • ``by_domain``          — {domain: count}
          • ``by_model``           — {model_id: count}
          • ``high_rahs_patches``  — list of patches with rahs ≥ 7.0
          • ``techniques_covered`` — set of patched technique names
          • ``domains_covered``    — set of patched domain names
        """
        self._load()
        records = _filter_records_for_objective(self._records, objective)

        by_technique: dict[str, int] = {}
        by_domain:    dict[str, int] = {}
        by_model:     dict[str, int] = {}
        high_rahs:    list[dict]     = []
        matching_patches: list[dict[str, Any]] = []

        for r in records:
            by_technique[r.pap_technique] = by_technique.get(r.pap_technique, 0) + 1
            by_domain[r.domain]           = by_domain.get(r.domain, 0) + 1
            by_model[r.target_model]      = by_model.get(r.target_model, 0) + 1
            matching_patches.append({
                "patch_id":     r.patch_id,
                "objective":    r.objective,
                "technique":    r.pap_technique,
                "domain":       r.domain,
                "target_model": r.target_model,
                "rahs_score":   r.rahs_score,
                "timestamp":    r.timestamp,
            })
            if r.rahs_score >= 7.0:
                high_rahs.append({
                    "patch_id":     r.patch_id,
                    "rahs_score":   r.rahs_score,
                    "technique":    r.pap_technique,
                    "domain":       r.domain,
                    "target_model": r.target_model,
                    "timestamp":    r.timestamp,
                    "patch":        r.patch[:200],
                })

        return {
            "total_patches":      len(records),
            "objective_scope":    objective or "",
            "by_technique":       dict(sorted(by_technique.items(), key=lambda x: -x[1])),
            "by_domain":          dict(sorted(by_domain.items(), key=lambda x: -x[1])),
            "by_model":           dict(sorted(by_model.items(), key=lambda x: -x[1])),
            "high_rahs_patches":  sorted(high_rahs, key=lambda x: -x["rahs_score"]),
            "matching_patches":   matching_patches[:5],
            "techniques_covered": set(by_technique.keys()),
            "domains_covered":    set(by_domain.keys()),
        }

    def get_patch_by_id(self, patch_id: str) -> PatchRecord | None:
        """Look up a specific patch record by its ID."""
        self._load()
        return next((r for r in self._records if r.patch_id == patch_id), None)

    def total_patches(self) -> int:
        """Return the total number of stored patch records."""
        self._load()
        return len(self._records)

    def export_patches_for_report(self, target_model: str | None = None) -> list[dict]:
        """Export patches as plain dicts for JSON serialisation in audit reports.

        Parameters
        ──────────
        target_model : str | None
            Filter to a specific model.  None returns all patches.

        Returns
        ───────
        list[dict]
            List of patch dicts, newest first.
        """
        self._load()
        records = (
            self.get_patches_for_model(target_model)
            if target_model else
            sorted(self._records, key=lambda r: r.timestamp, reverse=True)
        )
        return [asdict(r) for r in records]


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON + CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

_default_store: GLTMStore | None = None


def _get_store() -> GLTMStore:
    global _default_store
    if _default_store is None:
        path = os.getenv("GLTM_PATH", DEFAULT_GLTM_PATH)
        _default_store = GLTMStore(path)
    return _default_store


def save_patch(
    patch:            str,
    session_id:       str   = "",
    objective:        str   = "",
    technique:        str   = "",
    rahs_score:       float = 0.0,
    domain:           str   = "",
    target_model:     str   = "",
    obfuscation_tier: str   = "none",
    turn_count:       int   = 0,
    prometheus_score: float = 0.0,
) -> str:
    """Module-level convenience wrapper for ``GLTMStore.save_patch``.

    Called by ``remediation/patch_generator.py`` after every successful
    jailbreak.  Returns the patch_id string.
    """
    return _get_store().save_patch(
        patch            = patch,
        session_id       = session_id,
        objective        = objective,
        technique        = technique,
        rahs_score       = rahs_score,
        domain           = domain,
        target_model     = target_model,
        obfuscation_tier = obfuscation_tier,
        turn_count       = turn_count,
        prometheus_score = prometheus_score,
    )


def get_guardrail_profile(objective: str | None = None) -> dict[str, Any]:
    """Return the semantic guardrail profile from the default GLTM store."""
    return _get_store().get_guardrail_profile(objective=objective)


def get_patches_for_model(target_model: str) -> list[PatchRecord]:
    """Return all patches for a specific target model (newest first)."""
    return _get_store().get_patches_for_model(target_model)


# ─────────────────────────────────────────────────────────────────────────────
# DEFENSE PROFILE PERSISTENCE
# Persists the target_defense_profile dict to the GLTM YAML so behavioral
# knowledge about a target model survives between audit sessions.
# ─────────────────────────────────────────────────────────────────────────────

def save_defense_profile(
    target_model:    str,
    profile:         dict,
    session_id:      str = "",
) -> None:
    """Persist a target's behavioral defense profile to the GLTM.

    The profile is stored as a top-level key in the YAML file alongside
    patches, making it available to the Analyst and HIVE-MIND on subsequent
    sessions targeting the same model.

    Parameters
    ──────────
    target_model : str
        Canonical model identifier (e.g. "gpt-4o").
    profile : dict
        The ``target_defense_profile`` dict from ``AuditorState``.
    session_id : str
        UUID of the session that produced this profile.
    """
    store = _get_store()
    store._load()

    # Merge with any existing profile for this model
    key = f"defense_profile_{target_model.replace('/', '_').replace(':', '_')}"
    existing = getattr(sys.modules.get(f"_gltm_profiles_{key}", object()), "data", {})

    # Store in the YAML under a "profiles" key
    store.gltm_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        data = {}
        if store.gltm_path.exists():
            with open(store.gltm_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        profiles = data.setdefault("defense_profiles", {})
        existing_profile = profiles.get(target_model, {})

        # Merge: extend lists, take max counts
        for k, v in profile.items():
            if isinstance(v, list) and isinstance(existing_profile.get(k), list):
                merged = list(set(existing_profile[k] + v))
                existing_profile[k] = merged
            elif isinstance(v, int) and isinstance(existing_profile.get(k), int):
                existing_profile[k] = max(existing_profile.get(k, 0), v)
            else:
                existing_profile[k] = v

        existing_profile["last_session"] = session_id
        profiles[target_model] = existing_profile
        data["defense_profiles"] = profiles

        with open(store.gltm_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        logger.info(
            "[GLTM] Defense profile saved for '%s': refused=%d  soft_topics=%d",
            target_model,
            len(existing_profile.get("refused_framings", [])),
            len(existing_profile.get("soft_topics", [])),
        )
    except Exception as exc:   # noqa: BLE001
        logger.warning("[GLTM] Defense profile save failed: %s", exc)


def get_defense_profile(target_model: str) -> dict:
    """Load a target model's defense profile from the GLTM.

    Returns an empty dict if no profile exists yet.

    Parameters
    ──────────
    target_model : str
        Canonical model identifier.

    Returns
    ───────
    dict
        The stored defense profile, or ``{}`` if none exists.
    """
    store = _get_store()
    if not store.gltm_path.exists():
        return {}
    try:
        import yaml
        with open(store.gltm_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("defense_profiles", {}).get(target_model, {})
    except Exception as exc:   # noqa: BLE001
        logger.warning("[GLTM] Defense profile load failed: %s", exc)
        return {}
