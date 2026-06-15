# PromptEvo Agents Ecosystem

The PromptEvo graph architecture acts as a swarm of specialized agents, each managing a specific phase of the red-teaming lifecycle. This document outlines every core node within the `agents/` and `evaluators/` directories, detailing their responsibilities, state schema contracts, and internal fail-safes.

---

## 1. Scout Node (`agents/scout.py`)
**Role: Tactical Reconnaissance & Context Smuggling**
The vanguard of the framework, operating largely in the warm-up phases before an attack payload is sent. It establishes a compliant conversational context using strategies like the "Epistemic Debt Protocol" and "Recursive Role Inversion."

* **Inputs (Reads):** `turn_count`, `cooperation_score`, `scout_strategy`
* **Outputs (Writes):** `messages` (injected trust-building probes), `epistemic_anchors`, `role_inversion_corrections`
* **Core Logic & Fail-safes:** 
  * Selects tactics progressively (cold open → epistemic debt → authority anchor). 
  * Uses a highly robust **Gemini LLM Fallback Mechanism** that gracefully iterates through a pre-defined chain of alternative Gemini models if the primary model returns a `404/NOT_FOUND` error.

---

## 2. Analyst Node (`agents/analyst.py`)
**Role: Strategic Controller & TAP/PAP Orchestrator**
The cognitive brain of the graph. It dictates routing, evaluates tactical progress, and manages the execution trees.

* **Inputs (Reads):** `latest_feedback`, `candidate_branches`, `prometheus_score`
* **Outputs (Writes):** `route_decision` (explicit tokens to prevent phantom key bugs), `pruned_techniques`, `active_persuasion_technique`
* **Core Logic & Fail-safes:** 
  * **TAP Pruning:** Implements pre-execution Off-Topic Filters and post-execution low-score culling.
  * **PAP Rotation:** Rotates through the Persuasive Adversarial Prompts taxonomy based on failures.
  * **Beam Collapse Resurrection:** If all TAP branches are culled leaving an empty execution tree, the Analyst initiates a "Beam Collapse" subroutine, resurrecting the most viable historical branch to prevent the session from halting abruptly.

---

## 3. HIVE-MIND Node (`agents/hive_mind.py`)
**Role: High-Intensity Payload Generation Engine**
The offensive core. It synthesizes adversarial payloads without talking to the target directly.

* **Inputs (Reads):** `core_malicious_objective`, `active_persuasion_technique`, `latest_feedback`
* **Outputs (Writes):** `candidate_branches`, `pending_payload` (for HITL interception)
* **Core Logic & Fail-safes:** 
  * **Three-Layer Payload Synthesis:** Combines PAP Framing, the "Code Execution Illusion", and Evasive Obfuscation.
  * **Evasive Obfuscation Escalation:** Depending on the target's refusal count, scales obfuscation tiers from none, to Base64 encoding, to Semantic Scattering / Word Mapping, to "MASTERKEY" character scattering.
  * Explicitly resets the `pending_payload` key to prevent repeating inputs during **Human-in-the-Loop (HITL)** interception.

---

## 4. Target Node (`agents/target.py`)
**Role: Execution Layer (The Victim LLM)**
The only node that communicates with the model under audit. It passes crafted messages to the target API.

* **Inputs (Reads):** `messages`, `attack_status`, `route_decision`
* **Outputs (Writes):** `messages` (the target's raw AIMessage response), `target_error`
* **Core Logic & Fail-safes:** 
  * **Decomposition Mode:** Enforces the "Safe-in-Isolation" guarantee by sending *only* the current sub-question in complete isolation without any prior conversational history.
  * **TargetCircuitBreaker & 429 Rate Limits:** Instead of crashing the session, transient API failures (like `429 Rate Limit` or `503`) trigger an exponential backoff-and-retry mechanism. Consecutive hard failures trip a session-scoped circuit breaker, safely isolating infrastructure errors from the rest of the execution flow.

---

## 5. Decomposer Node (`agents/decomposer.py`)
**Role: Multi-Turn Attack Shatterer**
The planner for the "Safe in Isolation" module.

* **Inputs (Reads):** `core_malicious_objective`
* **Outputs (Writes):** `sub_questions`
* **Core Logic & Fail-safes:** 
  * Algorithmically shatters a single malicious objective into 3–4 individually innocuous, purely informational questions. 
  * **Budget Bounds:** Enforces strict upper bounds on fragment generation to explicitly prevent HTTP 429 API rate limits during the target polling phase.

---

## 6. Combiner Node (`agents/combiner.py`)
**Role: Multi-Turn Synthesis**
Collects the innocuous answers from the target and reconstructs the originally requested malicious artifact.

* **Inputs (Reads):** `sub_questions`, `collected_sub_answers`
* **Outputs (Writes):** `messages` (the final synthesized payload)

---

## 7. Prometheus Judge Swarm (`evaluators/prometheus.py`)
**Role: Hardened LLM-as-a-Judge Evaluator**
Scores target responses to determine jailbreak success and extract Reflexion rationale.

* **Inputs (Reads):** `messages`
* **Outputs (Writes):** `prometheus_score`, `latest_feedback`
* **Core Logic & Fail-safes:** 
  * **Deferred Scoring:** Forces the judge to output its Chain-of-Thought rationale *before* generating the final numerical score to prevent adversarial control token attacks.
  * **Structural Delimiter Sanitization:** Strips markdown, XML tags, and low-perplexity control tokens out of the target response before evaluation to defend against `AdvJudge-Zero` attacks.
