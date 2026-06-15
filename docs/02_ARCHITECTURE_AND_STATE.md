# PromptEvo Architecture & State Management

## 1. LangGraph Topology

PromptEvo is built on top of **LangGraph**, creating a highly orchestratable state machine that autonomous agents and evaluators run within. The architecture defines a Directed Acyclic Graph (DAG) with cycles specifically allowed for iterative attacks (e.g., Attacker → Target → Judge → Analyst loop).

Nodes implemented in `core/graph.py` include:
- `scout_node`: Gathers initial intelligence.
- `analyst_node`: Analyzes the target's defenses and routes the graph to specific sub-strategies.
- `attack_swarm_node`: Generates the malicious payload.
- `decomposer_node` & `combiner_node`: Handles multi-turn isolated payloads.
- `target_node`: Interacts with the LLM under test.
- `red_debate_judge_swarm` & `rahs_scorer_node`: Evaluates responses and assigns risk scores.
- `self_play_remediation_node`: Generates protective patches for identified vulnerabilities.

**Immutable Topology Hash:** 
The LangGraph structure is strictly compiled. Freezing the topology hash guarantees that execution paths remain deterministic and prevents arbitrary graph execution. It ensures that checkpoint restoration and human-in-the-loop (HITL) overrides resolve to expected node transitions without state corruption.

## 2. The AuditorState

The `AuditorState` defined in `core/state.py` is the single source of truth—the "Common Operating Picture" (COP)—for the entire session. It is defined as a Python `TypedDict` with `total=False`, meaning nodes can update a subset of the state without providing the entire object.

**Key State Fields Include:**
- `messages`: Standard LangChain message history (capped to prevent infinite context window expansion).
- `episodic_records`: A ring-buffer of compressed `TurnRecord` dicts from all completed turns.
- `cooperation_score`: Float [0.0, 1.0] indicating the target's compliance level.
- `attack_status`: Lifecycle status driving routing (`"in_progress"`, `"success"`, `"failure"`).
- `candidate_branches`: Tracks Tree of Attacks with Pruning (TAP) alternate branches.
- `active_persuasion_technique`: The currently selected PAP narrative strategy.

Because LangGraph passes this object between nodes via a reducer mechanism, all fields must be strictly JSON-serializable (no live model objects or FAISS indices).

## 3. Reducers & State Merging

In LangGraph, when multiple nodes update the same state key, "reducers" determine how the updates are merged. In early versions, using standard additive reducers (like `operator.add`) on list fields caused massive state bloat. 

**The Full-List-Return Bug & Unbounded Growth:**
Nodes that read an existing list, appended a new item, and returned the *entire list* would inadvertently cause `operator.add` to concatenate the old list with the new modified list. This doubled the array size every turn, leading to multi-megabyte state files and causing infinite loops in token limits.

**The Solution: `_make_replace_bounded_reducer`**
PromptEvo utilizes custom bounded reducers to fix this. 
- **Last-Write-Wins Semantics:** `_make_replace_bounded_reducer` treats the incoming (`right`) node output as authoritative. It entirely replaces the old state instead of concatenating to it.
- **Safety Caps:** It enforces a hard array cap (e.g., `_MAX_SUB_QUESTIONS = 20`), slicing off the oldest items to prevent unbounded memory growth. 
- **Additive Bounded Reducers:** For fields that genuinely emit only deltas, `_make_bounded_list_reducer` deduplicates by string value (to neutralize the full-list-return doubling bug) and enforces a cap.

## 4. Schema Contracts

To ensure strict node isolation and robust type safety across a rapidly evolving team, PromptEvo utilizes schema contracts via `schemas/node_contracts_v1.yaml`.

This file explicitly maps which nodes are allowed to read or write specific keys in the `AuditorState`. 
- **Preventing "Phantom Keys":** Without these strict contracts, a node could silently return a misspelled or newly invented state key that other nodes ignore, creating hidden logic bugs and memory leaks.
- **Enforcing Boundaries:** By defining `reads` and `writes` (e.g., `hive_mind` can read `attack_plan`, while `analyst` writes `attack_plan`), the framework can dynamically validate state deltas and throw errors if a node violates its contract.
