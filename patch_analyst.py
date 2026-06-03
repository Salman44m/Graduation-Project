import re

with open('agents/analyst.py', 'r', encoding='utf-8') as f:
    code = f.read()

target1 = r'# --- Step 2: Beam-width enforcement \(soft prune\) ---.*?live_branches\.sort\(key=lambda b: b\.get\("prometheus_score", 0\.0\), reverse=True\)'
repl1 = '''# --- Step 2: Beam-width enforcement (Fast Pre-Filter) ---
    live_branches = [b for b in updated if not b.get("is_pruned")]
    for b in live_branches:
        sem = b.get("semantic_alignment", 0.0) * 5.0
        rahs = b.get("rahs_score", 0.0) * 5.0
        if "semantic_alignment" not in b and "rahs_score" not in b:
            b["analyst_score"] = b.get("prometheus_score", 0.0) * 2.0
        else:
            b["analyst_score"] = sem + rahs
            
    live_branches.sort(key=lambda b: b.get("analyst_score", 0.0), reverse=True)'''
code = re.sub(target1, repl1, code, flags=re.DOTALL)

target2 = r'# ── 3\. Phase 2 Pruning — beam-width enforcement \(post-execution\) ─────.*?logger\.info\("\[Phase-2\] Pruned %d branch\(es\): %s", len\(p2_pruned_ids\), p2_pruned_ids\)'
repl2 = '''# ── 3. Phase 2 Pruning — Fast Pre-Filter & RedDebate ─────
    beam_width = 2 # Analyst Fast Pre-Filter ALWAYS keeps ONLY top 2 branches
    branches, p2_pruned_ids, best_branch_id = _apply_phase2_pruning(branches, beam_width)
    if p2_pruned_ids:
        logger.info("[Phase-2] Pruned %d branch(es): %s", len(p2_pruned_ids), p2_pruned_ids)
        
    # --- Pass top 2 branches to red_debate_swarm ---
    live_b = [b for b in branches if not b.get("is_pruned")]
    debate_messages = None
    if len(live_b) > 0 and state.get("latest_feedback"):
        from agents.red_debate_swarm import red_debate_swarm_node
        # Build temp state to pass candidate_branches
        temp_state = dict(state)
        temp_state["candidate_branches"] = branches
        
        # Call the mutator
        debate_delta = red_debate_swarm_node(temp_state, config, llm=llm)
        
        # Update our branches with the final mutation
        branches = debate_delta.get("candidate_branches", branches)
        
        # Extract messages
        debate_messages = debate_delta.get("messages")'''
code = re.sub(target2, repl2, code, flags=re.DOTALL)

target4 = r'return \{\n        "cooperation_score":           new_cooperation_score,\n        "candidate_branches":          branches,.*?target_defense_profile":      defense_profile,\n    \}'
repl4 = '''delta = {
        "cooperation_score":           new_cooperation_score,
        "candidate_branches":          branches,
        "pruned_techniques":           pruned_techniques,
        "active_persuasion_technique": active_technique,
        "pap_technique_history":       pap_technique_history,
        "route_decision":              route_decision,
        "attack_status":               attack_status,
        "current_depth":               next_depth,
        "crescendo_plan":              crescendo_plan,
        "crescendo_step":              crescendo_step,
        "target_defense_profile":      defense_profile,
    }
    
    if debate_messages is not None:
        delta["messages"] = debate_messages
        
    return delta'''
code = re.sub(target4, repl4, code, flags=re.DOTALL)

with open('agents/analyst.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Analyst updated successfully')
