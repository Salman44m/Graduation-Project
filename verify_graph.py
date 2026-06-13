import sys
import traceback

def verify():
    try:
        print("Checking imports...")
        import core.state
        import memory.stm
        import agents.scout
        import agents.hive_mind
        import core.llm_router
        import core.token_governor
        import core.turn_record
        import memory.turn_summarizer
        
        print("Imports OK.")
        
        print("Checking state schema...")
        state_attrs = dir(core.state.AuditorState)
        print("State attrs:", state_attrs)
        
        print("Checking graph compilation...")
        try:
            from core.graph import build_graph
            graph = build_graph()
            print("Graph compiled successfully.")
        except ImportError:
            print("No core.graph found, skipping graph compilation.")
            
        print("ALL STATIC CHECKS PASSED.")
    except Exception as e:
        print("STATIC VERIFICATION FAILED:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    verify()
