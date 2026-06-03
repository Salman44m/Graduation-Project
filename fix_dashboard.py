import re

with open("dashboard.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update the button CSS selectors to be explicitly stronger
content = content.replace('.stButton > button[kind="primary"]', '.stApp .stButton > button[kind="primary"]')
content = content.replace('.stButton > button[kind="secondary"]', '.stApp .stButton > button[kind="secondary"]')

# Force contrast via hex if Dark mode is active, wait, the CSS is generated dynamically in dashboard.py:
# We just need to replace the var(--btn-primary-bg) with #EDEDED in Dark Mode CSS, but wait!
# If we change the selector to `.stApp .stButton...`, the CSS variables should work since they're in the :root and scoped correctly.
# But the user asked to "Force background-color: #EDEDED !important; and color: #000000 !important;" for dark mode primary button.
# To be 100% safe, I will update the dark mode root variables for the button.
content = content.replace('--btn-primary-bg: #FFFFFF;', '--btn-primary-bg: #EDEDED;')
content = content.replace('--btn-primary-text: #000000;', '--btn-primary-text: #000000;')

# 2. Extract the second sidebar block and merge it to the top.
# Find the second "with st.sidebar:" block.
# We know it starts with 'with st.sidebar:' and contains 'st.markdown(\'<div class="war-room-title">'
# Let's find it.

match = re.search(r'\nwith st\.sidebar:\n\s*st\.markdown\(\'<div class="war-room-title".*?# ──\n# HITL REVIEW PANEL', content, flags=re.DOTALL)
if match:
    full_second_sidebar = match.group(0)
    
    # We must remove it from its current location, EXCEPT we want to leave the `# ──\n# HITL REVIEW PANEL` part
    hitl_marker = '\n\n# ─────────────────────────────────────────────────────────────────────────────\n# HITL REVIEW PANEL'
    sidebar_only = full_second_sidebar.replace(hitl_marker, '')
    
    # Remove the second sidebar from the original content
    content = content.replace(sidebar_only, '')
    
    # Find the first sidebar block
    first_sidebar = '\nwith st.sidebar:\n    theme_choice = st.radio("Theme", ["Dark", "Light"], horizontal=True, key="theme_choice")\n'
    
    # Remove the `with st.sidebar:` from the second block so they can merge seamlessly
    inner_sidebar = sidebar_only.replace('\nwith st.sidebar:\n', '\n')
    
    merged_sidebar = first_sidebar + inner_sidebar
    
    content = content.replace(first_sidebar, merged_sidebar)

with open("dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done parsing and writing.")
