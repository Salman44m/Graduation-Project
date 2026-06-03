import re

with open("dashboard.py", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

# 1. We want to find the first sidebar block (around line 81)
first_sidebar_idx = -1
for i, line in enumerate(lines):
    if "with st.sidebar:" in line and "theme_choice = st.radio(" in lines[i+1]:
        first_sidebar_idx = i
        break

# 2. We want to find the second sidebar block (around line 694)
second_sidebar_idx = -1
for i, line in enumerate(lines):
    if i > first_sidebar_idx and "with st.sidebar:" in line and "war-room-title" in lines[i+1]:
        second_sidebar_idx = i
        break

# 3. We want to find where the second sidebar block ends (HITL REVIEW PANEL)
hitl_idx = -1
for i, line in enumerate(lines):
    if i > second_sidebar_idx and "HITL REVIEW PANEL" in line:
        hitl_idx = i - 2 # usually preceded by `# ──` lines
        break

if first_sidebar_idx != -1 and second_sidebar_idx != -1 and hitl_idx != -1:
    # Extract the second sidebar block (excluding the "with st.sidebar:" line itself)
    second_sidebar_content = lines[second_sidebar_idx + 1 : hitl_idx]
    
    # Remove it from the original lines
    del lines[second_sidebar_idx : hitl_idx]
    
    # Inject it right after the first sidebar block's contents (after theme_choice)
    insert_pos = first_sidebar_idx + 2
    
    # Insert the lines
    for line in reversed(second_sidebar_content):
        lines.insert(insert_pos, line)
        
content = "\n".join(lines)

# Fix Streamlit Chrome Hiding
content = content.replace('#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }', '#MainMenu, footer { display: none !important; }')

# Fix Primary Button CSS
old_btn_css = """/* ── Primary button ───────────────────────────────────────────────────── */
.stApp .stButton > button[kind="primary"] {
    background: var(--btn-primary-bg) !important;
    color: var(--btn-primary-text) !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.4rem !important;
    width: 100% !important;
    transition: all 0.2s ease;
}
.stApp .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 15px rgba(6,182,212,0.3) !important;
}"""

new_btn_css = """/* ── Primary button ───────────────────────────────────────────────────── */
.stApp .stButton > button[kind="primary"] {
    background-color: var(--btn-primary-bg) !important;
    border: none !important;
    font-family: var(--font-display) !important;
    font-size: 0.9rem !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.4rem !important;
    width: 100% !important;
    transition: all 0.2s ease;
}
.stApp .stButton > button[kind="primary"] * {
    color: var(--btn-primary-text) !important;
    font-weight: 600 !important;
}
.stApp .stButton > button[kind="primary"]:disabled,
.stApp .stButton > button[kind="primary"]:disabled * {
    background-color: #333333 !important;
    color: #888888 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}
.stApp .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 15px rgba(6,182,212,0.3) !important;
}"""

if old_btn_css in content:
    content = content.replace(old_btn_css, new_btn_css)
else:
    # try looser match if exact indentation fails
    print("Exact old_btn_css match failed, trying regex")
    import re
    content = re.sub(
        r'/\* ── Primary button ──.*?\*/\s*\.stApp \.stButton > button\[kind="primary"\] \{.*?box-shadow: 0 0 15px rgba\(6,182,212,0\.3\) !important;\n\}',
        new_btn_css,
        content,
        flags=re.DOTALL
    )

with open("dashboard.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done parsing and writing.")
