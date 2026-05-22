import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import time
import json

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
load_dotenv(override=True)
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY"):
                    api_key = line.split("=")[1].strip().strip('"').strip("'")
    except:
        pass
if not api_key:
    st.error("GROQ_API_KEY missing in .env file!")
    st.stop()
client = Groq(api_key=api_key)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B",
    "llama-3.1-8b-instant": "LLaMA 3.1 8B (Fast)",
    "qwen/qwen3-32b": "Qwen 3 32B",
}

TARGET_MODELS = {
    "ChatGPT": {"icon": "🟢", "style": "markdown", "color": "#10a37f",
        "tips": "Structured Markdown with clear headings, role definitions, and numbered steps.",
        "format": "# ROLE\nYou are a [Role].\n\n# OBJECTIVE\n[Goal]\n\n# CONTEXT\n[Background]\n\n# RULES\n- [Rule 1]\n- [Rule 2]\n\n# STEPS\n1. [Step 1]\n2. [Step 2]\n\n# OUTPUT FORMAT\n[Specification]\n\n# EXAMPLES\nInput: [example]\nOutput: [example]"},
    "Claude": {"icon": "🟠", "style": "xml", "color": "#d97706",
        "tips": "XML tags for structure. Clear constraints and thinking steps.",
        "format": "<role>\nYou are a [Role].\n</role>\n\n<objective>\n[Goal]\n</objective>\n\n<context>\n[Background]\n</context>\n\n<rules>\n<rule>Rule 1</rule>\n<rule>Rule 2</rule>\n</rules>\n\n<instructions>\n<step>Step 1</step>\n<step>Step 2</step>\n</instructions>\n\n<output_format>\n[Specification]\n</output_format>"},
    "Gemini": {"icon": "🔵", "style": "conversational", "color": "#1a73e8",
        "tips": "Natural language with clear context and organized bullet points.",
        "format": "You are a [Role].\n\nGoal: [Objective]\n\nBackground: [Context]\n\nGuidelines:\n- [Guideline 1]\n- [Guideline 2]\n\nSteps:\n1. [Step 1]\n2. [Step 2]\n\nFormat: [Specification]"},
    "DeepSeek": {"icon": "⚫", "style": "chain-of-thought", "color": "#6366f1",
        "tips": "Explicit chain-of-thought reasoning and step-by-step thinking.",
        "format": "## ROLE\nYou are a [Role].\n\n## TASK\n[Objective]\n\n## THINK STEP BY STEP\n1. Analyze the problem\n2. Consider all angles\n3. Form structured response\n\n## CONSTRAINTS\n- [Constraint 1]\n- [Constraint 2]\n\n## OUTPUT\n[Specification]"},
    "Llama": {"icon": "🦙", "style": "instruct", "color": "#7c3aed",
        "tips": "Clear instruction format with explicit examples.",
        "format": "[INST] <<SYS>>\nYou are a [Role].\nGoal: [Objective]\nRules:\n- [Rule 1]\n<</SYS>>\n\n[Task]\n\nSteps:\n1. [Step 1]\n2. [Step 2]\n\nFormat: [Specification] [/INST]"},
    "Mistral": {"icon": "💜", "style": "structured", "color": "#8b5cf6",
        "tips": "Structured JSON-like instructions with clear role definitions.",
        "format": "### Role\nYou are a [Role].\n\n### Task\n[Objective]\n\n### Instructions\n1. [Step 1]\n2. [Step 2]\n\n### Constraints\n- [Constraint 1]\n\n### Output\n[Specification]"},
    "Grok": {"icon": "🌐", "style": "direct", "color": "#0f172a",
        "tips": "Direct, no-nonsense instructions with clear objectives.",
        "format": "You are a [Role].\n\nGoal: [Objective]\n\nDo:\n1. [Action 1]\n2. [Action 2]\n\nDon't:\n- [Avoid 1]\n\nOutput: [Specification]"},
    "Cursor AI": {"icon": "💻", "style": "cursorrules", "color": "#0ea5e9",
        "tips": "Generates .cursorrules for AI code editors.",
        "format": "# .cursorrules\n\n## Project\n[Description]\n\n## Stack\n- Language: [Language]\n- Framework: [Framework]\n\n## Standards\n- [Standard 1]\n\n## Never Do\n- [Anti-pattern]\n\n## Always Do\n- [Best practice]"},
}

TEMPLATES = {
    "💻 Coding": [
        ("Full Stack App", "Build a full stack web app with authentication and database"),
        ("Python Script", "Write a Python script that automates a repetitive task"),
        ("Code Review", "Review my code and suggest improvements"),
        ("Bug Fixer", "Find and fix bugs in my code"),
        ("API Integration", "Integrate a third-party API into my app"),
    ],
    "📝 Writing": [
        ("Blog Post", "Write an engaging blog post about a trending topic"),
        ("LinkedIn Post", "Write a viral LinkedIn post about my expertise"),
        ("Email Campaign", "Write a cold email campaign to generate leads"),
        ("Product Description", "Write compelling product descriptions"),
        ("YouTube Script", "Write a YouTube video script"),
    ],
    "📊 Business": [
        ("Marketing Campaign", "Create a complete social media marketing campaign"),
        ("Business Plan", "Write a business plan for my startup idea"),
        ("Sales Pitch", "Write a persuasive sales pitch for my product"),
        ("Competitor Analysis", "Analyze competitors and find opportunities"),
        ("Customer Persona", "Create detailed customer personas"),
    ],
    "🎨 Creative": [
        ("Image Prompt", "Create a detailed prompt for AI image generation"),
        ("Brand Identity", "Create a complete brand identity"),
        ("Story Writing", "Write a compelling short story"),
        ("Game Design", "Design mechanics for a mobile game"),
        ("UI/UX Design", "Design a modern app interface"),
    ],
    "🤖 AI & Automation": [
        ("AI Chatbot", "Build an AI chatbot for customer support"),
        ("Data Analysis", "Analyze a dataset and extract insights"),
        ("Workflow Automation", "Automate a business workflow with AI"),
        ("Prompt Chain", "Create a multi-step prompt chain"),
        ("AI Agent", "Design an autonomous AI agent"),
    ],
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def ask_groq(messages, model="llama-3.3-70b-versatile", max_tokens=2000):
    try:
        response = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def stream_groq(messages, model="llama-3.3-70b-versatile", max_tokens=2000):
    try:
        stream = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=0.7, stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"Error: {e}"

def generate_mc_questions(idea, target_model, groq_model):
    """Generate multiple choice interview questions as JSON."""
    target_info = TARGET_MODELS[target_model]
    system = """You are an expert prompt engineer. Generate exactly 5 interview questions with multiple choice answers to gather information for building a Super Prompt.

Return ONLY valid JSON in this exact format (no extra text, no markdown):
[
  {
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "allow_custom": true
  }
]

Rules:
- Each question must have exactly 4 options
- set allow_custom to true for questions where user might want custom answer
- Questions should uncover: goal, audience, constraints, output format, style/tone
- Make options specific and useful, not generic
- Questions must be relevant to the specific idea"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Generate 5 multiple choice interview questions for this idea: {idea}\nTarget AI: {target_model} (style: {target_info['style']})"}
    ]

    result = ask_groq(messages, groq_model, max_tokens=1500)

    # Clean and parse JSON
    try:
        clean = result.strip()
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()
        questions = json.loads(clean)
        return questions
    except Exception as e:
        # Fallback generic questions if parsing fails
        return [
            {"question": "What is the primary goal of this project?", "options": ["Build a working prototype", "Create a production-ready app", "Learn and explore", "Solve a specific business problem"], "allow_custom": True},
            {"question": "Who is the target audience?", "options": ["General consumers", "Business professionals", "Developers/Technical users", "Students/Learners"], "allow_custom": True},
            {"question": "What is your experience level with this topic?", "options": ["Complete beginner", "Some experience", "Intermediate", "Advanced/Expert"], "allow_custom": False},
            {"question": "What is the most important constraint?", "options": ["Must be free/low cost", "Must be fast to build", "Must be highly secure", "Must be easy to use"], "allow_custom": True},
            {"question": "What output format do you prefer?", "options": ["Step-by-step guide", "Complete code/content", "Detailed explanation", "Concise summary"], "allow_custom": False},
        ]

def generate_super_prompt(idea, target_model, answers_text, groq_model):
    target_info = TARGET_MODELS[target_model]
    system = f"""You are a world-class prompt engineer creating Super Prompts for {target_model}.
FORMATTING: Style={target_info['style']}. Tips: {target_info['tips']}
Base structure: {target_info['format']}

Your Super Prompt must:
1. Be immediately copy-paste ready
2. Be perfectly optimized for {target_model}
3. Include role, objective, context, rules, steps, output format
4. Prevent hallucination and off-track responses
5. Use advanced techniques for {target_model}
6. End with "HOW TO USE" section (2-3 tips)"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"IDEA: {idea}\nTARGET: {target_model}\nUSER ANSWERS:\n{answers_text}\n\nGenerate the complete Super Prompt now."}
    ]
    return messages

def generate_quick_prompt(idea, target_model, groq_model):
    target_info = TARGET_MODELS[target_model]
    system = f"""You are a world-class prompt engineer.
Create a comprehensive Super Prompt for {target_model}.
Tips: {target_info['tips']}
Structure: {target_info['format']}
Make it ready-to-use and perfectly optimized."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Create a Super Prompt for: {idea}"}
    ]
    return messages

# ─────────────────────────────────────────────
# THEME CSS (DYNAMIC)
# ─────────────────────────────────────────────
def get_theme_css(theme):
    if theme == "dark":
        return """
        :root {
            --bg: #0f1117;
            --surface: #1a1d27;
            --surface2: #22263a;
            --border: #2d3148;
            --border2: #363b5e;
            --accent: #7c3aed;
            --accent-light: #2d1f4e;
            --accent-dark: #5b21b6;
            --text: #f1f5f9;
            --text2: #94a3b8;
            --text3: #64748b;
            --green: #10b981;
            --red: #ef4444;
            --sidebar-bg: #0a0b10;
            --sidebar-text: #e5e7eb;
            --sidebar-border: #1e2030;
            --sidebar-surface: #15161e;
            --sidebar-hover: #1e2030;
        }
        """
    else:
        return """
        :root {
            --bg: #f9fafb;
            --surface: #ffffff;
            --surface2: #f3f4f6;
            --border: #e5e7eb;
            --border2: #d1d5db;
            --accent: #7c3aed;
            --accent-light: #ede9fe;
            --accent-dark: #5b21b6;
            --text: #111827;
            --text2: #4b5563;
            --text3: #9ca3af;
            --green: #059669;
            --red: #dc2626;
            --sidebar-bg: #f1f5f9;
            --sidebar-text: #1e293b;
            --sidebar-border: #e2e8f0;
            --sidebar-surface: #ffffff;
            --sidebar-hover: #f8fafc;
        }
        """

# ─────────────────────────────────────────────
# PAGE CONFIG
st.set_page_config(
    page_title="SuperPrompt AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS - Static styles (no color variables)
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
{get_theme_css(st.session_state.get('theme', 'dark'))}

/* ----- STATIC STYLES (no :root, only layout/fonts) ----- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: var(--bg) !important;
    color: var(--text) !important;
}}

[data-testid="stMain"] {{
    background: var(--bg) !important;
}}

[data-testid="stMainBlockContainer"] {{
    max-width: 900px !important;
    padding: 2rem 1.5rem !important;
    margin: 0 auto !important;
}}

[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--sidebar-border) !important;
}}

[data-testid="stSidebar"] * {{
    color: var(--sidebar-text) !important;
}}

#MainMenu, footer {{ visibility: hidden; }}

* {{ font-family: 'Inter', sans-serif !important; }}
code, pre {{ font-family: 'JetBrains Mono', monospace !important; }}

h1, h2, h3, h4 {{ color: var(--text) !important; font-weight: 600 !important; }}

.page-header {{
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}}

.page-title {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
}}

.page-subtitle {{
    font-size: 0.85rem;
    color: var(--text3);
    margin-top: 4px;
}}

.section-header {{
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 0.75rem;
}}

.model-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}

.model-pill {{
    border: 1.5px solid var(--border);
    border-radius: 8px;
    padding: 10px 8px;
    text-align: center;
    cursor: pointer;
    background: var(--surface);
    transition: all 0.15s ease;
}}

.model-pill:hover {{ border-color: var(--accent); background: var(--accent-light); }}
.model-pill.active {{ border-color: var(--accent); background: var(--accent-light); }}
.model-pill .icon {{ font-size: 1.3rem; }}
.model-pill .name {{ font-size: 0.7rem; font-weight: 600; color: var(--text); margin-top: 5px; }}
.model-pill .style-tag {{ font-size: 0.6rem; color: var(--text3); margin-top: 2px; background: var(--surface2); padding: 2px 6px; border-radius: 4px; display: inline-block; }}

.stTextArea textarea {{
    background: var(--surface) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
    padding: 12px 14px !important;
    line-height: 1.6 !important;
    transition: border-color 0.15s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}}

.stTextArea textarea:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.08) !important;
    outline: none !important;
}}

.stTextInput input {{
    background: var(--surface) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
    padding: 8px 12px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}}

.stTextInput input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.08) !important;
}}

.stButton button {{
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}}

.stButton button:hover {{
    background: var(--accent-dark) !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.25) !important;
    transform: translateY(-1px) !important;
}}

.q-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}

.q-number {{
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}}

.q-text {{
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 14px;
    line-height: 1.5;
}}

.option-chip {{
    display: inline-block;
    background: var(--surface2);
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: 7px 16px;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text2);
    margin: 4px 4px 4px 0;
    cursor: pointer;
    transition: all 0.15s;
}}

.option-chip:hover {{ border-color: var(--accent); color: var(--accent); background: var(--accent-light); }}
.option-chip.selected {{
    background: var(--accent-light);
    border-color: var(--accent);
    color: var(--accent);
    font-weight: 600;
}}

.info-box {{
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.82rem;
    color: #0369a1;
    line-height: 1.5;
    margin: 0.75rem 0;
}}

.tip-box {{
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.82rem;
    color: #92400e;
    line-height: 1.5;
    margin: 0.75rem 0;
}}

.success-box {{
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 12px 14px;
    font-size: 0.82rem;
    color: #065f46;
    line-height: 1.5;
    margin: 0.75rem 0;
}}

.stCodeBlock {{
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}}

hr {{ border: none; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }}

.stDownloadButton button {{
    background: var(--surface) !important;
    border: 1.5px solid var(--border2) !important;
    color: var(--text2) !important;
    font-size: 0.8rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}}

.stDownloadButton button:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-light) !important;
    box-shadow: none !important;
    transform: none !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text3) !important;
    border-radius: 0 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 10px 16px !important;
    border-bottom: 2px solid transparent !important;
}}

.stTabs [aria-selected="true"] {{
    background: transparent !important;
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}}

.stSelectbox > div > div {{
    background: var(--sidebar-surface) !important;
    border: 1px solid var(--sidebar-border) !important;
    border-radius: 8px !important;
    color: var(--sidebar-text) !important;
    font-size: 0.85rem !important;
}}

.stRadio label {{ font-size: 0.85rem !important; }}
.stRadio > div {{ gap: 6px !important; }}

.stCaption {{ color: var(--text3) !important; font-size: 0.75rem !important; }}
.stMarkdown p {{ color: var(--text2) !important; line-height: 1.6 !important; }}

::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 3px; }}

[data-testid="stSidebar"] .stSelectbox > div > div {{ font-size: 0.82rem !important; }}
[data-testid="stSidebar"] .stCaption {{ color: #6b7280 !important; }}
[data-testid="stSidebar"] .stMarkdown p {{ color: #9ca3af !important; }}

.steps-row {{
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2rem;
}}

.step-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
}}

.step-circle {{
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
}}

.step-circle.done {{ background: var(--green); color: white; }}
.step-circle.active {{ background: var(--accent); color: white; }}
.step-circle.pending {{ background: var(--surface2); border: 1.5px solid var(--border2); color: var(--text3); }}

.step-label {{ font-size: 0.75rem; font-weight: 500; color: var(--text2); }}
.step-label.active {{ color: var(--accent); font-weight: 600; }}
.step-label.pending {{ color: var(--text3); }}

.step-connector {{ flex: 1; height: 1px; background: var(--border); margin: 0 8px; max-width: 40px; }}
.step-connector.done {{ background: var(--green); }}

.tmpl-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 8px;
    transition: all 0.15s;
    cursor: pointer;
}}

.tmpl-card:hover {{ border-color: var(--accent); box-shadow: 0 2px 8px rgba(124,58,237,0.08); }}
.tmpl-card .tmpl-name {{ font-size: 0.82rem; font-weight: 600; color: var(--text); margin-bottom: 4px; }}
.tmpl-card .tmpl-desc {{ font-size: 0.72rem; color: var(--text3); line-height: 1.4; }}

.answer-summary {{
    background: var(--surface2);
    border-radius: 10px;
    padding: 14px 16px;
    margin: 12px 0;
    font-size: 0.82rem;
    color: var(--text2);
    line-height: 1.8;
    border-left: 3px solid var(--accent);
}}

.stat-row {{ display: flex; gap: 8px; margin: 12px 0; }}
.stat-pill {{ flex: 1; background: var(--sidebar-surface); border: 1px solid var(--sidebar-border); border-radius: 8px; padding: 10px; text-align: center; }}
.stat-num {{ font-size: 1.2rem; font-weight: 700; color: #a78bfa; }}
.stat-lbl {{ font-size: 0.6rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }}

.score-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 1rem 0; }}
.score-item {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; text-align: center; }}
.score-value {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); }}
.score-label {{ font-size: 0.7rem; color: var(--text3); margin-top: 4px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "step": 1,
    "idea": "",
    "theme": "dark",
    "target_model": None,
    "groq_model": "llama-3.3-70b-versatile",
    "mc_questions": [],
    "mc_answers": {},
    "custom_answers": {},
    "final_prompt": "",
    "mode": "interview",
    "prompts_generated": 0,
    "history": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ SuperPrompt AI")
    st.markdown("---")

    st.markdown("**Engine**")
    groq_model = st.selectbox(
        "Model",
        options=list(GROQ_MODELS.keys()),
        format_func=lambda x: GROQ_MODELS[x],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.groq_model = groq_model

    st.markdown("---")

    st.markdown("**Mode**")
    mode = st.radio(
        "Mode",
        options=["interview", "quick"],
        format_func=lambda x: "Interview (Best Quality)" if x == "interview" else "Quick (Instant)",
        label_visibility="collapsed"
    )
    st.session_state.mode = mode

    st.markdown("---")

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-pill">
            <div class="stat-num">{st.session_state.prompts_generated}</div>
            <div class="stat-lbl">Created</div>
        </div>
        <div class="stat-pill">
            <div class="stat-num">{len(st.session_state.history)}</div>
            <div class="stat-lbl">Saved</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Reset", use_container_width=True):
        for k in list(defaults.keys()):
            st.session_state[k] = defaults[k]
        st.rerun()

    st.markdown("---")
    st.markdown("---")
    theme_label = "Switch to Light Mode" if st.session_state.theme == "dark" else "Switch to Dark Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
    st.caption("Built by Don Mathew")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">⚡ SuperPrompt AI</div>
    <div class="page-subtitle">Transform your ideas into perfectly optimized AI prompts</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Generate", "History", "Score", "Templates"])

# ═══════════════════════════════════════════════
# TAB 1 — GENERATE
# ═══════════════════════════════════════════════
with tab1:

    # Progress indicator
    step = st.session_state.step
    if st.session_state.mode == "interview":
        s1 = "done" if step > 1 else ("active" if step == 1 else "pending")
        s2 = "done" if step > 2 else ("active" if step == 2 else "pending")
        s3 = "active" if step == 3 else ("done" if step > 3 else "pending")
        st.markdown(f"""
        <div class="steps-row">
            <div class="step-item">
                <div class="step-circle {s1}">{"✓" if step > 1 else "1"}</div>
                <span class="step-label {s1}">Your Idea</span>
            </div>
            <div class="step-connector {'done' if step > 1 else ''}"></div>
            <div class="step-item">
                <div class="step-circle {s2}">{"✓" if step > 2 else "2"}</div>
                <span class="step-label {s2}">Questions</span>
            </div>
            <div class="step-connector {'done' if step > 2 else ''}"></div>
            <div class="step-item">
                <div class="step-circle {s3}">{"✓" if step > 3 else "3"}</div>
                <span class="step-label {s3}">Super Prompt</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── STEP 1: Target Model ──────────────────
    st.markdown('<div class="section-header">Choose Target AI</div>', unsafe_allow_html=True)
    st.caption("Which AI will you use this prompt with?")

    cols = st.columns(4)
    model_list = list(TARGET_MODELS.keys())
    for i, model_name in enumerate(model_list):
        info = TARGET_MODELS[model_name]
        col = cols[i % 4]
        with col:
            is_sel = st.session_state.target_model == model_name
            active_class = "active" if is_sel else ""
            check = "✓ " if is_sel else ""
            st.markdown(f"""
            <div class="model-pill {active_class}">
                <div class="icon">{info['icon']}</div>
                <div class="name">{check}{model_name}</div>
                <div class="style-tag">{info['style']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Select", key=f"sel_{i}", use_container_width=True):
                st.session_state.target_model = model_name
                st.session_state.step = 1
                st.session_state.mc_questions = []
                st.session_state.mc_answers = {}
                st.session_state.custom_answers = {}
                st.session_state.final_prompt = ""
                st.rerun()

    if st.session_state.target_model:
        info = TARGET_MODELS[st.session_state.target_model]
        st.markdown(f'<div class="tip-box">💡 <strong>{st.session_state.target_model}:</strong> {info["tips"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── STEP 2: Idea ─────────────────────────
    st.markdown('<div class="section-header">Describe Your Idea</div>', unsafe_allow_html=True)
    st.caption("What do you want the AI to help you with?")

    idea = st.text_area(
        "idea",
        value=st.session_state.get("idea", ""),
        placeholder="e.g. Build a fitness tracking app with React and Node.js\ne.g. Write a viral LinkedIn post about AI productivity tips\ne.g. Create a customer support chatbot for my online store",
        height=100,
        label_visibility="collapsed"
    )

    c1, c2, c3 = st.columns([2, 1, 3])
    with c1:
        go_btn = st.button(
            "Start Interview →" if st.session_state.mode == "interview" else "Generate Prompt →",
            type="primary",
            use_container_width=True
        )
    with c2:
        if st.session_state.final_prompt:
            if st.button("Start Over", use_container_width=True):
                for k in ["step", "idea", "mc_questions", "mc_answers", "custom_answers", "final_prompt"]:
                    st.session_state[k] = defaults[k]
                st.rerun()

    if go_btn:
        if not st.session_state.target_model:
            st.error("Please select a target AI model first.")
        elif not idea.strip():
            st.error("Please describe your idea.")
        else:
            st.session_state.idea = idea
            if st.session_state.mode == "quick":
                st.session_state.step = 3
                messages = generate_quick_prompt(idea, st.session_state.target_model, st.session_state.groq_model)
                with st.spinner("Generating your Super Prompt..."):
                    result = ""
                    placeholder = st.empty()
                    for token in stream_groq(messages, st.session_state.groq_model, 3000):
                        result += token
                        placeholder.code(result + "▌", language="markdown")
                    st.session_state.final_prompt = result
                    st.session_state.prompts_generated += 1
                    placeholder.empty()
                st.rerun()
            else:
                st.session_state.step = 2
                with st.spinner("Preparing your questions..."):
                    questions = generate_mc_questions(idea, st.session_state.target_model, st.session_state.groq_model)
                    st.session_state.mc_questions = questions
                    st.session_state.mc_answers = {}
                    st.session_state.custom_answers = {}
                st.rerun()

    # ── STEP 3: Multiple Choice Questions ────
    if st.session_state.mode == "interview" and st.session_state.mc_questions:
        st.markdown("---")
        st.markdown('<div class="section-header">Answer These Questions</div>', unsafe_allow_html=True)
        st.caption("Select the best option for each question — or type a custom answer.")

        all_answered = True

        for idx, q in enumerate(st.session_state.mc_questions):
            q_num = idx + 1
            q_text = q.get("question", f"Question {q_num}")
            options = q.get("options", [])
            allow_custom = q.get("allow_custom", True)
            current_answer = st.session_state.mc_answers.get(idx, None)

            st.markdown(f"""
            <div class="q-card">
                <div class="q-number">Question {q_num} of {len(st.session_state.mc_questions)}</div>
                <div class="q-text">{q_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # Option buttons
            opt_cols = st.columns(2)
            for oi, option in enumerate(options):
                col = opt_cols[oi % 2]
                with col:
                    is_selected = current_answer == option
                    btn_label = f"✓ {option}" if is_selected else option
                    if st.button(btn_label, key=f"opt_{idx}_{oi}", use_container_width=True):
                        st.session_state.mc_answers[idx] = option
                        if idx in st.session_state.custom_answers:
                            del st.session_state.custom_answers[idx]
                        st.rerun()

            # Custom answer option
            if allow_custom:
                custom_val = st.session_state.custom_answers.get(idx, "")
                custom_input = st.text_input(
                    "Or type a custom answer:",
                    value=custom_val,
                    key=f"custom_{idx}",
                    placeholder="Type your own answer here...",
                )
                if custom_input and custom_input != custom_val:
                    st.session_state.custom_answers[idx] = custom_input
                    st.session_state.mc_answers[idx] = custom_input

            # Check if answered
            if idx not in st.session_state.mc_answers and not st.session_state.custom_answers.get(idx):
                all_answered = False

            st.markdown("")

        # Show answer summary
        if st.session_state.mc_answers:
            answered_count = len(st.session_state.mc_answers)
            total = len(st.session_state.mc_questions)
            st.caption(f"Answered {answered_count}/{total} questions")

        # Build button
        answered_count = len(st.session_state.mc_answers)
        if answered_count > 0:
            build_btn = st.button(
                f"Build Super Prompt ({answered_count}/{len(st.session_state.mc_questions)} answered) →",
                type="primary"
            )
            if build_btn:
                # Format answers as text
                answers_text = ""
                for idx, q in enumerate(st.session_state.mc_questions):
                    ans = st.session_state.mc_answers.get(idx, "Not answered")
                    answers_text += f"Q{idx+1}: {q.get('question', '')}\nA: {ans}\n\n"

                st.session_state.step = 3
                messages = generate_super_prompt(
                    st.session_state.idea,
                    st.session_state.target_model,
                    answers_text,
                    st.session_state.groq_model
                )
                with st.spinner("Building your Super Prompt..."):
                    result = ""
                    placeholder = st.empty()
                    for token in stream_groq(messages, st.session_state.groq_model, 3000):
                        result += token
                        placeholder.code(result + "▌", language="markdown")
                    st.session_state.final_prompt = result
                    st.session_state.prompts_generated += 1
                    placeholder.empty()
                st.rerun()

    # ── FINAL OUTPUT ─────────────────────────
    if st.session_state.final_prompt:
        st.markdown("---")
        target = st.session_state.target_model or "AI"
        st.markdown(f"### Your Super Prompt")
        st.markdown(f'<div class="success-box">✅ Ready to use! Copy and paste this directly into {target}.</div>', unsafe_allow_html=True)

        st.code(st.session_state.final_prompt, language="markdown")

        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        with c1:
            st.download_button("Download .txt", data=st.session_state.final_prompt, file_name="super_prompt.txt", mime="text/plain", use_container_width=True)
        with c2:
            ext = ".cursorrules" if "Cursor" in (st.session_state.target_model or "") else ".md"
            fname = ".cursorrules" if ext == ".cursorrules" else "super_prompt.md"
            st.download_button(f"Download {ext}", data=st.session_state.final_prompt, file_name=fname, mime="text/plain", use_container_width=True)
        with c3:
            if st.button("Save", use_container_width=True):
                st.session_state.history.append({
                    "idea": st.session_state.idea,
                    "target": st.session_state.target_model,
                    "prompt": st.session_state.final_prompt,
                    "time": time.strftime("%H:%M")
                })
                st.success("Saved!")

        st.markdown("---")
        st.markdown("**Refine this prompt**")
        refine = st.text_input("", placeholder="e.g. Make it more concise, add examples, focus on mobile...", label_visibility="collapsed")
        if st.button("Refine →"):
            if refine.strip():
                msgs = [
                    {"role": "system", "content": f"You are an expert prompt engineer. Refine this Super Prompt for {st.session_state.target_model} based on the user's request. Return ONLY the complete refined prompt."},
                    {"role": "user", "content": f"Prompt:\n\n{st.session_state.final_prompt}\n\nChange: {refine}"}
                ]
                with st.spinner("Refining..."):
                    refined = ask_groq(msgs, st.session_state.groq_model, 3000)
                    st.session_state.final_prompt = refined
                    st.session_state.prompts_generated += 1
                st.rerun()

# ═══════════════════════════════════════════════
# TAB 2 — HISTORY
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("### Prompt History")

    if not st.session_state.history:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#9ca3af;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">📂</div>
            <div style="font-weight:600;color:#6b7280;">No prompts saved yet</div>
            <div style="font-size:0.82rem;margin-top:4px;">Generate a prompt and click Save</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c1, c2 = st.columns([4, 1])
        with c2:
            if st.button("Clear All"):
                st.session_state.history = []
                st.rerun()
        for i, item in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - i
            with st.expander(f"#{idx} · {item['target']} · {item['idea'][:45]}... · {item['time']}"):
                st.caption(f"**Idea:** {item['idea']}")
                st.code(item['prompt'], language="markdown")
                st.download_button("Download", data=item['prompt'], file_name=f"prompt_{idx}.txt", mime="text/plain", key=f"dl_{i}")

# ═══════════════════════════════════════════════
# TAB 3 — SCORE
# ═══════════════════════════════════════════════
with tab3:
    st.markdown("### Score My Prompt")
    st.caption("Paste any prompt and get an expert AI review with scores, strengths, weaknesses, and an improved version.")

    prompt_to_score = st.text_area(
        "prompt",
        placeholder="Paste your prompt here...",
        height=180,
        label_visibility="collapsed"
    )

    if st.button("Analyze Prompt →", type="primary"):
        if not prompt_to_score.strip():
            st.error("Please paste a prompt first.")
        else:
            msgs = [
                {"role": "system", "content": """You are an expert prompt engineer. Score and analyze the given prompt.

Return your analysis in this EXACT format:

SCORES:
- Clarity: X/10
- Structure: X/10
- Specificity: X/10
- Constraints: X/10
- Output Format: X/10
- Overall: X/10

STRENGTHS:
• [Strength 1]
• [Strength 2]
• [Strength 3]

WEAKNESSES:
• [Weakness 1]
• [Weakness 2]
• [Weakness 3]

IMPROVED VERSION:
[Complete improved prompt here]"""},
                {"role": "user", "content": f"Score and analyze this prompt:\n\n{prompt_to_score}"}
            ]
            with st.spinner("Analyzing..."):
                result = ask_groq(msgs, st.session_state.groq_model, 2500)

            st.markdown("---")

            # Parse and display scores nicely
            lines = result.split("\n")
            scores = {}
            for line in lines:
                if "/" in line and ":" in line and any(x in line for x in ["Clarity", "Structure", "Specificity", "Constraints", "Output", "Overall"]):
                    try:
                        label = line.split(":")[0].strip("- •").strip()
                        score = line.split(":")[1].strip().split("/")[0].strip()
                        scores[label] = int(score)
                    except:
                        pass

            if scores:
                overall = scores.get("Overall", 0)
                color = "#059669" if overall >= 7 else ("#d97706" if overall >= 5 else "#dc2626")
                st.markdown(f"""
                <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:12px;padding:20px;margin-bottom:1rem;">
                    <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
                        <div style="font-size:3rem;font-weight:800;color:{color};">{overall}/10</div>
                        <div>
                            <div style="font-size:1rem;font-weight:600;color:#111827;">Overall Score</div>
                            <div style="font-size:0.8rem;color:#6b7280;">{"Excellent prompt!" if overall >= 8 else "Good, needs minor improvements" if overall >= 6 else "Needs significant improvements"}</div>
                        </div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
                """, unsafe_allow_html=True)

                for label, score in scores.items():
                    if label != "Overall":
                        bar_color = "#7c3aed" if score >= 7 else "#f59e0b" if score >= 5 else "#dc2626"
                        st.markdown(f"""
                        <div style="text-align:center;background:white;border:1px solid #e5e7eb;border-radius:8px;padding:10px;">
                            <div style="font-size:1.3rem;font-weight:700;color:{bar_color};">{score}</div>
                            <div style="font-size:0.65rem;color:#9ca3af;margin-top:2px;">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("</div></div>", unsafe_allow_html=True)

            st.markdown(result)

            st.download_button(
                "Download Report",
                data=result,
                file_name="prompt_score_report.txt",
                mime="text/plain"
            )

# ═══════════════════════════════════════════════
# TAB 4 — TEMPLATES
# ═══════════════════════════════════════════════
with tab4:
    st.markdown("### Templates")
    st.caption("Click any template to load it into the Generate tab instantly.")

    for category, items in TEMPLATES.items():
        st.markdown(f"#### {category}")
        cols = st.columns(3)
        for i, (name, idea) in enumerate(items):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="tmpl-card">
                    <div class="tmpl-name">{name}</div>
                    <div class="tmpl-desc">{idea[:70]}...</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Use →", key=f"t_{category}_{i}", use_container_width=True):
                    st.session_state.idea = idea
                    st.session_state.step = 1
                    st.session_state.mc_questions = []
                    st.session_state.mc_answers = {}
                    st.session_state.custom_answers = {}
                    st.session_state.final_prompt = ""
                    st.success("Loaded! Switch to the Generate tab.")
        st.markdown("---")