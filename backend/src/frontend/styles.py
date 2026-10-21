"""Custom CSS theme for the Streamlit shell.

Streamlit doesn't let injected HTML truly *wrap* native widgets, so the
approach here is: use `st.container(border=True)` for real card boundaries
(native, real DOM nesting) and restyle those containers globally via CSS,
then use small inline-HTML spans/badges *within single st.markdown calls*
for typographic detail. No external font/CDN requests - system font
stacks only, so the UI still renders correctly fully offline.
"""
from __future__ import annotations

CSS = """
<style>
:root {
  --bg: #0b0e13;
  --bg-panel: #10141b;
  --bg-card: #151a22;
  --bg-card-raised: #1a2029;
  --border: #262d3a;
  --border-soft: #1c222c;
  --text: #e6e9ef;
  --text-dim: #9aa4b2;
  --text-faint: #5b6472;
  --accent: #38bdf8;
  --accent-soft: rgba(56, 189, 248, 0.12);
  --amber: #f59e0b;
  --amber-soft: rgba(245, 158, 11, 0.14);
  --green: #22c55e;
  --green-soft: rgba(34, 197, 94, 0.12);
  --red: #ef4444;
  --red-soft: rgba(239, 68, 68, 0.12);
  --radius: 10px;
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

.stApp { background: var(--bg); }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
html, body, [class*="css"] { font-family: var(--font-sans); }

/* Kill Streamlit's default chrome bits we don't want in the demo view */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* Bordered containers -> our card system */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: var(--radius); }

/* Buttons */
.stButton > button {
  background: var(--bg-card-raised);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-weight: 500;
  transition: border-color 120ms ease, background 120ms ease;
}
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] {
  background: var(--accent);
  color: #05131c;
  border: 1px solid var(--accent);
  font-weight: 600;
}
.stButton > button[kind="primary"]:hover { background: #5bc8fb; color: #05131c; }

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
  background: var(--bg-card-raised) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}

/* Tabs -> segmented top nav */
button[data-baseweb="tab"] {
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--text-dim);
}
button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent); }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
div[data-baseweb="tab-border"] { background-color: var(--border) !important; }

/* Progress bar (used for recommendation confidence) */
.stProgress > div > div > div > div { background: var(--accent); }

/* Metrics */
div[data-testid="stMetric"] {
  background: var(--bg-card-raised);
  border: 1px solid var(--border-soft);
  border-radius: 8px;
  padding: 0.6rem 0.8rem;
}
div[data-testid="stMetricLabel"] { color: var(--text-faint); }

/* Expander (developer drawer) */
details {
  background: var(--bg-panel);
  border: 1px solid var(--border-soft);
  border-radius: 8px;
}

/* ---- Custom component classes ---- */

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.9rem 1.1rem;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 1.1rem;
}
.app-header .brand { display: flex; flex-direction: column; gap: 2px; }
.app-header .brand-name { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.01em; color: var(--text); }
.app-header .brand-tag { font-size: 0.78rem; color: var(--text-faint); }
.app-header .status-row { display: flex; gap: 0.9rem; align-items: center; flex-wrap: wrap; }

.status-pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.76rem; font-family: var(--font-mono);
  color: var(--text-dim); padding: 3px 9px 3px 7px;
  border: 1px solid var(--border-soft); border-radius: 999px;
  background: var(--bg-card);
}
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot-green { background: var(--green); box-shadow: 0 0 6px rgba(34,197,94,0.6); }
.dot-red { background: var(--red); box-shadow: 0 0 6px rgba(239,68,68,0.6); }
.dot-amber { background: var(--amber); box-shadow: 0 0 6px rgba(245,158,11,0.6); }
.dot-gray { background: var(--text-faint); }

.section-label {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
  color: var(--text-faint); text-transform: uppercase; margin-bottom: 0.5rem;
}

.evidence-tag {
  display: inline-block; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
  padding: 2px 7px; border-radius: 4px; text-transform: uppercase; margin-bottom: 6px;
}
.tag-evidence { background: var(--accent-soft); color: var(--accent); }
.tag-inference { background: var(--amber-soft); color: var(--amber); }

.kv-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid var(--border-soft); font-size: 0.86rem; }
.kv-row:last-child { border-bottom: none; }
.kv-key { color: var(--text-faint); }
.kv-val { color: var(--text); font-family: var(--font-mono); font-weight: 600; text-align: right; }

.badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 999px;
}
.badge-pass { background: var(--green-soft); color: var(--green); }
.badge-fail { background: var(--red-soft); color: var(--red); }
.badge-wait { background: var(--amber-soft); color: var(--amber); }
.badge-idle { background: var(--bg-card-raised); color: var(--text-faint); }

.stage-head { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.stage-num {
  font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700;
  color: var(--text-faint); border: 1px solid var(--border); border-radius: 5px;
  width: 22px; height: 22px; display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.stage-title { font-weight: 700; font-size: 0.92rem; letter-spacing: 0.03em; }
.stage-action { color: var(--text-dim); font-size: 0.82rem; margin: 4px 0 8px 32px; }
.stage-connector { width: 1px; height: 14px; background: var(--border); margin-left: 10px; }

.stage-complete .stage-num { border-color: var(--green); color: var(--green); background: var(--green-soft); }
.stage-running .stage-num { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); animation: pulse 1.4s ease-in-out infinite; }
.stage-blocked .stage-num { border-color: var(--amber); color: var(--amber); background: var(--amber-soft); }
.stage-failed .stage-num { border-color: var(--red); color: var(--red); background: var(--red-soft); }
.stage-pending .stage-num { color: var(--text-faint); }

@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }

.checkpoint-line { display: flex; justify-content: space-between; font-size: 0.78rem; padding: 2px 0 2px 32px; color: var(--text-dim); }
.checkpoint-line .mark-pass { color: var(--green); }
.checkpoint-line .mark-fail { color: var(--amber); }

.catalogue-card {
  border: 1px solid var(--border-soft); border-radius: 8px; padding: 10px 12px; margin-bottom: 8px; background: var(--bg-card-raised);
}
.catalogue-card.best { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
.catalogue-name { font-weight: 700; font-size: 0.9rem; }
.catalogue-meta { color: var(--text-faint); font-size: 0.76rem; margin-top: 2px; }
.catalogue-score { font-family: var(--font-mono); font-weight: 700; color: var(--accent); }

.compat-check { display: flex; justify-content: space-between; font-size: 0.8rem; padding: 3px 0; }
.compat-check .key { color: var(--text-dim); }

.clarify-card {
  border: 1px solid var(--amber); border-radius: var(--radius);
  background: var(--amber-soft); padding: 16px 18px; margin: 6px 0 10px 0;
}
.clarify-title { color: var(--amber); font-weight: 700; font-size: 0.85rem; letter-spacing: 0.05em; margin-bottom: 6px; }
.clarify-question { color: var(--text); font-size: 0.95rem; margin-bottom: 10px; }

.recommend-hero {
  border: 1px solid var(--accent); border-radius: var(--radius);
  background: linear-gradient(180deg, var(--accent-soft), var(--bg-card));
  padding: 20px 22px;
}
.recommend-label { color: var(--accent); font-weight: 700; font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; }
.recommend-name { font-size: 1.3rem; font-weight: 800; margin: 6px 0 10px 0; color: var(--text); }
.recommend-price { font-family: var(--font-mono); font-size: 1.05rem; color: var(--text); }

.empty-state { text-align: center; padding: 2.4rem 1.2rem; color: var(--text-dim); }
.empty-state .icon { font-size: 2rem; margin-bottom: 10px; }
.empty-state .title { font-weight: 700; color: var(--text); font-size: 1.05rem; margin-bottom: 6px; }
.empty-state .sub { font-size: 0.86rem; color: var(--text-faint); max-width: 420px; margin: 0 auto; }

.hero-panel {
  border: 1px solid var(--border); border-radius: var(--radius);
  background: radial-gradient(circle at 20% 0%, var(--accent-soft), transparent 55%), var(--bg-panel);
  padding: 2.2rem 2rem; margin-bottom: 1.2rem;
}
.hero-title { font-size: 1.5rem; font-weight: 800; color: var(--text); margin-bottom: 6px; }
.hero-sub { color: var(--text-dim); font-size: 0.95rem; max-width: 640px; line-height: 1.5; }

.provider-chip {
  display: inline-flex; align-items: center; gap: 6px; font-size: 0.74rem;
  font-family: var(--font-mono); color: var(--text-dim); background: var(--bg-card-raised);
  border: 1px solid var(--border-soft); border-radius: 6px; padding: 3px 8px;
}

/* ---- Agent state badge (command-center header of the AI panel) ---- */
.agent-state-badge {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; border-radius: 8px; margin-bottom: 10px;
  border: 1px solid var(--border); background: var(--bg-card-raised);
}
.agent-state-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.agent-state-label { font-weight: 700; font-size: 0.85rem; letter-spacing: 0.06em; }
.agent-state-idle .agent-state-dot { background: var(--text-faint); }
.agent-state-idle .agent-state-label { color: var(--text-faint); }
.agent-state-active .agent-state-dot { background: var(--accent); box-shadow: 0 0 8px rgba(56,189,248,0.7); animation: pulse 1.4s ease-in-out infinite; }
.agent-state-active .agent-state-label { color: var(--accent); }
.agent-state-wait .agent-state-dot { background: var(--amber); box-shadow: 0 0 8px rgba(245,158,11,0.7); animation: pulse 1.1s ease-in-out infinite; }
.agent-state-wait .agent-state-label { color: var(--amber); }
.agent-state-done .agent-state-dot { background: var(--green); }
.agent-state-done .agent-state-label { color: var(--green); }
.agent-state-error .agent-state-dot { background: var(--red); }
.agent-state-error .agent-state-label { color: var(--red); }

/* ---- Activity timeline (real AgentState.chat_history, timestamped) ---- */
.timeline-item { display: flex; gap: 10px; padding: 5px 0; border-bottom: 1px solid var(--border-soft); }
.timeline-item:last-child { border-bottom: none; }
.timeline-time { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-faint); flex-shrink: 0; padding-top: 1px; min-width: 58px; }
.timeline-text { font-size: 0.82rem; color: var(--text-dim); line-height: 1.4; }

/* ---- MVID visual candidates - visually distinct (teal) from catalogue (blue) ---- */
.mvid-card {
  border: 1px solid rgba(45, 212, 191, 0.35); border-radius: 8px; padding: 10px 12px;
  margin-bottom: 8px; background: var(--bg-card-raised);
}
.mvid-card.best { border-color: #2dd4bf; box-shadow: 0 0 0 1px #2dd4bf inset; }
.mvid-tag {
  display: inline-block; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.07em;
  color: #2dd4bf; background: rgba(45, 212, 191, 0.12); border-radius: 4px; padding: 2px 6px;
  text-transform: uppercase; margin-bottom: 4px;
}
.mvid-score { font-family: var(--font-mono); font-weight: 700; color: #2dd4bf; }

/* ---- Candidate fusion table ---- */
.fusion-row {
  display: grid; grid-template-columns: 1.6fr 0.9fr 0.9fr 0.9fr 0.9fr 1fr;
  gap: 8px; align-items: center; font-size: 0.78rem; padding: 7px 4px;
  border-bottom: 1px solid var(--border-soft);
}
.fusion-row.header { color: var(--text-faint); font-weight: 700; font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; }
.fusion-status { font-weight: 700; font-size: 0.7rem; padding: 2px 7px; border-radius: 999px; text-align: center; }
.fusion-status.exact { background: var(--green-soft); color: var(--green); }
.fusion-status.likely { background: var(--accent-soft); color: var(--accent); }
.fusion-status.incompatible { background: var(--red-soft); color: var(--red); }
.fusion-status.unverified { background: var(--bg-card-raised); color: var(--text-faint); border: 1px solid var(--border-soft); }

/* ---- Source panel ---- */
.source-verified { border-left: 3px solid var(--green); padding: 8px 12px; background: var(--green-soft); border-radius: 0 6px 6px 0; }
.source-unverified { border-left: 3px solid var(--text-faint); padding: 8px 12px; background: var(--bg-card-raised); border-radius: 0 6px 6px 0; color: var(--text-dim); font-size: 0.85rem; }

/* ---- Live/drone status panel ---- */
.live-status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; font-size: 0.78rem; }
.live-status-grid .k { color: var(--text-faint); }
.live-status-grid .v { font-family: var(--font-mono); font-weight: 600; }

/* ---- Small technical/benchmark note (not part of primary flow) ---- */
.benchmark-note {
  border: 1px dashed var(--border); border-radius: 8px; padding: 10px 12px;
  font-size: 0.76rem; color: var(--text-faint); background: var(--bg-panel);
}
.benchmark-note b { color: var(--text-dim); }
</style>
"""


def inject_theme(st) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
