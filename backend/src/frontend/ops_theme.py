"""Purple/blue/black theme for the Agent Ops dashboard tab only.

Scoped under `.ops-root` so it layers on top of (and doesn't fight with)
the app-wide dark theme in styles.py - the rest of the app keeps its blue
accent; this tab gets its own violet/indigo palette to read as a distinct
"internals / telemetry" surface.
"""
from __future__ import annotations

OPS_CSS = """
<style>
:root {
  --ops-bg: #0a0714;
  --ops-panel: #120c24;
  --ops-card: #17102e;
  --ops-card-raised: #1e1640;
  --ops-border: #34255c;
  --ops-border-soft: #241a45;
  --ops-text: #ece9f7;
  --ops-text-dim: #ab9fd4;
  --ops-text-faint: #6c5f96;
  --ops-violet: #a78bfa;
  --ops-violet-soft: rgba(167, 139, 250, 0.14);
  --ops-blue: #60a5fa;
  --ops-blue-soft: rgba(96, 165, 250, 0.14);
  --ops-cyan: #67e8f9;
  --ops-green: #34d399;
  --ops-amber: #fbbf24;
  --ops-red: #f87171;
}

.ops-root { color: var(--ops-text); }

.ops-badge-replay {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--ops-violet); background: var(--ops-violet-soft);
  border: 1px solid var(--ops-border); border-radius: 999px; padding: 4px 11px 4px 8px;
}
.ops-badge-replay .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ops-violet); box-shadow: 0 0 6px rgba(167,139,250,0.8); }

.ops-header {
  border: 1px solid var(--ops-border); border-radius: 12px;
  background: linear-gradient(135deg, rgba(96,165,250,0.10), rgba(10,7,20,0.2) 60%), var(--ops-panel);
  padding: 1.4rem 1.6rem; margin-bottom: 1.1rem;
}
.ops-header .title { font-size: 1.3rem; font-weight: 800; color: var(--ops-text); margin: 6px 0 4px 0; }
.ops-header .sub { color: var(--ops-text-dim); font-size: 0.85rem; }

.ops-card {
  border: 1px solid var(--ops-border-soft); border-radius: 10px;
  background: var(--ops-card); padding: 14px 16px; margin-bottom: 12px;
}
.ops-card-title {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ops-text-faint); margin-bottom: 10px;
}

.ops-flow-row { display: flex; align-items: stretch; gap: 6px; overflow-x: auto; padding-bottom: 4px; }
.ops-flow-node {
  flex: 1 0 128px; border: 1px solid var(--ops-border); border-radius: 10px;
  background: var(--ops-card-raised); padding: 10px 12px; position: relative;
}
.ops-flow-node.active { border-color: var(--ops-violet); box-shadow: 0 0 0 1px var(--ops-violet) inset, 0 0 14px rgba(167,139,250,0.35); }
.ops-flow-node.skipped { opacity: 0.45; }
.ops-flow-node .n-name { font-weight: 700; font-size: 0.82rem; color: var(--ops-text); }
.ops-flow-node .n-meta { font-size: 0.68rem; color: var(--ops-text-faint); margin-top: 4px; font-family: "SFMono-Regular", Consolas, monospace; }
.ops-flow-node .n-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.ops-flow-node .n-dot.on { background: var(--ops-violet); box-shadow: 0 0 6px rgba(167,139,250,0.8); }
.ops-flow-node .n-dot.off { background: var(--ops-text-faint); }
.ops-flow-arrow { align-self: center; color: var(--ops-text-faint); font-size: 1rem; flex-shrink: 0; }

.ops-metric-row { display: flex; gap: 10px; flex-wrap: wrap; }
.ops-metric {
  flex: 1 0 140px; border: 1px solid var(--ops-border-soft); border-radius: 10px;
  background: var(--ops-card-raised); padding: 10px 12px;
}
.ops-metric .m-label { font-size: 0.66rem; color: var(--ops-text-faint); text-transform: uppercase; letter-spacing: 0.08em; }
.ops-metric .m-value { font-size: 1.2rem; font-weight: 800; color: var(--ops-text); font-family: "SFMono-Regular", Consolas, monospace; margin-top: 2px; }
.ops-metric .m-sub { font-size: 0.7rem; color: var(--ops-text-dim); margin-top: 2px; }
.ops-metric.accent-blue .m-value { color: var(--ops-blue); }
.ops-metric.accent-violet .m-value { color: var(--ops-violet); }
.ops-metric.accent-cyan .m-value { color: var(--ops-cyan); }

.ops-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.ops-table th { text-align: left; color: var(--ops-text-faint); font-weight: 700; text-transform: uppercase; font-size: 0.66rem; letter-spacing: 0.06em; padding: 6px 8px; border-bottom: 1px solid var(--ops-border); }
.ops-table td { padding: 6px 8px; border-bottom: 1px solid var(--ops-border-soft); color: var(--ops-text-dim); font-family: "SFMono-Regular", Consolas, monospace; }
.ops-table tr:hover td { background: var(--ops-card-raised); }

.ops-pill { display: inline-flex; align-items: center; gap: 5px; font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
.ops-pill.pass { background: rgba(52,211,153,0.14); color: var(--ops-green); }
.ops-pill.fail { background: rgba(248,113,113,0.14); color: var(--ops-red); }
.ops-pill.mock { background: var(--ops-blue-soft); color: var(--ops-blue); }
.ops-pill.live { background: var(--ops-violet-soft); color: var(--ops-violet); }

.ops-bar-track { background: var(--ops-border-soft); border-radius: 6px; height: 8px; overflow: hidden; }
.ops-bar-fill { height: 100%; background: linear-gradient(90deg, var(--ops-blue), var(--ops-violet)); }

.ops-kv { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid var(--ops-border-soft); font-size: 0.8rem; }
.ops-kv:last-child { border-bottom: none; }
.ops-kv .k { color: var(--ops-text-faint); }
.ops-kv .v { color: var(--ops-text); font-family: "SFMono-Regular", Consolas, monospace; font-weight: 600; }
</style>
"""
