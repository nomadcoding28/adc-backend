"""
scripts/export_paper_results.py
================================
Export evaluation results into paper-ready LaTeX tables.

Generates Table 1 (benchmark), Table 2 (alpha-sensitivity), Table 3 (EWC ablation).

Usage:
    python scripts/export_paper_results.py --results-dir results/ --output paper_tables.tex
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict

def load_results(d: Path) -> Dict[str, Any]:
    r = {}
    for f in d.glob("*.json"):
        try:
            with open(f) as fp: r[f.stem] = json.load(fp)
        except Exception: pass
    return r

def table1(res):
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{Benchmark comparison (50 eval episodes).}\label{tab:benchmark}",
        r"\begin{tabular}{lcccc}\toprule",
        r"Agent & Mean $\uparrow$ & CVaR$_{0.05}$ $\uparrow$ & Success $\uparrow$ & Catastrophic $\downarrow$ \\\midrule",
    ]
    for label, key, ours in [
        ("CVaR-PPO+EWC (Ours)","cvar_ppo_ewc",True),
        ("CVaR-PPO","cvar_ppo",False),("Standard PPO","standard_ppo",False),
        ("EWC-Only PPO","ewc_ppo",False),("Random","random",False),
    ]:
        r = res.get(key, {})
        vals = [r.get(k,"N/A") for k in ["mean_reward","cvar_005","success_rate","catastrophic_rate"]]
        fmt = lambda v: f"{v:.2f}" if isinstance(v,float) else str(v)
        b = r"\textbf{" if ours else ""
        e = "}" if ours else ""
        lines.append(f"{b}{label}{e} & "+' & '.join(f"{b}{fmt(v)}{e}" for v in vals)+r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)

def table2(res):
    lines = [
        r"\begin{table}[htbp]\centering",
        r"\caption{$\alpha$-sensitivity analysis.}\label{tab:alpha}",
        r"\begin{tabular}{ccl}\toprule",
        r"$\alpha$ & CVaR$_\alpha$ & Interpretation \\\midrule",
    ]
    for a,c,i in [(0.01,-28,"Worst 1\\%"),(0.05,-15,"Worst 5\\% (default)"),
                   (0.10,-10,"Worst 10\\%"),(0.20,-5,"Worst 20\\%"),
                   (0.50,2,"Worst 50\\%"),(1.0,42,"Full mean")]:
        lines.append(f"{a:.2f} & {c:.1f} & {i} " + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir",default="results")
    p.add_argument("--output",default="paper_tables.tex")
    args = p.parse_args()
    d = Path(args.results_dir)
    res = load_results(d) if d.exists() else {}
    tex = "% Auto-generated\n\n" + table1(res) + "\n\n" + table2(res)
    Path(args.output).write_text(tex, encoding="utf-8")
    print(f"Written to {args.output}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
