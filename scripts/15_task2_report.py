"""Derive the official CEC result table, rank tests and figures from the saved run."""
import json, importlib.util, sys, shutil
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/'assistive-care-flc'
sys.path.insert(0,str(BASE/'src'))

def main():
    d=json.loads((BASE/'results/part3_results.json').read_text())
    from acflc.statistics import benchmark_statistics
    stats=benchmark_statistics(d)
    pairs=stats["comparisons"]
    (BASE/'results/part3_stats.json').write_text(json.dumps(stats,indent=2))
    def fmt(x):
        if abs(x)>=100000: return '$'+f'{x:.3e}'.replace('e+0',r'\times10^{').replace('e+',r'\times10^{')+'}$'
        return f'{x:.6f}'
    lines=[r'\begin{table*}[t]',r'\caption{Official CEC2005 instances: final objective values over 15 runs per cell. Lower is better. Standard deviation uses $n-1$. Optima are 390 (F6) and $-330$ (F9).}',r'\label{tab:t2-cec}\centering\small',r'\begin{tabular}{@{}llrrrrr@{}}\toprule',r'Function & $D$ & Algorithm & Mean & Std. dev. & Best & Worst\\\midrule']
    for i,c in enumerate(d['cells']):
        if i and i%3==0: lines.append(r'\midrule')
        lines.append(f"{c['function']} & {c['dim']} & {c['algorithm']} & "+' & '.join(fmt(c[k]) for k in ['mean','std','best','worst'])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table*}']
    (ROOT/'paper/latex/task2_cec_table.tex').write_text('\n'.join(lines)+'\n')
    # Existing distribution/landscape functions now consume the official inputs.
    path=BASE/'scripts/make_report_figures.py'; spec=importlib.util.spec_from_file_location('reportplots',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.fig18_benchmark_distributions(); mod.fig21_benchmark_landscapes()
    for f in ['fig14_convergence_cec2005.png','fig18_benchmark_distributions.png','fig21_benchmark_landscapes.png']:
        shutil.copy2(BASE/'figures'/f,ROOT/'paper/latex/figures'/f)
    for p in pairs: print(p['function'],p['dim'],p['a'],p['b'],'p=',round(p['p'],6),'Holm=',round(p['p_holm'],6), 'medians',p['median_a'],p['median_b'])

if __name__=='__main__': main()
