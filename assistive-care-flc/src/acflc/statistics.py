"""Distributional comparisons with family-wise Holm adjustment."""
from itertools import combinations
import numpy as np
from scipy.stats import mannwhitneyu

def benchmark_statistics(result):
    pairs=[]
    for function in result['protocol']['functions']:
        for dim in result['protocol']['dims']:
            sub={c['algorithm']:c for c in result['cells'] if c['function']==function and c['dim']==dim}
            for a,b in combinations(result['protocol']['algorithms'],2):
                u,p=mannwhitneyu(sub[a]['values'],sub[b]['values'],alternative='two-sided')
                pairs.append({'function':function,'dim':dim,'a':a,'b':b,'U':float(u),'p':float(p),
                              'median_a':float(np.median(sub[a]['values'])),'median_b':float(np.median(sub[b]['values']))})
    order=sorted(range(len(pairs)),key=lambda i:pairs[i]['p']); adjusted=0.
    for rank,i in enumerate(order):
        adjusted=max(adjusted,min(1.,(len(pairs)-rank)*pairs[i]['p']))
        pairs[i]['p_holm']=adjusted
        pairs[i]['significant_holm_0.05']=adjusted<.05
        pairs[i]['significant_at_0.05']=pairs[i]['p']<.05
        pairs[i]['lower_median']=pairs[i]['a'] if pairs[i]['median_a']<pairs[i]['median_b'] else pairs[i]['b']
    return {'test':'two-sided Mann-Whitney U; Holm correction across all pairwise cells',
            'note':result['protocol'].get('shift_note','see experiment protocol'),
            'comparisons':pairs}
