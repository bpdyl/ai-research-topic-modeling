"""Additional-seed stability of the three fixed LDA configurations.

The seeds 100–109 were not used in the archived search objective. This measures
initialisation sensitivity, not repeated search performance or held-out accuracy.
"""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gensim.models import LdaModel
from nrtm.models.lda import fit_lda, topic_top_words
from nrtm.evaluation.stability import matched_jaccard
from itertools import combinations

def main():
    run=ROOT/'results/runs/20260915_seed_validation'
    run.mkdir(exist_ok=True)
    dictionary=LdaModel.load(str(ROOT/'results/runs/20260905_220137_ga_lda/ga_lda_model')).id2word
    tokens=[json.loads(x)['tokens'] for x in (ROOT/'data/processed/tokens.jsonl').read_text(encoding='utf-8').splitlines()]
    corpus=[dictionary.doc2bow(t) for t in tokens]
    seeds=list(range(100,110))
    specs={'Standard LDA':(10,'symmetric',None), 'GA-Optimized LDA':(7,.154521,.387562), 'Random-Search LDA':(8,.0287,.646904)}
    for name,folder,filename in [('GA-Optimized LDA','20260905_220137_ga_lda','ga_lda.json'),('Random-Search LDA','20260906_112958_random_search','random_search.json')]:
        p=json.loads((ROOT/'results/runs'/folder/filename).read_text()); g=p['best_genome']; specs[name]=(g['k'],g['alpha'],g['eta'])
    out={'seeds':seeds,'top_n':10,'fit_settings':{'passes':10,'iterations':100,'chunksize':500},'scope':'fixed configurations, same corpus, unused initialisation seeds; not held-out evaluation','models':{}}
    for name,(k,alpha,eta) in specs.items():
        topics=[]
        for seed in seeds:
            model=fit_lda(corpus,dictionary,k,seed=seed,alpha=alpha,eta=eta,**out['fit_settings'])
            topics.append(topic_top_words(model,10)); print(name,seed,flush=True)
        scores=[matched_jaccard(a,b,10) for a,b in combinations(topics,2)]
        out['models'][name]={'k':k,'alpha':alpha,'eta':eta,'mean_matched_jaccard':float(np.mean(scores)),'pairwise_range':[float(min(scores)),float(max(scores))],'pairwise':scores,'topics_by_seed':dict(zip(map(str,seeds),topics))}
        (run/'stability_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print({k:v['mean_matched_jaccard'] for k,v in out['models'].items()},flush=True)

if __name__=='__main__': main()
