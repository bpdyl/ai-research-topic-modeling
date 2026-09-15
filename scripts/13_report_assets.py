"""Rebuild Task 1 report figures from pinned runs, with auditable denominators.

No model training or new LLM judgments. PNG and vector PDF versions are emitted.
Run from any directory: python scripts/13_report_assets.py
"""
from pathlib import Path
import hashlib
import json
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nrtm.temporal.proportions import window_proportions

OUT = ROOT / 'paper/latex/figures'
RUNS = ROOT / 'results/runs'
PINNED = {'Standard LDA':'20260905_231154_lda_baseline',
          'GA-Optimized LDA':'20260905_220137_ga_lda',
          'Random-Search LDA':'20260906_112958_random_search',
          'BERTopic':'20260906_080709_bertopic'}
FILES = ['lda_baseline.json','ga_lda.json','random_search.json','bertopic.json']
DATA = {name:json.loads((RUNS/run/f).read_text(encoding='utf-8'))
        for (name,run),f in zip(PINNED.items(),FILES)}
LABELS = json.loads((ROOT/'data/processed/final_topic_labels.json').read_text(encoding='utf-8'))
PANEL = json.loads((ROOT/'data/processed/interpretability_panel.json').read_text(encoding='utf-8'))
DOCS = [json.loads(x) for x in (ROOT/'data/processed/corpus_frozen.jsonl').read_text(encoding='utf-8').splitlines()]
WINDOWS = ['2015-2017','2018-2020','2021-2023','2024-2025']
COLORS = ['#56697c','#b76336','#387c78','#665c91']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':140,'savefig.dpi':260,
                     'pdf.fonttype':42,'axes.titlesize':11})

def save(fig, name):
    for suffix in ['pdf','png']:
        fig.savefig(OUT/f'{name}.{suffix}',bbox_inches='tight',facecolor='white')
    plt.close(fig)

def pipeline():
    fig,ax=plt.subplots(figsize=(10,4.8)); ax.axis('off'); ax.set(xlim=(0,10),ylim=(0,5))
    boxes=[(0.1,3.4,2.0,1.1,'OpenAlex retrieval\n1,693 records'),
           (2.65,3.4,2.0,1.1,'Inclusion filters\n1,074 documents'),
           (5.3,3.4,4.4,1.1,'Shared title + abstract corpus\n2015–2025 · Nepal affiliation'),
           (1.4,1.8,3.4,1.0,'Token / phrase branch → BoW\nGrid LDA · GA-LDA · random LDA'),
           (5.4,1.8,3.4,1.0,'Natural-sentence branch\nEmbeddings → UMAP → HDBSCAN'),
           (1.4,.1,7.4,1.05,'Evaluation: coherence, diversity, stability, automated judgments\nTemporal reporting: fixed topics; explicit assigned/total counts')]
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',fc='#f1f4f5',ec='#86939c',lw=.8))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
    for a,b in [((2.15,3.95),(2.58,3.95)),((4.71,3.95),(5.24,3.95)),((6.4,3.33),(3.1,2.86)),((7.7,3.33),(7.1,2.86)),((3.1,1.73),(4.1,1.21)),((7.1,1.73),(6.1,1.21))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#586874'))
    save(fig,'figPipeline')

def search():
    ga=[json.loads(x) for x in (RUNS/PINNED['GA-Optimized LDA']/'fitness_trace.jsonl').read_text().splitlines()]
    unique=[r for r in ga if not r.get('cached',False)]
    rs=DATA['Random-Search LDA']['history']
    assert len(unique)==len(rs)==183
    fig,ax=plt.subplots(figsize=(7,3.6))
    ax.step(np.arange(1,184),np.maximum.accumulate([r['fitness'] for r in unique]),where='post',color=COLORS[1],label='GA-LDA')
    ax.step(np.arange(1,184),[r['best_so_far'] for r in rs],where='post',color=COLORS[2],label='Random-search LDA')
    ax.set(xlabel='Cumulative unique fitness evaluations',ylabel='Best scalarised fitness',title='Matched search budget: one run per strategy')
    ax.legend(loc='lower right',frameon=False); ax.grid(alpha=.15); save(fig,'figSearchBudget')

def quality():
    fig,ax=plt.subplots(figsize=(6.8,3.8))
    for i,(name,d) in enumerate(DATA.items()):
        x=d['metrics']['c_v']; y=PANEL['summary'][name]['mean']
        label=['Standard LDA','GA-LDA','Random search','BERTopic'][i]
        ax.scatter(x,y,s=65,color=COLORS[i],zorder=4)
        ax.annotate(label,(x,y),xytext=(7,3),textcoords='offset points',fontsize=9)
    ax.set(xlim=(.439,.541),ylim=(1.6,4.6),xlabel='Topic coherence Cᵥ',ylabel='Mean blinded LLM rating (1–5)',title='Coherence and automated interpretability')
    ax.grid(alpha=.15); save(fig,'figQualityTradeoff')

def label_audit():
    fig,ax=plt.subplots(figsize=(6.8,3.0))
    order=list(DATA); bottom=np.zeros(4)
    for verdict,color in [('accept','#387c78'),('revise','#c49a5a'),('reject','#ab5b57')]:
        values=np.array([PANEL['label_review']['by_model'][m].get(verdict,0) for m in order])
        ax.barh(range(4),values,left=bottom,label=verdict.title(),color=color,height=.55)
        for i,v in enumerate(values):
            if v: ax.text(bottom[i]+v/2,i,str(v),ha='center',va='center',color='white',fontsize=9)
        bottom+=values
    ax.set(yticks=range(4),yticklabels=['Standard LDA','GA-LDA','Random search','BERTopic'],xlabel='Number of generated topic labels',title='Separate automated label audit (39 labels)')
    ax.invert_yaxis(); ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.22)); save(fig,'figLabelAudit')

def topics():
    fig,axes=plt.subplots(1,3,figsize=(10,3.4))
    for ax,(name,k,title) in zip(axes,[('GA-Optimized LDA',5,'GA-LDA T5: generic evaluation'),('GA-Optimized LDA',4,'GA-LDA T4: education'),('BERTopic',11,'BERTopic T11: drug discovery')]):
        words=DATA[name]['topics'][k]
        ax.axis('off'); ax.set_title(title,loc='left',fontsize=10,pad=12)
        for j,w in enumerate(words):
            ax.text(.02,.95-j*.085,f'{j+1:2d}  {w.replace("_"," ")}',va='top',fontsize=10)
    save(fig,'figTopicExamples')

def temporal():
    evidence={}; rng=np.random.default_rng(20260915)
    for name,key in [('GA-Optimized LDA','GA'),('BERTopic','BT')]:
        dt=np.load(RUNS/PINNED[name]/'doc_topics.npy')
        assert dt.shape[0]==len(DOCS)
        if name=='BERTopic': assert DATA[name]['doc_ids']==[d['doc_id'] for d in DOCS]
        doc_windows=[d['time_window'] for d in DOCS]
        props,counts=window_proportions(dt,WINDOWS,doc_windows)
        assigned=[int(np.sum(dt[np.array(doc_windows)==w].sum(axis=1)>0)) for w in WINDOWS]
        labels=[f'T{i} '+LABELS[name][str(i)].replace(' (low coherence)',' [mixed]') for i in range(dt.shape[1])]
        fig,ax=plt.subplots(figsize=(10,4.6 if key=='GA' else 6.6))
        im=ax.imshow(props.T*100,aspect='auto',cmap='Blues',vmin=0,vmax=max(30,props.max()*100))
        ax.set(xticks=range(4),xticklabels=[f'{w}\nassigned {a}/{n}' for w,a,n in zip(WINDOWS,assigned,counts)],yticks=range(len(labels)),yticklabels=[textwrap.fill(x,43) for x in labels],xlabel='Publication window and assigned/total documents')
        # Transpose at creation so the image extent matches the topic rows.
        ax.set_xlim(-.5,3.5); ax.set_ylim(len(labels)-.5,-.5)
        for i in range(len(labels)):
            for j in range(4):
                v=props[j,i]*100; ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=9,color='white' if v>20 else '#26313b')
        ax.axvline(.5,color='#667581',lw=1.5,ls='--')
        ax.set_title(('GA-LDA: mean topic mass, all documents' if key=='GA' else 'BERTopic: share among assigned documents')+'\n2015–2017 is descriptive only (n=8)',loc='left',pad=12)
        fig.colorbar(im,ax=ax,pad=.02,label='Within-window share (%)',shrink=.7)
        save(fig,'figTemporal'+key)
        edu=4 if key=='GA' else 2
        samples=[]
        for w in WINDOWS[1::2]:
            block=dt[np.array(doc_windows)==w]; block=block[block.sum(axis=1)>0]
            vals=block[:,edu]/block.sum(axis=1)
            samples.append(np.array([rng.choice(vals,size=len(vals),replace=True).mean() for _ in range(2000)]))
        delta=samples[1]-samples[0]
        early=[{'id':d['doc_id'],'year':d['year'],'title':d['title'],'education_mass':round(float(v[edu]),5)}
               for d,v in zip(DOCS,dt) if d['year']<=2017]
        evidence[name]={'run':PINNED[name],'windows':WINDOWS,'n_total':counts,'n_assigned':assigned,'shares':props.tolist(),
                        'shares_all_documents':[(dt[np.array(doc_windows)==w].mean(axis=0)).tolist() for w in WINDOWS],
                        'labels':LABELS[name], 'education_delta_bootstrap_95pct':np.percentile(delta,[2.5,97.5]).tolist(),
                        'bootstrap_note':'2,000 resamples within each endpoint window, conditional on fitted model and assigned documents; 2018–2020 to 2024–2025',
                        'early_documents':early}
    return evidence

def coverage(evidence):
    d=evidence['BERTopic']; total=np.array(d['n_total']); assigned=np.array(d['n_assigned'])
    fig,axes=plt.subplots(1,2,figsize=(10,3.6)); x=np.arange(4)
    axes[0].bar(x,100*assigned/total,color=COLORS[3])
    axes[0].set(xticks=x,xticklabels=['2015–17','2018–20','2021–23','2024–25'],ylim=(0,100),ylabel='Documents assigned (%)',title='BERTopic assignment coverage')
    for i,(a,n) in enumerate(zip(assigned,total)): axes[0].text(i,100*a/n+2,f'{a}/{n}',ha='center',fontsize=9)
    axes[1].plot(x,np.array(d['shares'])[:,2]*100,'o-',color=COLORS[3],label='Among assigned documents')
    axes[1].plot(x,np.array(d['shares_all_documents'])[:,2]*100,'s--',color=COLORS[0],label='Among all retrieved documents')
    axes[1].set(xticks=x,xticklabels=['2015–17','2018–20','2021–23','2024–25'],ylabel='Education cluster share (%)',title='Effect of denominator choice')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes: ax.set_xlabel('Publication window'); ax.axvspan(-.45,.45,color='#e6e8eb',alpha=.4)
    fig.tight_layout(); save(fig,'figCoverageSensitivity')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pipeline(); search(); quality(); label_audit(); topics(); evidence=temporal(); coverage(evidence)
    inputs=[ROOT/'data/processed/final_topic_labels.json',ROOT/'data/processed/interpretability_panel.json',ROOT/'data/processed/corpus_frozen.jsonl']+[RUNS/r/f for r,f in zip(PINNED.values(),FILES)]
    inputs += [RUNS/PINNED[n]/'doc_topics.npy' for n in ['GA-Optimized LDA','BERTopic']]
    inputs += [RUNS/PINNED['GA-Optimized LDA']/'fitness_trace.jsonl']
    manifest={'pinned_runs':PINNED,'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'temporal':evidence}
    path=ROOT/'paper/report_evidence.json'; path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Wrote eight revised figure pairs and paper/report_evidence.json')
    for name,d in evidence.items(): print(name,d['n_assigned'],d['education_delta_bootstrap_95pct'])

if __name__=='__main__': main()
