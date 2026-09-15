"""Informational counts under the tutor's 2026-09-15 approval.

Task 1: up to 4,500 words. Task 2: no limit. References, mathematical
expressions and tabular data are excluded; titles and keywords are included.
"""
import re
from pathlib import Path
from check_report import expand, strip_comments
HERE=Path(__file__).resolve().parent

def count(text, captions=False):
    text=strip_comments(text)
    text=re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', ' ',text,flags=re.S)
    for env in ('equation','equation*','align','align*','tabular','lstlisting'):
        text=re.sub(r'\\begin\{'+re.escape(env)+r'\}.*?\\end\{'+re.escape(env)+r'\}', ' ',text,flags=re.S)
    if not captions:
        for env in ('figure','figure*','table','table*'):
            text=re.sub(r'\\begin\{'+re.escape(env)+r'\}.*?\\end\{'+re.escape(env)+r'\}', ' ',text,flags=re.S)
    text=re.sub(r'\\(?:includegraphics|lstinputlisting)(?:\[[^]]*\])?\{[^}]*\}', ' ',text)
    text=re.sub(r'\\(?:cite|ref|eqref|label|url|input)\{[^}]*\}', ' ',text)
    text=re.sub(r'\$[^$]*\$', ' ',text)
    text=re.sub(r'\\(?:begin|end)\{[^}]*\}', ' ',text)
    text=re.sub(r'\\[a-zA-Z@]+\*?(?:\[[^]]*\])?', ' ',text)
    text=re.sub(r'[{}~&\\]', ' ',text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*",text))

def main():
    for title,files in [('Task 1',['task1.tex']),('Task 2',['task2_head.tex','task2.tex','appendix_task2.tex'])]:
        text='\n'.join(expand(HERE/f) for f in files)
        print(f'{title}: {count(text):,} prose words; {count(text,True):,} including captions/headings (approximate)')
    print('Approved: Task 1 up to 4,500 words; Task 2 uncapped; two columns and extra pages.')
    print('Titles/keywords included; references, tabular data and mathematical expressions excluded.')

if __name__=='__main__': main()
