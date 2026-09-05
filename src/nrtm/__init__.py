"""
nrtm — Nepal Research Topic Modeling (ST7085CEM Task 1).

IMPORTANT: the BLAS thread caps below are applied at import time, *before*
numpy/scipy are pulled in by anything else. They must stay at the top of this
file.

Why: gensim's CoherenceModel and LdaMulticore spawn worker processes. On
Windows those use `spawn`, so each child re-imports the entire numpy/scipy/BLAS
stack and starts its own OpenBLAS thread pool. On a 16-core machine that is ~15
copies. During development this exhausted the system commit limit and failed
with `ImportError: DLL load failed while importing _flapack: The paging file is
too small for this operation to complete.`

Capping the thread pools here, and passing processes=1 to CoherenceModel
(see config/default.yaml: coherence_processes), keeps the GA's ~900 coherence
evaluations survivable. See ../../.brain/RISKS.md RISK-006 and RISK-015.
"""

import os as _os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_var, "1")

__version__ = "0.1.0"
