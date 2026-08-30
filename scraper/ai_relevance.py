"""
Strict AI-relevance filter for the Nepal-AI corpus.

OpenAlex topic tags are too loose: a climatology or earthquake paper can
pick up a weak secondary 'computer science applications' topic and sneak
in. For topic-evolution of *AI research*, keep a paper only if the title
or abstract actually talks about AI methods (ML, DL, NLP, LLMs, neural
nets, CV, classical AI, etc.).

Application domain is allowed — 'deep learning for flood mapping' is AI
research. 'Strong-motion observations of the Gorkha earthquake' is not.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Strong method / field vocabulary. Intentionally excludes generic science
# words (classification, prediction, clustering, algorithm, gis, remote
# sensing) that fire on non-AI papers.
_AI_TERM_RE = re.compile(
    r"""
    \b(?:
        artificial\s+intelligence
        | explainable\s+ai
        | generative\s+ai
        | machine\s+learning
        | deep\s+learning
        | reinforcement\s+learning
        | transfer\s+learning
        | supervised\s+learning
        | unsupervised\s+learning
        | semi-?supervised\s+learning
        | self-?supervised
        | few-?shot\s+learning
        | zero-?shot
        | neural\s+network
        | artificial\s+neural
        | deep\s+neural
        | convolutional
        | recurrent\s+neural
        | multilayer\s+perceptron
        | back-?propagation
        | autoencoder
        | vision\s+transformer
        | transformer\s+(?:model|models|architecture|network|encoder|decoder)
        | large\s+language\s+models?
        | language\s+models?
        | natural\s+language\s+processing
        | named\s+entity
        | sentiment\s+analysis
        | question\s+answering
        | text\s+classification
        | topic\s+model(?:ing)?
        | latent\s+dirichlet
        | word2vec
        | word\s+embedding
        | computer\s+vision
        | object\s+detection
        | image\s+segmentation
        | semantic\s+segmentation
        | instance\s+segmentation
        | image\s+classification
        | face\s+recognition
        | speech\s+recognition
        | generative\s+adversarial
        | support\s+vector\s+machine
        | random\s+forest
        | gradient\s+boosting
        | decision\s+tree
        | k-?nearest\s+neighbou?rs
        | k-?means
        | genetic\s+algorithm
        | particle\s+swarm
        | swarm\s+intelligence
        | expert\s+system
        | fuzzy\s+(?:logic|inference)
        | multi-?agent
        | intelligent\s+agent
        | knowledge[- ]based\s+system
        | pytorch
        | tensorflow
        | scikit-?learn
        | hugging\s?face
        | bertopic
        | chatgpt
        | gpt-[234]
        | llama\s?\d
        | yolov?\d
        | resnet
        | densenet
        | efficientnet
        | vgg-?\d
        | unet
        | u-net
        | xgboost
        | lightgbm
        | catboost
        | keras
        | \bcnn\b
        | \brnn\b
        | \blstm\b
        | \bgru\b
        | \bnlp\b
        | \bllm\b
        | \bgans?\b
        | \bsvm\b
        | \bbert\b
        | \bxai\b
        | \bmlp\b
        | \bdnn\b
    )
    \b
    |
    \b(?:ai-based|ai-driven|ai-powered|ai-assisted|ai-enabled)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Whole-word AI / ML as a last-resort title signal (not used on abstracts
# alone — too many false hits in running text).
_TITLE_AI_ABBREV_RE = re.compile(r"(?<![A-Za-z])(?:AI|ML)(?![A-Za-z])")


def is_ai_paper(title, abstract=""):
    """True if title or abstract uses AI-method vocabulary."""
    title = title or ""
    abstract = abstract or ""
    blob = f"{title} {abstract}"
    if _AI_TERM_RE.search(blob):
        return True
    if _TITLE_AI_ABBREV_RE.search(title):
        return True
    return False


def filter_ai_papers(papers):
    kept = []
    dropped = 0
    for paper in papers:
        if is_ai_paper(paper.get("title"), paper.get("abstract")):
            kept.append(paper)
        else:
            dropped += 1
    logger.info("AI-method filter: %d kept, %d dropped (no AI vocabulary in title/abstract)", len(kept), dropped)
    return kept
