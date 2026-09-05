"""
Tests for cleaning and tokenisation.

Weighted towards acronym preservation, because that is an explicit proposal
commitment and the failure mode is silent: the acronyms simply vanish and
nobody notices until the topics look oddly generic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nrtm.preprocessing.clean import clean_text, normalise_unicode          # noqa: E402
from nrtm.preprocessing.tokenize import (                                    # noqa: E402
    tokenise_document, tokenise_documents, build_stopwords, lemmatise,
)

PROTECTED = frozenset(["ai", "ml", "nlp", "cnn", "gan", "llm", "cv"])


def _stops(domain=(), keep=PROTECTED):
    return build_stopwords(domain, keep=keep)


class TestClean:
    def test_strips_leading_abstract_prefix(self):
        assert clean_text("Abstract: We train a CNN.").startswith("We train")
        assert clean_text("ABSTRACT We train a CNN.").startswith("We train")
        assert clean_text("Abstract - We train.").startswith("We train")

    def test_does_not_strip_abstract_mid_sentence(self):
        out = clean_text("We present an abstract representation of graphs.")
        assert "abstract representation" in out

    def test_strips_copyright_tail(self):
        out = clean_text("We train a model. Copyright 2023 Elsevier Ltd.")
        assert "Elsevier" not in out
        assert "We train a model." in out

    def test_strips_urls_dois_and_emails(self):
        out = clean_text("See https://example.com/x and doi:10.1234/abc or a@b.com now")
        for fragment in ("https", "example.com", "10.1234", "a@b.com"):
            assert fragment not in out

    def test_strips_structured_headers(self):
        out = clean_text("Background: AI is growing. Methods: We used a CNN.")
        assert "Background:" not in out and "Methods:" not in out
        assert "AI is growing" in out and "We used a CNN" in out

    def test_normalises_curly_quotes_and_dashes(self):
        assert normalise_unicode("don’t") == "don't"
        assert normalise_unicode("2015–2025") == "2015-2025"

    def test_preserves_accented_characters(self):
        # Stripping to ASCII would corrupt author names and loanwords.
        assert "é" in normalise_unicode("café")

    def test_preserves_case_and_punctuation_for_bertopic(self):
        out = clean_text("Deep Learning improves accuracy. It works well!")
        assert "Deep Learning" in out and "." in out

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""


class TestAcronymPreservation:
    def test_short_acronyms_survive_the_length_filter(self):
        toks = tokenise_document("AI and ML transform CV research", _stops(), PROTECTED, 3)
        assert "ai" in toks and "ml" in toks and "cv" in toks

    def test_unprotected_short_words_are_removed(self):
        toks = tokenise_document("we go to it", _stops(), PROTECTED, 3)
        assert toks == []

    def test_plural_acronyms_fold_to_singular(self):
        toks = tokenise_document("Several CNNs and GANs were trained", _stops(), PROTECTED, 3)
        assert "cnn" in toks and "gan" in toks
        assert "cnns" not in toks and "gans" not in toks

    def test_acronyms_are_not_lemmatised(self):
        # 'ai' would otherwise be untouched by luck; 'gas'-like forms would not.
        toks = tokenise_document("LLM and NLP", _stops(), PROTECTED, 3)
        assert "llm" in toks and "nlp" in toks

    def test_domain_stopword_cannot_shadow_a_protected_acronym(self):
        stops = build_stopwords(domain_stopwords=["ai", "cv"], keep=PROTECTED)
        assert "ai" not in stops
        toks = tokenise_document("AI and CV", stops, PROTECTED, 3)
        assert "ai" in toks and "cv" in toks


class TestTokenise:
    def test_removes_english_stopwords(self):
        toks = tokenise_document("the model is trained on the data", _stops(), PROTECTED, 3)
        assert "the" not in toks and "is" not in toks

    def test_removes_domain_stopwords(self):
        stops = _stops(domain=["paper", "propose"])
        toks = tokenise_document("This paper proposes a network", stops, PROTECTED, 3)
        assert "paper" not in toks
        assert "network" in toks

    def test_lemmatises_nouns_and_verbs(self):
        assert lemmatise("studies") == "study"
        assert lemmatise("trained") == "train"

    def test_drops_numbers_and_punctuation(self):
        toks = tokenise_document("accuracy of 95.3% (p<0.05)", _stops(), PROTECTED, 3)
        assert all(t.isalpha() for t in toks)

    def test_splits_hyphenated_compounds(self):
        # Phrase detection rejoins the meaningful ones as deep_learning.
        toks = tokenise_document("deep-learning models", _stops(), PROTECTED, 3)
        assert "deep" in toks and "learn" in toks

    def test_empty_document_yields_no_tokens(self):
        assert tokenise_document("", _stops(), PROTECTED, 3) == []

    def test_batch_tokenisation_matches_config_shape(self):
        out = tokenise_documents(
            ["AI improves diagnosis", "the data was analysed"],
            domain_stopwords=["data"],
            protected_acronyms=["ai"],
            min_token_length=3,
        )
        assert "ai" in out[0]
        assert "data" not in out[1]


class TestBranchSeparation:
    def test_bertopic_text_is_not_tokenised(self):
        """The BERTopic branch must keep natural language intact (OQ-015)."""
        raw = "Deep Learning is transforming medical imaging in Nepal."
        cleaned = clean_text(raw)
        assert cleaned == raw                      # untouched by cleaning
        tokens = tokenise_document(cleaned, _stops(), PROTECTED, 3)
        assert "is" not in tokens                  # but the LDA branch strips it
        assert len(cleaned.split()) > len(tokens)  # and is strictly richer


class TestParentheticalAcronyms:
    """Authors write 'artificial intelligence (AI)', which made phrase detection
    emit both `artificial_intelligence` and `artificial_intelligence_ai` for one
    concept (191 vs 160 documents). The gloss is stripped; the acronym survives
    wherever it is used standalone."""

    def test_strips_the_gloss(self):
        out = clean_text("We apply artificial intelligence (AI) to imaging.")
        assert "(AI)" not in out
        assert "artificial intelligence" in out

    def test_keeps_standalone_acronym_usage(self):
        out = clean_text("Machine learning (ML) is common. ML models perform well.")
        assert "(ML)" not in out
        assert "ML models" in out

    def test_does_not_strip_ordinary_parentheses(self):
        out = clean_text("Accuracy improved (from 80% to 92%) after tuning.")
        assert "from 80% to 92%" in out

    def test_does_not_strip_long_parentheticals(self):
        out = clean_text("The model (which we describe later) performs well.")
        assert "which we describe later" in out
