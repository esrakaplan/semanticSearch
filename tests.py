"""
Tests for the Semantic Search Engine
======================================
Run with:
    python -m pytest tests.py -v
  or:
    python tests.py
"""

import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from semantic_search import (
    preprocess,
    build_vocabulary,
    compute_tf,
    compute_idf,
    embed_document,
    cosine_similarity,
    cosine_similarity_manual,
    SemanticSearchEngine,
)


# ─────────────────────────────────────────────
# PREPROCESSING TESTS
# ─────────────────────────────────────────────

def test_preprocess_lowercases():
    result = preprocess("Hello WORLD")
    assert result == ["hello", "world"]

def test_preprocess_removes_punctuation():
    result = preprocess("cats, dogs! fish?")
    assert result == ["cats", "dogs", "fish"]

def test_preprocess_empty_string():
    result = preprocess("")
    assert result == []

def test_build_vocabulary_sorted():
    docs = ["banana apple", "cherry apple"]
    vocab = build_vocabulary(docs)
    assert vocab == sorted(set(["banana", "apple", "cherry"]))


# ─────────────────────────────────────────────
# TF TESTS
# ─────────────────────────────────────────────

def test_tf_sums_to_one_for_single_word():
    vocab = ["cat"]
    tokens = ["cat"]
    tf = compute_tf(tokens, vocab)
    assert abs(tf[0] - 1.0) < 1e-9

def test_tf_proportional_counts():
    vocab = ["cat", "dog"]
    tokens = ["cat", "cat", "dog"]   # cat appears twice, dog once
    tf = compute_tf(tokens, vocab)
    assert abs(tf[0] - 2/3) < 1e-6   # cat = 2/3
    assert abs(tf[1] - 1/3) < 1e-6   # dog = 1/3

def test_tf_unknown_word_ignored():
    vocab = ["cat"]
    tokens = ["cat", "elephant"]   # elephant not in vocab
    tf = compute_tf(tokens, vocab)
    assert tf[0] == 0.5   # cat = 1/2 total tokens


# ─────────────────────────────────────────────
# COSINE SIMILARITY TESTS
# ─────────────────────────────────────────────

def test_cosine_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    sim = cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-9

def test_cosine_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    sim = cosine_similarity(a, b)
    assert abs(sim - (-1.0)) < 1e-9

def test_cosine_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    sim = cosine_similarity(a, b)
    assert abs(sim - 0.0) < 1e-9

def test_cosine_zero_vector():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])
    sim = cosine_similarity(a, b)
    assert sim == 0.0

def test_cosine_manual_matches_numpy():
    a = np.array([3.0, 1.0, 4.0])
    b = np.array([1.0, 5.0, 9.0])
    sim_np = cosine_similarity(a, b)
    sim_manual = cosine_similarity_manual(a, b)
    assert abs(sim_np - sim_manual) < 1e-9

def test_cosine_magnitude_invariant():
    """
    Cosine similarity only cares about direction, not length.
    Scaling a vector should NOT change the similarity.
    """
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    sim_original = cosine_similarity(a, b)
    sim_scaled   = cosine_similarity(a * 1000, b)  # scale a by 1000
    assert abs(sim_original - sim_scaled) < 1e-9


# ─────────────────────────────────────────────
# SEARCH ENGINE TESTS
# ─────────────────────────────────────────────

def test_engine_top_result_is_exact_match():
    """
    An exact copy of a document should always rank #1.
    """
    corpus = [
        "dogs are friendly pets",
        "python is great for machine learning",
        "pizza has cheese and tomato",
    ]
    engine = SemanticSearchEngine()
    engine.fit(corpus)

    results = engine.search("python is great for machine learning", top_k=3)
    assert results[0]["document"] == "python is great for machine learning"

def test_engine_related_docs_rank_higher():
    corpus = [
        "cats and dogs are common household pets",
        "machine learning uses neural networks",
        "quantum physics explores subatomic particles",
    ]
    engine = SemanticSearchEngine()
    engine.fit(corpus)

    results = engine.search("deep learning and AI", top_k=3)
    top_doc = results[0]["document"]
    assert "machine learning" in top_doc

def test_engine_scores_in_descending_order():
    corpus = [
        "neural networks for classification",
        "baking bread at home",
        "deep learning and gradient descent",
    ]
    engine = SemanticSearchEngine()
    engine.fit(corpus)

    results = engine.search("machine learning", top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_engine_raises_before_fit():
    engine = SemanticSearchEngine()
    try:
        engine.search("anything")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass   # expected

def test_engine_top_k_respected():
    corpus = ["doc one", "doc two", "doc three", "doc four", "doc five"]
    engine = SemanticSearchEngine()
    engine.fit(corpus)
    results = engine.search("document", top_k=2)
    assert len(results) == 2


# ─────────────────────────────────────────────
# RUNNER (if not using pytest)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_functions = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  ✓  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {fn.__name__}  →  {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
