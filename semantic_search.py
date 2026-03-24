"""
Semantic Search Engine - Learning Project
==========================================
Learns: embeddings, cosine similarity, vector search

This module implements a minimal but complete semantic search engine
using only numpy (no external ML libraries required).

The key idea: convert text → numbers (vectors), then measure
similarity between vectors using cosine similarity.
"""

import numpy as np
from collections import Counter
import math
import re


# ─────────────────────────────────────────────
# STEP 1: TEXT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(text: str) -> list[str]:
    """
    Tokenize and clean text into a list of lowercase words.

    Preprocessing is the first step in any NLP pipeline.
    We remove punctuation, lowercase everything, and split into tokens.

    Example:
        "Hello, World!" → ["hello", "world"]
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # remove punctuation
    tokens = text.split()
    return tokens


def build_vocabulary(documents: list[str]) -> list[str]:
    """
    Build a sorted vocabulary list from all documents.

    The vocabulary defines the DIMENSIONS of our vector space.
    Every unique word becomes one dimension (axis) in the space.

    Example with 3 docs → vocab might be:
        ["apple", "banana", "cat", "dog", ...]
        Each word = one dimension in the embedding vector.
    """
    all_tokens = []
    for doc in documents:
        all_tokens.extend(preprocess(doc))

    unique_words = sorted(set(all_tokens))   # sorted for determinism
    return unique_words


# ─────────────────────────────────────────────
# STEP 2: TF-IDF EMBEDDING
# ─────────────────────────────────────────────
#
# TF-IDF = Term Frequency × Inverse Document Frequency
#
# TF  = how often a word appears in THIS document
# IDF = how rare the word is across ALL documents
#       (rare words are more meaningful / informative)
#
# A word like "the" → high TF, very low IDF → low score
# A word like "quantum" → moderate TF, high IDF → high score
#

def compute_tf(tokens: list[str], vocab: list[str]) -> np.ndarray:
    """
    Compute Term Frequency vector for a list of tokens.

    TF(word) = count(word in doc) / total_words_in_doc

    Returns a vector of shape (vocab_size,).
    Each position corresponds to a word in the vocabulary.
    """
    word_index = {word: i for i, word in enumerate(vocab)}
    tf_vector = np.zeros(len(vocab))

    token_counts = Counter(tokens)
    total_tokens = len(tokens) if tokens else 1   # avoid division by zero

    for word, count in token_counts.items():
        if word in word_index:
            idx = word_index[word]
            tf_vector[idx] = count / total_tokens

    return tf_vector


def compute_idf(documents: list[str], vocab: list[str]) -> np.ndarray:
    """
    Compute Inverse Document Frequency for all vocabulary words.

    IDF(word) = log( N / (1 + df(word)) )
        N      = total number of documents
        df     = number of documents containing the word
        +1     = smoothing to avoid log(0)

    Words that appear in many documents get a LOW idf score.
    Words that appear in few documents get a HIGH idf score.
    """
    n_docs = len(documents)
    word_index = {word: i for i, word in enumerate(vocab)}
    doc_freq = np.zeros(len(vocab))   # how many docs contain each word

    for doc in documents:
        tokens = set(preprocess(doc))   # set: count each word once per doc
        for token in tokens:
            if token in word_index:
                doc_freq[word_index[token]] += 1

    idf_vector = np.log(n_docs / (1 + doc_freq))
    return idf_vector


def embed_document(text: str, vocab: list[str], idf: np.ndarray) -> np.ndarray:
    """
    Convert a text string into a TF-IDF embedding vector.

    embedding = TF_vector * IDF_vector  (element-wise multiplication)

    The result is a dense vector where:
      - High values → the word is frequent HERE and rare elsewhere
      - Low values  → the word is common everywhere (not distinctive)
    """
    tokens = preprocess(text)
    tf_vector = compute_tf(tokens, vocab)
    tfidf_vector = tf_vector * idf
    return tfidf_vector


# ─────────────────────────────────────────────
# STEP 3: COSINE SIMILARITY
# ─────────────────────────────────────────────
#
# Cosine similarity measures the ANGLE between two vectors.
# It does NOT care about vector length — only direction.
#
#                  A · B
# cosine_sim = ──────────────
#               ‖A‖ × ‖B‖
#
# Result range:
#   1.0  → identical direction (very similar meaning)
#   0.5  → 60° apart (somewhat related)
#   0.0  → 90° apart (unrelated / orthogonal)
#  -1.0  → opposite directions (opposite meaning)
#

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Uses the dot-product formula:
        sim = (A · B) / (||A|| * ||B||)

    Returns a float in [-1.0, 1.0].
    Returns 0.0 if either vector is a zero vector (all zeros).
    """
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0   # zero vector has no direction → undefined similarity

    similarity = dot_product / (magnitude_a * magnitude_b)
    return float(similarity)


def cosine_similarity_manual(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Same calculation but written step-by-step for learning purposes.

    Breaks down every part of the formula so you can see what's happening.
    """
    # Step 1: dot product  →  sum of element-wise products
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

    # Step 2: magnitude of A  →  square root of sum of squares
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))

    # Step 3: magnitude of B
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    # Step 4: divide dot product by product of magnitudes
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity


# ─────────────────────────────────────────────
# STEP 4: SEMANTIC SEARCH ENGINE
# ─────────────────────────────────────────────

class SemanticSearchEngine:
    """
    A complete semantic search engine using TF-IDF + Cosine Similarity.

    Workflow:
        1. fit(documents)   → build vocab, compute IDF, embed all docs
        2. search(query)    → embed query, rank docs by cosine similarity

    In production systems (OpenAI, BERT, etc.), step 1 uses a
    pre-trained neural network instead of TF-IDF. The search
    step (cosine similarity ranking) stays exactly the same.
    """

    def __init__(self):
        self.documents: list[str] = []
        self.vocab: list[str] = []
        self.idf: np.ndarray = None
        self.document_vectors: np.ndarray = None   # shape: (n_docs, vocab_size)
        self.is_fitted: bool = False

    def fit(self, documents: list[str]) -> None:
        """
        Index a collection of documents.

        This is the "offline" phase — done once before any searches.
        In production this would store vectors in a vector database.

        Steps:
            1. Build vocabulary from all documents
            2. Compute IDF weights across the corpus
            3. Embed every document and store the vectors
        """
        print(f"[Index] Fitting {len(documents)} documents...")

        self.documents = documents
        self.vocab = build_vocabulary(documents)

        print(f"[Index] Vocabulary size: {len(self.vocab)} unique words")

        self.idf = compute_idf(documents, self.vocab)

        # Embed all documents and stack into a matrix
        # Shape: (n_documents, vocab_size)
        doc_vectors = [
            embed_document(doc, self.vocab, self.idf)
            for doc in documents
        ]
        self.document_vectors = np.array(doc_vectors)

        self.is_fitted = True
        print(f"[Index] Document matrix shape: {self.document_vectors.shape}")
        print(f"[Index] Ready to search!\n")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for the most semantically similar documents to a query.

        Steps:
            1. Embed the query using the same vocab + IDF
            2. Compute cosine similarity against every document vector
            3. Sort by similarity score (descending)
            4. Return top_k results

        Args:
            query:  The search string from the user
            top_k:  How many results to return

        Returns:
            List of dicts with keys: rank, document, score, index
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit(documents) before searching.")

        # Embed the query into the same vector space as the documents
        query_vector = embed_document(query, self.vocab, self.idf)

        # Compute similarity between query and EVERY document
        # This is a single matrix operation: (n_docs, vocab) @ (vocab,)
        scores = np.array([
            cosine_similarity(query_vector, doc_vec)
            for doc_vec in self.document_vectors
        ])

        # Sort indices from highest to lowest similarity
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, doc_idx in enumerate(ranked_indices, start=1):
            results.append({
                "rank":     rank,
                "score":    round(float(scores[doc_idx]), 4),
                "index":    int(doc_idx),
                "document": self.documents[doc_idx],
            })

        return results

    def explain_query(self, query: str) -> None:
        """
        Show what the query looks like as a vector (non-zero dims only).
        Useful for understanding how TF-IDF represents text.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit(documents) before explaining.")

        query_vector = embed_document(query, self.vocab, self.idf)
        word_index = {word: i for i, word in enumerate(self.vocab)}

        print(f"\n[Explain] Query: '{query}'")
        print(f"[Explain] Tokens: {preprocess(query)}")
        print(f"[Explain] Active dimensions (non-zero TF-IDF weights):")

        nonzero_dims = [
            (word, query_vector[idx])
            for word, idx in word_index.items()
            if query_vector[idx] > 0
        ]
        nonzero_dims.sort(key=lambda x: x[1], reverse=True)

        for word, weight in nonzero_dims:
            bar = "█" * int(weight * 100)
            print(f"   '{word}': {weight:.4f}  {bar}")

        if not nonzero_dims:
            print("   (all zeros — words not in vocabulary)")
