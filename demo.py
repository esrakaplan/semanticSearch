"""
Semantic Search Demo
====================
Run this file to see the search engine in action.

    python demo.py

Shows:
  1. Basic search with a sample corpus
  2. Cosine similarity worked example (manual, step by step)
  3. Query vector explanation
  4. Comparison: keyword search vs semantic search
"""

import numpy as np
from semantic_search import (
    SemanticSearchEngine,
    cosine_similarity,
    cosine_similarity_manual,
    preprocess,
)


# ─────────────────────────────────────────────
# SAMPLE CORPUS
# ─────────────────────────────────────────────

CORPUS = [
    # Tech / AI
    "Machine learning models learn patterns from data",
    "Deep learning uses neural networks with many layers",
    "Python is widely used for artificial intelligence",
    "Transformers are the backbone of modern NLP models",
    "GPUs accelerate matrix operations in deep learning",

    # Animals / Pets
    "Dogs are loyal and friendly companion animals",
    "Cats are independent and curious domestic pets",
    "Puppies require training, patience, and lots of love",
    "Golden retrievers are known for their gentle temperament",
    "Veterinarians provide medical care for pets and animals",

    # Food / Cooking
    "Pizza is a popular Italian dish with cheese and tomato",
    "Sushi is a traditional Japanese food with rice and fish",
    "Pasta can be made from wheat flour and eggs",
    "Cooking at home is healthier than eating at restaurants",
    "Fermentation is used to produce bread, cheese, and wine",

    # Science
    "Quantum mechanics describes the behavior of subatomic particles",
    "DNA carries genetic information in all living organisms",
    "The speed of light is approximately 300,000 km per second",
    "Black holes have gravity so strong that light cannot escape",
    "Climate change is driven by greenhouse gas emissions",

    # Sports
    "Football requires teamwork, strategy, and physical endurance",
    "Swimming is a full-body workout that improves cardiovascular health",
    "Basketball players need agility, speed, and accurate shooting",
    "Cycling is both a competitive sport and a daily commuting option",
]


def separator(title: str = "") -> None:
    line = "─" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(f"{line}")
    else:
        print(line)


# ─────────────────────────────────────────────
# DEMO 1 — BASIC SEARCH
# ─────────────────────────────────────────────

def demo_basic_search():
    separator("DEMO 1 — Basic Semantic Search")

    engine = SemanticSearchEngine()
    engine.fit(CORPUS)

    test_queries = [
        "friendly household pets",
        "artificial intelligence and neural networks",
        "Italian and Japanese cuisine",
        "physical exercise and fitness",
        "space and astrophysics",
    ]

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        results = engine.search(query, top_k=3)

        for r in results:
            score_bar = "█" * int(r["score"] * 20)
            print(f"  #{r['rank']} [{r['score']:.3f}] {score_bar}")
            print(f"       {r['document']}")


# ─────────────────────────────────────────────
# DEMO 2 — COSINE SIMILARITY STEP BY STEP
# ─────────────────────────────────────────────

def demo_cosine_similarity():
    separator("DEMO 2 — Cosine Similarity: Manual Walkthrough")

    print("""
Imagine these two sentences as 4-dimensional vectors
representing the presence of: [dog, cat, code, pizza]
""")

    # Manually crafted toy vectors for clarity
    vec_dog_article  = np.array([0.8, 0.3, 0.0, 0.0])   # about dogs
    vec_cat_article  = np.array([0.2, 0.9, 0.0, 0.0])   # about cats
    vec_tech_article = np.array([0.0, 0.0, 0.9, 0.1])   # about coding

    pairs = [
        ("Dog article",  vec_dog_article,  "Cat article",  vec_cat_article),
        ("Dog article",  vec_dog_article,  "Tech article", vec_tech_article),
        ("Cat article",  vec_cat_article,  "Tech article", vec_tech_article),
    ]

    print(f"{'Pair':<40} {'sim':>7}  {'meaning'}")
    print("─" * 65)
    for name_a, va, name_b, vb in pairs:
        sim = cosine_similarity(va, vb)
        meaning = (
            "very similar"  if sim > 0.7 else
            "related"       if sim > 0.3 else
            "unrelated"
        )
        print(f"  {name_a} ↔ {name_b:<25}  {sim:.4f}   {meaning}")

    # Full manual breakdown for one pair
    print(f"\n--- Manual breakdown: Dog ↔ Cat ---")
    a, b = vec_dog_article, vec_cat_article
    dims = ["dog", "cat", "code", "pizza"]

    print(f"\nVector A (Dog article): {a}")
    print(f"Vector B (Cat article): {b}")

    dot = sum(a[i] * b[i] for i in range(len(a)))
    print(f"\n1) Dot product  A·B = ", end="")
    terms = [f"({a[i]:.1f}×{b[i]:.1f})" for i in range(len(a))]
    print(" + ".join(terms), f"= {dot:.4f}")

    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    print(f"\n2) ‖A‖ = sqrt({' + '.join([f'{x:.2f}²' for x in a])}) = {mag_a:.4f}")
    print(f"   ‖B‖ = sqrt({' + '.join([f'{x:.2f}²' for x in b])}) = {mag_b:.4f}")

    sim = dot / (mag_a * mag_b)
    import math
    angle = math.degrees(math.acos(min(1.0, max(-1.0, sim))))
    print(f"\n3) cosine_sim = {dot:.4f} / ({mag_a:.4f} × {mag_b:.4f}) = {sim:.4f}")
    print(f"   Angle between vectors: {angle:.1f}°")


# ─────────────────────────────────────────────
# DEMO 3 — QUERY VECTOR EXPLANATION
# ─────────────────────────────────────────────

def demo_query_explanation():
    separator("DEMO 3 — What Does a Query Look Like as a Vector?")

    engine = SemanticSearchEngine()
    engine.fit(CORPUS)

    engine.explain_query("neural network training")
    engine.explain_query("cats and dogs as pets")


# ─────────────────────────────────────────────
# DEMO 4 — KEYWORD SEARCH vs SEMANTIC SEARCH
# ─────────────────────────────────────────────

def demo_keyword_vs_semantic():
    separator("DEMO 4 — Keyword Search vs Semantic Search")

    query = "canine companions"

    print(f"\nQuery: \"{query}\"")
    print(f"\n--- Keyword search (exact word match) ---")

    query_words = set(preprocess(query))
    keyword_hits = [
        doc for doc in CORPUS
        if any(word in preprocess(doc) for word in query_words)
    ]
    if keyword_hits:
        for hit in keyword_hits:
            print(f"  ✓ {hit}")
    else:
        print("  (no results — none of the query words appear in the corpus)")

    print(f"\n--- Semantic search (meaning-based) ---")
    engine = SemanticSearchEngine()
    engine.fit(CORPUS)
    results = engine.search(query, top_k=3)
    for r in results:
        print(f"  #{r['rank']} [{r['score']:.3f}] {r['document']}")

    print("""
Observation:
  "canine companions" contains neither "dog" nor "cat",
  yet semantic search finds dog/pet-related documents.
  This works because TF-IDF gives shared context words
  (training, companion, animals) overlapping weights.

  In production (BERT, OpenAI embeddings), this effect
  is much stronger — "canine" and "dog" map to nearly
  identical vectors because the model was trained on
  billions of examples where they co-occur.
""")


# ─────────────────────────────────────────────
# DEMO 5 — SIMILARITY MATRIX
# ─────────────────────────────────────────────

def demo_similarity_matrix():
    separator("DEMO 5 — Document Similarity Matrix (subset)")

    engine = SemanticSearchEngine()
    engine.fit(CORPUS)

    # Pick a representative subset
    subset_indices = [0, 5, 10, 15, 20]   # tech, animal, food, science, sport
    subset_docs = [CORPUS[i] for i in subset_indices]
    subset_vecs = engine.document_vectors[subset_indices]
    short_labels = ["ML model", "Dogs", "Pizza", "Quantum", "Football"]

    print("\nCosine similarity between 5 representative documents:\n")

    # Header row
    header = f"{'':12}" + "".join(f"{l:>11}" for l in short_labels)
    print(header)
    print("─" * (12 + 11 * len(short_labels)))

    for i, (label_i, vec_i) in enumerate(zip(short_labels, subset_vecs)):
        row = f"{label_i:<12}"
        for j, vec_j in enumerate(subset_vecs):
            sim = cosine_similarity(vec_i, vec_j)
            # diagonal is always 1.0 (doc vs itself)
            cell = f"{sim:.2f}"
            row += f"{cell:>11}"
        print(row)

    print("""
Expected pattern:
  • Diagonal = 1.00  (every document is identical to itself)
  • Same-topic pairs have higher scores than cross-topic pairs
  • Unrelated topics (ML ↔ Pizza) should be near 0.00
""")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SEMANTIC SEARCH ENGINE — LEARNING PROJECT")
    print("=" * 60)

    demo_basic_search()
    demo_cosine_similarity()
    demo_query_explanation()
    demo_keyword_vs_semantic()
    demo_similarity_matrix()

    print("\n" + "=" * 60)
    print("  Done! Check semantic_search.py for the full implementation.")
    print("=" * 60)
