from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data import documents

model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)

def semantic_search(query, top_k=3):

    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": documents[idx],
            "score": float(similarities[idx])
        })

    return results


if __name__ == "__main__":
    query = input("Search: ")

    results = semantic_search(query)

    print("\nResults:\n")
    for r in results:
        print(f"{r['text']} (score: {r['score']:.4f})")