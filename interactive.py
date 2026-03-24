"""
Interactive Semantic Search CLI
================================
A simple command-line interface to explore the search engine interactively.

Usage:
    python interactive.py

Commands inside the shell:
    search <query>       — find similar documents
    add <text>           — add a new document to the index
    explain <query>      — show TF-IDF weights for a query
    compare <a> | <b>    — cosine similarity between two strings
    list                 — list all indexed documents
    quit                 — exit
"""

from semantic_search import SemanticSearchEngine, cosine_similarity, embed_document


CORPUS = [
    "Machine learning models learn patterns from data",
    "Deep learning uses neural networks with many layers",
    "Python is widely used for artificial intelligence",
    "Transformers are the backbone of modern NLP models",
    "Dogs are loyal and friendly companion animals",
    "Cats are independent and curious domestic pets",
    "Puppies require training, patience, and lots of love",
    "Pizza is a popular Italian dish with cheese and tomato",
    "Sushi is a traditional Japanese food with rice and fish",
    "Quantum mechanics describes the behavior of subatomic particles",
    "Football requires teamwork, strategy, and physical endurance",
    "Swimming is a full-body workout that improves cardiovascular health",
]

HELP_TEXT = """
Commands:
  search <query>          search the corpus
  add <text>              add document and re-index
  explain <query>         show query vector weights
  compare <a> | <b>       cosine similarity between two phrases
  list                    list all documents
  help                    show this message
  quit                    exit
"""


def rebuild_engine(corpus: list[str]) -> SemanticSearchEngine:
    engine = SemanticSearchEngine()
    engine.fit(corpus)
    return engine


def handle_search(engine: SemanticSearchEngine, args: str):
    if not args.strip():
        print("Usage: search <your query>")
        return
    results = engine.search(args.strip(), top_k=5)
    print(f"\nResults for: \"{args.strip()}\"\n")
    for r in results:
        bar = "█" * int(r["score"] * 25)
        print(f"  #{r['rank']}  {r['score']:.4f}  {bar}")
        print(f"        {r['document']}\n")


def handle_compare(engine: SemanticSearchEngine, args: str):
    if "|" not in args:
        print("Usage: compare <phrase a> | <phrase b>")
        return
    parts = args.split("|", 1)
    text_a, text_b = parts[0].strip(), parts[1].strip()
    vec_a = embed_document(text_a, engine.vocab, engine.idf)
    vec_b = embed_document(text_b, engine.vocab, engine.idf)
    sim = cosine_similarity(vec_a, vec_b)

    meaning = (
        "very similar"  if sim > 0.8 else
        "similar"       if sim > 0.5 else
        "somewhat related" if sim > 0.2 else
        "unrelated"
    )
    print(f"\n  \"{text_a}\"")
    print(f"  \"{text_b}\"")
    print(f"  cosine similarity = {sim:.4f}  ({meaning})\n")


def handle_list(corpus: list[str]):
    print(f"\nIndexed documents ({len(corpus)} total):\n")
    for i, doc in enumerate(corpus):
        print(f"  [{i:2d}] {doc}")
    print()


def main():
    corpus = list(CORPUS)
    engine = rebuild_engine(corpus)

    print("=" * 55)
    print("  Semantic Search — Interactive Shell")
    print("=" * 55)
    print(HELP_TEXT)

    while True:
        try:
            raw = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        parts = raw.split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("Bye!")
            break
        elif cmd == "search":
            handle_search(engine, args)
        elif cmd == "add":
            if args.strip():
                corpus.append(args.strip())
                engine = rebuild_engine(corpus)
                print(f"  Added. Index now has {len(corpus)} documents.\n")
            else:
                print("Usage: add <document text>")
        elif cmd == "explain":
            engine.explain_query(args.strip() or "example")
        elif cmd == "compare":
            handle_compare(engine, args)
        elif cmd == "list":
            handle_list(corpus)
        elif cmd == "help":
            print(HELP_TEXT)
        else:
            print(f"Unknown command: '{cmd}'. Type 'help' for a list.")


if __name__ == "__main__":
    main()
