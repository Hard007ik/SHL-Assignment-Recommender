import json
import datetime
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_embedding_text(r: dict) -> str:
    return (
        f"{r['name']}. Category: {', '.join(r.get('keys', []))}. "
        f"Job level: {', '.join(r.get('job_levels', []))}. {r.get('description', '')}"
    )

# Lazy singletons for search
_model = None
_index = None
_meta = None

def load_resources():
    global _model, _index, _meta
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if _index is None:
        _index = faiss.read_index("catalog.index")
    if _meta is None:
        with open("catalog_meta.json", "r", encoding="utf-8") as f:
            _meta = json.load(f)

def semantic_search(query: str, top_k: int = 20) -> list[dict]:
    """
    Loads catalog.index, catalog_meta.json, and the same sentence-transformers
    model (BAAI/bge-small-en-v1.5) as singletons (load once, reuse across calls
    — do not reload per call).
    Prepends the instruction prefix "Represent this sentence for searching
    relevant passages: " to `query` before encoding (bge-small requires this on
    queries for good retrieval quality; catalog documents were embedded WITHOUT
    this prefix, so do not add it anywhere except here).
    Encodes with the same normalize_embeddings=True setting used at build time.
    Runs faiss_index.search, returns up to top_k records from catalog_meta.json
    at the matched row indices, each with an added "score": float field
    (the raw inner-product/cosine similarity), sorted descending by score.
    If the index returns -1 for any slot (fewer than top_k matches), skip it.
    """
    load_resources()
    
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
    
    # Encode query
    query_vector = _model.encode(
        [prefixed_query], 
        batch_size=1, 
        normalize_embeddings=True
    ).astype(np.float32)
    
    # Search
    scores, indices = _index.search(query_vector, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            record = _meta[idx].copy()
            record["score"] = float(score)
            results.append(record)
            
    return results

def main():
    print("Loading catalog_clean.json...")
    with open("catalog_clean.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    # Step 1: Self-heal
    for r in records:
        r["test_type"] = ",".join(r.get("test_type_codes", []))
        
    # Step 2: Build embedding text
    texts = [build_embedding_text(r) for r in records]
    
    # Step 3: Encode
    print("Loading model BAAI/bge-small-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    print(f"Encoding {len(texts)} records...")
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)
    
    dim = embeddings.shape[1]
    
    # Step 4: Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # Step 5: Persist artifacts
    print("Saving artifacts...")
    faiss.write_index(index, "catalog.index")
    
    with open("catalog_meta.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    model_info = {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dim": int(dim),
        "num_records": len(records),
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    with open("model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)
        
    print(f"Success! Embedded {len(records)} records (dim={dim}).")
    print("Wrote catalog.index, catalog_meta.json, and model_info.json")
    
    # Step 7: Sanity checks
    print("\n--- Sanity Check 1: 'Java programming test' ---")
    res1 = semantic_search("Java programming test", top_k=5)
    for r in res1:
        print(f"Score: {r['score']:.4f} | Name: {r['name']}")
        
    print("\n--- Sanity Check 2: 'personality assessment for managers' ---")
    res2 = semantic_search("personality assessment for managers", top_k=5)
    for r in res2:
        print(f"Score: {r['score']:.4f} | Name: {r['name']}")

if __name__ == "__main__":
    main()
