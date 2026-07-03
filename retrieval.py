import json
import faiss
import numpy as np
import pickle
import re
import argparse
from sentence_transformers import SentenceTransformer
from build_embeddings import build_embedding_text

# Lazy singletons for search
_model = None
_index = None
_meta = None
_bm25 = None
_faiss_matrix = None

def load_resources():
    global _model, _index, _meta, _faiss_matrix
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if _index is None:
        _index = faiss.read_index("catalog.index")
        _faiss_matrix = _index.reconstruct_n(0, _index.ntotal)
    if _meta is None:
        with open("catalog_meta.json", "r", encoding="utf-8") as f:
            _meta = json.load(f)

def load_bm25():
    global _bm25
    if _bm25 is None:
        with open("bm25_index.pkl", "rb") as f:
            _bm25 = pickle.load(f)

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

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def build_bm25_index():
    from rank_bm25 import BM25Okapi
    print("Loading catalog_meta.json...")
    with open("catalog_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    print("Tokenizing corpus...")
    corpus_tokens = [tokenize(build_embedding_text(r)) for r in meta]
    
    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_tokens)
    
    print("Saving bm25_index.pkl...")
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
        
    print(f"Success! Built BM25 index over {len(meta)} records.")

def hybrid_search(
    query: str,
    top_k: int = 10,
    filters: dict | None = None,
    rrf_k: int = 60,
) -> list[dict]:
    """
    filters (all optional, all AND'd together if multiple given):
      - "job_levels": list[str]      -> keep record if ANY overlap with record["job_levels"]
      - "test_type_codes": list[str] -> keep record if ANY overlap with record["test_type_codes"]
      - "max_duration_minutes": int  -> keep record if record["duration_minutes"] is None
                                         OR record["duration_minutes"] <= max_duration_minutes

    Returns up to top_k records from catalog_meta.json, each with these fields
    added: "fused_score" (float), "semantic_score" (float, raw cosine),
    "semantic_rank" (int, 1-based, within the filtered candidate set),
    "bm25_score" (float, raw), "bm25_rank" (int, 1-based, within the filtered
    candidate set). Sorted descending by fused_score.
    """
    load_resources()
    load_bm25()
    
    # Pre-filter first
    candidate_indices = []
    for i, r in enumerate(_meta):
        keep = True
        if filters:
            if "job_levels" in filters and filters["job_levels"]:
                # keep if any overlap
                if not any(jl in r.get("job_levels", []) for jl in filters["job_levels"]):
                    keep = False
            if keep and "test_type_codes" in filters and filters["test_type_codes"]:
                if not any(ttc in r.get("test_type_codes", []) for ttc in filters["test_type_codes"]):
                    keep = False
            if keep and "max_duration_minutes" in filters and filters["max_duration_minutes"] is not None:
                duration = r.get("duration_minutes")
                # Keep if None OR <= max_duration_minutes
                if duration is not None and duration > filters["max_duration_minutes"]:
                    keep = False
        if keep:
            candidate_indices.append(i)
            
    if not candidate_indices:
        return []
        
    # Semantic scores over candidate set only
    prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
    query_vector = _model.encode(
        [prefixed_query], 
        batch_size=1, 
        normalize_embeddings=True
    ).astype(np.float32).flatten()
    
    # Compute cosine similarity
    candidate_matrix = _faiss_matrix[candidate_indices]
    semantic_scores = np.dot(candidate_matrix, query_vector)
    
    # Rank semantic scores (descending)
    sem_sort_order = np.argsort(-semantic_scores)
    sem_ranks = [None] * len(candidate_indices)
    current_rank = 1
    for idx in sem_sort_order:
        if semantic_scores[idx] > 0:
            sem_ranks[idx] = current_rank
            current_rank += 1
    
    # BM25 scores over candidate set only
    bm25_all_scores = _bm25.get_scores(tokenize(query))
    bm25_scores = np.array([bm25_all_scores[i] for i in candidate_indices])
    
    bm25_sort_order = np.argsort(-bm25_scores)
    bm25_ranks = [None] * len(candidate_indices)
    current_rank = 1
    for idx in bm25_sort_order:
        if bm25_scores[idx] > 0:
            bm25_ranks[idx] = current_rank
            current_rank += 1
    
    # Fuse via RRF
    fused_scores = np.zeros(len(candidate_indices))
    for i in range(len(candidate_indices)):
        sem_contrib = 1.0 / (rrf_k + sem_ranks[i]) if sem_ranks[i] is not None else 0.0
        bm25_contrib = 1.0 / (rrf_k + bm25_ranks[i]) if bm25_ranks[i] is not None else 0.0
        fused_scores[i] = sem_contrib + bm25_contrib
    
    # Final sort
    final_sort_order = np.argsort(-fused_scores)
    
    results = []
    for idx_in_candidates in final_sort_order:
        if sem_ranks[idx_in_candidates] is None and bm25_ranks[idx_in_candidates] is None:
            continue
            
        orig_idx = candidate_indices[idx_in_candidates]
        record = _meta[orig_idx].copy()
        record["fused_score"] = float(fused_scores[idx_in_candidates])
        record["semantic_score"] = float(semantic_scores[idx_in_candidates])
        record["semantic_rank"] = sem_ranks[idx_in_candidates]
        record["bm25_score"] = float(bm25_scores[idx_in_candidates])
        record["bm25_rank"] = bm25_ranks[idx_in_candidates]
        results.append(record)
        
        if len(results) >= top_k:
            break
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-bm25", action="store_true", help="Build and persist the BM25 index")
    args = parser.parse_args()
    
    if args.build_bm25:
        build_bm25_index()
