import argparse
from retrieval import semantic_search, hybrid_search

def main():
    queries = [
        "Java programming test",
        "personality assessment for managers",
        "SQL",
        "Kubernetes",
        "OPQ32r",
    ]
    
    for query in queries:
        print(f"=== Query: \"{query}\" ===")
        
        print("-- semantic_search top 5 --")
        try:
            sem_results = semantic_search(query, top_k=5)
            for r in sem_results:
                print(f"  {r['score']:.4f} | {r.get('name', 'N/A')}")
        except Exception as e:
            print(f"  ERROR in semantic_search: {e}")
            
        print("-- hybrid_search top 5 --")
        try:
            hyb_results = hybrid_search(query, top_k=5)
            for r in hyb_results:
                print(f"  fused={r['fused_score']:.4f} sem_rank={r['semantic_rank']} bm25_rank={r['bm25_rank']} | {r.get('name', 'N/A')}")
        except Exception as e:
            print(f"  ERROR in hybrid_search: {e}")
            
        print()

if __name__ == "__main__":
    main()
