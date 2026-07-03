from retrieval import hybrid_search

def run_tests():
    passed_cases = 0
    total_cases = 5

    # Case 1
    print("=== Case 1: job_levels filter ===")
    results1 = hybrid_search("customer service skills", top_k=10, filters={"job_levels": ["Entry-Level"]})
    violators1 = [r for r in results1 if "Entry-Level" not in r.get("job_levels", [])]
    if not violators1:
        print(f"PASS (Count: {len(results1)})")
        passed_cases += 1
    else:
        print(f"FAIL (Count: {len(results1)})")
        print(f"Violators: {[r.get('name') for r in violators1]}")
    for r in results1[:5]:
        print(f"  - {r.get('name')[:40]:<40} | job_levels: {r.get('job_levels', [])}")
    print()

    # Case 2
    print("=== Case 2: test_type_codes filter ===")
    results2 = hybrid_search("problem solving ability", top_k=10, filters={"test_type_codes": ["P"]})
    violators2 = [r for r in results2 if "P" not in r.get("test_type_codes", [])]
    if not violators2:
        print(f"PASS (Count: {len(results2)})")
        passed_cases += 1
    else:
        print(f"FAIL (Count: {len(results2)})")
        print(f"Violators: {[r.get('name') for r in violators2]}")
    for r in results2[:5]:
        print(f"  - {r.get('name')[:40]:<40} | test_type_codes: {r.get('test_type_codes', [])}")
    print()

    # Case 3
    print("=== Case 3: max_duration_minutes filter ===")
    results3 = hybrid_search("assessment", top_k=10, filters={"max_duration_minutes": 15})
    violators3 = [r for r in results3 if r.get("duration_minutes") is not None and r.get("duration_minutes") > 15]
    if not violators3:
        print(f"PASS (Count: {len(results3)})")
        passed_cases += 1
    else:
        print(f"FAIL (Count: {len(results3)})")
        print(f"Violators: {[r.get('name') for r in violators3]}")
    for r in results3[:5]:
        print(f"  - {r.get('name')[:40]:<40} | duration_minutes: {r.get('duration_minutes')}")
    print()

    # Case 4
    print("=== Case 4: combined filters (AND logic) ===")
    results4 = hybrid_search(
        "technical skills test", 
        top_k=10, 
        filters={
            "job_levels": ["Entry-Level"],
            "test_type_codes": ["K"],
            "max_duration_minutes": 20
        }
    )
    violators4 = []
    for r in results4:
        valid_jl = "Entry-Level" in r.get("job_levels", [])
        valid_ttc = "K" in r.get("test_type_codes", [])
        valid_dur = r.get("duration_minutes") is None or r.get("duration_minutes") <= 20
        if not (valid_jl and valid_ttc and valid_dur):
            violators4.append(r)
            
    if not violators4:
        print(f"PASS (Count: {len(results4)})")
        passed_cases += 1
    else:
        print(f"FAIL (Count: {len(results4)})")
        print(f"Violators: {[r.get('name') for r in violators4]}")
    for r in results4[:5]:
        print(f"  - {r.get('name')[:40]:<40} | job_levels: {r.get('job_levels', [])}, test_type_codes: {r.get('test_type_codes', [])}, duration_minutes: {r.get('duration_minutes')}")
    print()

    # Case 5
    print("=== Case 5: impossible filter (should return empty, not crash) ===")
    results5 = hybrid_search("test", top_k=10, filters={"job_levels": ["Nonexistent-Level-XYZ"]})
    if len(results5) == 0:
        print(f"PASS (Count: {len(results5)})")
        passed_cases += 1
    else:
        print(f"FAIL (Count: {len(results5)})")
        print(f"Violators: {[r.get('name') for r in results5]}")
    for r in results5[:5]:
        print(f"  - {r.get('name')[:40]:<40}")
    print()

    print(f"{passed_cases}/{total_cases} cases passed.")

if __name__ == "__main__":
    run_tests()
