import json
import csv
import re
import sys

def parse_duration(duration_str):
    if not duration_str:
        return None
    match = re.search(r'(\d+)', str(duration_str))
    if match:
        val = int(match.group(1))
        if 'hour' in str(duration_str).lower():
            val *= 60
        return val
    return None

def task_a_classify(record):
    link = record.get("link", "")
    name = record.get("name", "")
    keys = record.get("keys", [])
    desc = record.get("description", "").lower()
    
    # Signal 0 - URL path
    if any(p in link for p in ["/verify/", "/skills/", "/simulations/"]):
        return "INDIVIDUAL_TEST_SOLUTION"
    if any(p in link for p in ["/job-focused-assessments/", "/job-solutions/"]):
        return "JOB_SOLUTION"
        
    # Signal 1 - Name pattern
    if name.lower().endswith("solution"):
        return "JOB_SOLUTION"
        
    # Signal 2 - keys composition
    job_solution_keys = {"Simulations", "Personality & Behavior", "Biodata & Situational Judgment"}
    distinct_keys = set(keys)
    sig2_job = len(distinct_keys) >= 2 and len(distinct_keys.intersection(job_solution_keys)) > 0
    
    # Signal 3 - Description keyword scan
    job_keywords = [
        "solution", "bundle", "combines", "includes a ", "designed for the role of",
        "sample tasks include"
    ]
    sig3_job = any(k in desc for k in job_keywords)
    
    # Decision logic
    if sig2_job and sig3_job:
        return "JOB_SOLUTION"
    elif sig2_job or sig3_job:
        return "AMBIGUOUS"
    else:
        return "INDIVIDUAL_TEST_SOLUTION"

def task_b_map_keys(keys):
    mapping = {
        "Ability & Aptitude": "A",
        "Biodata & Situational Judgment": "B",
        "Competencies": "C",
        "Development & 360": "D",
        "Assessment Exercises": "E",
        "Knowledge & Skills": "K",
        "Personality & Behavior": "P",
        "Simulations": "S"
    }
    
    codes = []
    for k in keys:
        if k in mapping:
            codes.append(mapping[k])
    return codes

def task_c_clean(record, idx):
    codes = task_b_map_keys(record.get("keys", []))
    primary_code = codes[0] if codes else None
    
    duration = parse_duration(record.get("duration", ""))
    
    # Convert "yes"/"no" strings to actual booleans if necessary
    remote_val = record.get("remote", False)
    if isinstance(remote_val, str):
        remote_val = remote_val.lower() == "yes"
        
    adaptive_val = record.get("adaptive", False)
    if isinstance(adaptive_val, str):
        adaptive_val = adaptive_val.lower() == "yes"
    
    return {
        "id": idx,
        "name": record.get("name"),
        "test_type": primary_code,
        "test_type_codes": codes,
        "keys": record.get("keys", []),
        "duration_minutes": duration,
        "languages": record.get("languages", []),
        "job_levels": record.get("job_levels", []),
        "remote": remote_val,
        "adaptive": adaptive_val,
        "description": record.get("description", ""),
        "url": record.get("link", "")
    }

def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--clean-only':
        print("Running in clean-only mode using existing catalog_filtered.json...")
        try:
            with open("catalog_filtered.json", "r", encoding="utf-8") as f:
                filtered = json.load(f)
        except FileNotFoundError:
            print("Error: catalog_filtered.json not found in the current directory.")
            return

        # Task C: Build the clean dataset
        clean_data = []
        for i, r in enumerate(filtered, start=1):
            clean_data.append(task_c_clean(r, i))
            
        # Write final clean JSON
        with open("catalog_clean.json", "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2)
            
        # Write final clean CSV
        if clean_data:
            keys = list(clean_data[0].keys())
            with open("catalog_clean.csv", "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in clean_data:
                    row = r.copy()
                    row["test_type_codes"] = ";".join(row["test_type_codes"])
                    row["keys"] = ";".join(row["keys"])
                    row["languages"] = ";".join(row["languages"])
                    row["job_levels"] = ";".join(row["job_levels"])
                    writer.writerow(row)
        
        print(f"Cleaned data written to catalog_clean.json and catalog_clean.csv.")
        print(f"Total INDIVIDUAL_TEST_SOLUTION records processed: {len(filtered)}")
        return

    # Normal execution (Task A + Task C)
    try:
        with open("catalog_raw.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("Error: catalog_raw.json not found in the current directory.")
        return

    filtered = []
    ambiguous = []
    
    # Task A: Filter and classify
    for r in raw_data:
        cls = task_a_classify(r)
        if cls == "INDIVIDUAL_TEST_SOLUTION":
            filtered.append(r)
        elif cls == "AMBIGUOUS":
            ambiguous.append(r)
            
    # Write ambiguous records
    with open("ambiguous_review.json", "w", encoding="utf-8") as f:
        json.dump(ambiguous, f, indent=2)
        
    # Write intermediate filtered records
    with open("catalog_filtered.json", "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)
        
    # Task C: Build the clean dataset
    clean_data = []
    for i, r in enumerate(filtered, start=1):
        clean_data.append(task_c_clean(r, i))
        
    # Write final clean JSON
    with open("catalog_clean.json", "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2)
        
    # Write final clean CSV
    if clean_data:
        keys = list(clean_data[0].keys())
        with open("catalog_clean.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in clean_data:
                row = r.copy()
                row["test_type_codes"] = ";".join(row["test_type_codes"])
                row["keys"] = ";".join(row["keys"])
                row["languages"] = ";".join(row["languages"])
                row["job_levels"] = ";".join(row["job_levels"])
                writer.writerow(row)
                
    # Summary report
    print(f"Total raw records read: {len(raw_data)}")
    print(f"Excluded as JOB_SOLUTION: {len(raw_data) - len(filtered) - len(ambiguous)}")
    print(f"Flagged AMBIGUOUS: {len(ambiguous)} (written to ambiguous_review.json)")
    print(f"INDIVIDUAL_TEST_SOLUTION written to catalog_clean.json: {len(filtered)}")

if __name__ == "__main__":
    main()
