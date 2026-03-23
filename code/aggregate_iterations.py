import os
import glob
import csv
import re

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    
    # We find all summary csvs that end in _summary.csv and are for k3/iter10/disc1
    pattern = os.path.join(results_dir, "*__k3__iter10__disc1__*_summary.csv")
    files = glob.glob(pattern)
    
    # regex to extract strategy
    filename_regex = re.compile(r"([^/\\]+)__k\d+__iter\d+__disc\d+__\d+_\d+_summary\.csv")
    
    data = []
    max_iters = 0
    
    for f in files:
        basename = os.path.basename(f)
        match = filename_regex.match(basename)
        if not match:
            continue
            
        strategy = match.group(1)
        
        scores = []
        with open(f, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                scores.append(int(row['best_sad']))
        
        data.append({
            "strategy": strategy,
            "scores": scores
        })
        max_iters = max(max_iters, len(scores))
        
    # Sort data by strategy alphabetically
    data.sort(key=lambda x: x["strategy"])
    
    # Write aggregated CSV
    out_path = os.path.join(results_dir, "aggregated_iteration_scores.csv")
    
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Strategy"] + [f"Iter {i+1}" for i in range(max_iters)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for item in data:
            row = {"Strategy": item["strategy"]}
            for i in range(max_iters):
                if i < len(item["scores"]):
                    row[f"Iter {i+1}"] = item["scores"][i]
                else:
                    row[f"Iter {i+1}"] = ""
            writer.writerow(row)
            
    print(f"Aggregated {len(data)} strategies into {out_path}")

if __name__ == "__main__":
    main()
