import os
import random

data_dir = r"E:\Projects\amlc\amazon_shared\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
files_to_check = ["train_source1.tsv", "train_source2.tsv"]

for fname in files_to_check:
    fpath = os.path.join(data_dir, fname)
    print(f"\n=== PEEKING AT: {fname} ===")
    
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}")
        continue
        
    with open(fpath, 'r', encoding='utf-8') as f:
        header = f.readline()
        # Grab the first 50,000 lines to sample from so it runs instantly
        lines = [f.readline() for _ in range(50000)]
        
        # Filter out empty lines just in case
        lines = [line for line in lines if line.strip()]
        
        sample = random.sample(lines, 10)
        for line in sample:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 4:
                country = parts[3]
                name = parts[1]
                address = parts[2]
                print(f"[{country[:2].upper()}] {name[:30]:<30} | {address}")
