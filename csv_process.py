import csv
import os

# ----------------------------
# Configuration
# ----------------------------

# List CSV files to merge
INPUT_CSV_FILES = [
    "test_human2.csv",
    "test_chirp-v2-xxl-alpha.csv", 
    "test_chirp-v3.csv",
    "test_chirp-v3.5.csv",
    "test_udio-120s.csv",
    "test_udio-30s.csv"
]

# Output CSV filename
OUTPUT_CSV_FILE = "ml_sonics_original_all_f1.csv"

# ----------------------------
# Implementation
# ----------------------------

def merge_csv_files(input_files, output_file):
    """Merge multiple CSV files assuming identical headers."""
    merged_data = []
    header = None
    
    print(f"Preparing to merge {len(input_files)} files...")

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"[WARN] File not found: {file_path} (skipped)")
            continue
        
        print(f"Reading: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                # Read header
                try:
                    current_header = next(reader)
                except StopIteration:
                    print(f"[WARN] File is empty: {file_path} (skipped)")
                    continue

                # Validate header
                if header is None:
                    header = current_header
                else:
                    # Simple check: headers must match exactly
                    if current_header != header:
                        print(f"[ERROR] Header mismatch: {file_path}")
                        print(f"      expected: {header}")
                        print(f"      actual: {current_header}")
                        print("      (file skipped)")
                        continue
                
                # Read rows
                count = 0
                for row in reader:
                    merged_data.append(row)
                    count += 1
                print(f"      -> Read {count} rows")

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    # Write output
    if header and merged_data:
        print(f"\nWriting output file: {output_file}")
        try:
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(header)
                # Write all rows
                writer.writerows(merged_data)
            print(f"Done. Wrote {len(merged_data)} rows (header excluded).")
            print(f"Output: {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"[ERROR] Failed to write output: {e}")
    else:
        print("\n[INFO] No valid data to merge.")

if __name__ == "__main__":
    # Warn when no inputs are configured
    if not INPUT_CSV_FILES:
        print("Please add CSV paths to INPUT_CSV_FILES at the top of this script.")
    else:
        merge_csv_files(INPUT_CSV_FILES, OUTPUT_CSV_FILE)
