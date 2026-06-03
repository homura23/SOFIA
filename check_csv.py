import pandas as pd
import os

def clean_invalid_audio_paths(input_file, output_file=None):
    """
    Check whether audio paths in a CSV exist and drop invalid rows.
    
    Args:
        input_file: Input CSV path
        output_file: Output CSV path (overwrites input when None)
    """
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Print columns for verification
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    
    # Define check function
    def check_file_exists(file_path):
        """Check whether a file exists; normalize leading prefixes."""
        if pd.isna(file_path) or file_path == "":
            return False
        # Strip leading slash/dot
        clean_path = str(file_path).lstrip('/').lstrip('.')
        return os.path.exists(clean_path)
    
    # Check rows
    df['full_path_exists'] = df['full_path'].apply(check_file_exists)
    df['vocal_path_exists'] = df['vocal_path'].apply(check_file_exists)
    df['valid'] = df['full_path_exists'] & df['vocal_path_exists']
    
    # Summary
    total_rows = len(df)
    valid_rows = df['valid'].sum()
    invalid_rows = total_rows - valid_rows
    
    print("\nPath existence summary:")
    print(f"  full_path exists: {df['full_path_exists'].sum()} / {total_rows}")
    print(f"  vocal_path exists: {df['vocal_path_exists'].sum()} / {total_rows}")
    print(f"  both exist: {valid_rows} / {total_rows}")
    print(f"  rows to drop: {invalid_rows} / {total_rows}")
    
    # Show a few invalid samples
    if invalid_rows > 0:
        print("\nInvalid sample rows (first 5):")
        invalid_samples = df[~df['valid']].head(5)
        for idx, row in invalid_samples.iterrows():
            print(f"\n  Row {idx + 2}:")
            print(f"    full_path: {row['full_path']}")
            print(f"      -> exists: {row['full_path_exists']}")
            print(f"    vocal_path: {row['vocal_path']}")
            print(f"      -> exists: {row['vocal_path_exists']}")
            
            # Show resolved path
            if pd.notna(row['full_path']):
                clean_full = str(row['full_path']).lstrip('/').lstrip('.')
                print(f"    checked path: {clean_full}")
    
    # Filter valid rows
    df_valid = df[df['valid']].drop(columns=['full_path_exists', 'vocal_path_exists', 'valid'])
    
    # Save output
    if output_file is None:
        output_file = input_file
    
    df_valid.to_csv(output_file, index=False)
    print(f"\nSaved valid rows to: {output_file}")
    print(f"Dropped invalid rows: {invalid_rows}")
    
    return df_valid

# Method 2: memory-friendly processing for large files
import csv

def clean_invalid_paths_large_file(input_file, output_file):
    """Process large CSVs line-by-line to save memory."""
    
    original_count = 0
    valid_count = 0
    invalid_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        header = next(reader)
        
        # Find column indices
        try:
            full_path_idx = header.index('full_path')
            vocal_path_idx = header.index('vocal_path')
        except ValueError as e:
            print(f"Error: required columns not found - {e}")
            print(f"Available columns: {header}")
            return
        
        writer = csv.writer(outfile)
        writer.writerow(header)
        
        for row_num, row in enumerate(reader, start=2):
            original_count += 1
            
            # Get paths
            full_path = row[full_path_idx] if full_path_idx < len(row) else ""
            vocal_path = row[vocal_path_idx] if vocal_path_idx < len(row) else ""
            
            # Check existence
            full_exists = False
            if full_path and full_path.strip():
                clean_path = full_path.lstrip('/').lstrip('.')
                full_exists = os.path.exists(clean_path)
            
            vocal_exists = False
            if vocal_path and vocal_path.strip():
                clean_path = vocal_path.lstrip('/').lstrip('.')
                vocal_exists = os.path.exists(clean_path)
            
            if full_exists and vocal_exists:
                writer.writerow(row)
                valid_count += 1
            else:
                invalid_count += 1
                if invalid_count <= 10:
                    print(f"Drop row {row_num}: full_path={full_path} ({full_exists}), vocal_path={vocal_path} ({vocal_exists})")
    
    print("\nDone:")
    print(f"  total rows: {original_count}")
    print(f"  valid rows: {valid_count}")
    print(f"  dropped rows: {invalid_count}")
    print(f"  output: {output_file}")

# Method 3: validate paths and attempt fixes
def verify_and_fix_paths(input_file, output_file, base_dir=None):
    """Validate paths and attempt to fix them."""
    
    df = pd.read_csv(input_file)
    
    def check_and_fix_path(path, base_dir):
        if pd.isna(path) or path == "":
            return False, path
        
        # Try path variants
        path_variants = [
            path,
            path.lstrip('/'),
            path.lstrip('.'),
            path.lstrip('/.'),
            os.path.join(base_dir, path) if base_dir else path,
        ]
        
        for variant in path_variants:
            if os.path.exists(variant):
                return True, variant
        
        return False, path
    
    # Check and fix paths
    results = df.apply(lambda row: (
        check_and_fix_path(row['full_path'], base_dir),
        check_and_fix_path(row['vocal_path'], base_dir)
    ), axis=1)
    
    df['full_exists'], df['fixed_full_path'] = zip(*[r[0] for r in results])
    df['vocal_exists'], df['fixed_vocal_path'] = zip(*[r[1] for r in results])
    
    # Keep rows where both files exist
    df_valid = df[df['full_exists'] & df['vocal_exists']].copy()
    
    # Use fixed paths if available
    df_valid['full_path'] = df_valid['fixed_full_path']
    df_valid['vocal_path'] = df_valid['fixed_vocal_path']
    
    # Save output
    df_valid = df_valid.drop(columns=['full_exists', 'vocal_exists', 'fixed_full_path', 'fixed_vocal_path'])
    df_valid.to_csv(output_file, index=False)
    
    print(f"Original: {len(df)} rows, valid: {len(df_valid)}, dropped: {len(df)-len(df_valid)}")
    
    return df_valid

# Run cleanup
if __name__ == "__main__":
    input_file = "test_mom_human2.csv"
    output_file = "test_mom_human2_cleaned.csv"
    
    # Select method based on file size
    # Use method 1 for small files
    clean_invalid_audio_paths(input_file, output_file)
    
    # Use method 2 for large files
    # clean_invalid_paths_large_file(input_file, output_file)
    
    # If paths are relative, provide base_dir
    # base_dir = "data/datasets/mom/"
    # verify_and_fix_paths(input_file, output_file, base_dir=base_dir)