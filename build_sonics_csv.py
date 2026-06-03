import os
import csv
import glob

# Path configuration
base_wav = "data/datasets/sonics/audio/wav"
base_vocal = "data/datasets/sonics/audio/vocals"
output_dir = "data/sonics_csv"

# Subdirectory list
subdirs = [
    "chirp-v2-xxl-alpha",
    "chirp-v3",
    "chirp-v3.5",
    "udio-120s",
    "udio-30s"
]

def build_csvs():
    for subdir in subdirs:
        wav_dir = os.path.join(base_wav, subdir)
        vocal_dir = os.path.join(base_vocal, subdir)
        
        # Build output file name, e.g. test_udio-120s.csv
        csv_filename = f"test_{subdir}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        print(f"Generating: {csv_path} ...")
        
        rows = []
        
        # Find all wav files
        if not os.path.exists(wav_dir):
            print(f"Warning: missing dir {wav_dir}, skipping.")
            continue
            
        files = glob.glob(os.path.join(wav_dir, "*.wav"))
        # Sort for determinism
        files.sort()
        
        if not files:
            print(f"Warning: no wav files found in {subdir}.")
            continue

        for full_path in files:
            filename = os.path.basename(full_path)
            # Expected vocal filename suffix
            vocal_filename = filename.replace(".wav", "_vocal_16k.wav")
            expected_vocal_path = os.path.join(vocal_dir, vocal_filename)
            
            # Check if vocals exist
            if os.path.exists(expected_vocal_path):
                vocal_path = expected_vocal_path
            else:
                # If missing, leave blank or raise depending on your workflow
                vocal_path = "" 
            
            rows.append({
                "full_path": full_path,
                "vocal_path": vocal_path,
                "label": 1,
                "source": subdir
            })
            
        # Write CSV
        if rows:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["full_path", "vocal_path", "label", "source"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"Done. Wrote {len(rows)} rows.")
        else:
            print("No data generated.")

if __name__ == "__main__":
    build_csvs()
