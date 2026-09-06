import os
import scipy.io
import pandas as pd

# ---------------------------------------------------------
# CWRU Dataset Metadata Preparation
# Project: Induction Motor Health Assessment
# ---------------------------------------------------------

# Location of raw CWRU .mat files
RAW_DIR = os.path.join("..", "dataset", "raw")

# Sampling frequency
FS = 12000

# Selected CWRU recordings
dataset_info = {
    97:  ("Normal", 0),
    98:  ("Normal", 1),
    99:  ("Normal", 2),
    100: ("Normal", 3),

    105: ("Inner Race", 0),
    106: ("Inner Race", 1),
    107: ("Inner Race", 2),
    108: ("Inner Race", 3),

    118: ("Ball", 0),
    119: ("Ball", 1),
    120: ("Ball", 2),
    121: ("Ball", 3),

    130: ("Outer Race", 0),
    131: ("Outer Race", 1),
    132: ("Outer Race", 2),
    133: ("Outer Race", 3),
}

records = []

for file_number, (fault_type, load_hp) in dataset_info.items():

    filename = f"{file_number}.mat"
    filepath = os.path.join(RAW_DIR, filename)

    if not os.path.exists(filepath):
        print(f"WARNING: {filename} not found")
        continue

    try:
        data = scipy.io.loadmat(filepath)

        # Drive-End vibration variable
        de_key = f"X{file_number:03d}_DE_time"

        if de_key not in data:
            print(f"WARNING: DE signal not found in {filename}")
            continue

        signal = data[de_key].flatten()

        # RPM
        rpm_key = f"X{file_number:03d}RPM"

        if rpm_key in data:
            rpm = float(data[rpm_key].flatten()[0])
        else:
            rpm = None

        records.append({
            "file": filename,
            "fault_type": fault_type,
            "load_hp": load_hp,
            "sampling_frequency_hz": FS,
            "rpm": rpm,
            "signal_length": len(signal)
        })

        print(f"Processed {filename}")

    except Exception as e:
        print(f"ERROR processing {filename}: {e}")


# Create metadata table
metadata = pd.DataFrame(records)

# Save metadata
OUTPUT_DIR = os.path.join("..", "dataset")

output_file = os.path.join(OUTPUT_DIR, "metadata.csv")

metadata.to_csv(output_file, index=False)

print("\n-----------------------------------------")
print("Dataset metadata preparation completed.")
print("-----------------------------------------")
print(f"Total recordings processed: {len(metadata)}")
print(f"Metadata saved to: {output_file}")
print("-----------------------------------------")