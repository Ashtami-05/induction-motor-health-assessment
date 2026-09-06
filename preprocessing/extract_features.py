import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

PROCESSED_DIR = os.path.join("..", "dataset", "processed")
METADATA_FILE = os.path.join("..", "dataset", "metadata.csv")
OUTPUT_FILE = os.path.join("..", "dataset", "features.csv")

metadata = pd.read_csv(METADATA_FILE)
metadata_lookup = metadata.set_index("file")

records = []

segment_files = sorted(
    f for f in os.listdir(PROCESSED_DIR)
    if f.endswith(".npy")
)

print(f"Total segments found: {len(segment_files)}")

for count, filename in enumerate(segment_files, start=1):

    parts = filename.replace(".npy", "").split("_")

    file_number = int(parts[0])
    segment_number = int(parts[1])

    original_file = f"{file_number}.mat"

    signal = np.load(
        os.path.join(PROCESSED_DIR, filename)
    )

    mean_value = np.mean(signal)
    std_value = np.std(signal)
    rms_value = np.sqrt(np.mean(signal ** 2))
    peak_value = np.max(np.abs(signal))
    peak_to_peak = np.ptp(signal)
    mean_absolute = np.mean(np.abs(signal))
    kurtosis_value = kurtosis(signal, fisher=False)
    skewness_value = skew(signal)

    crest_factor = (
        peak_value / rms_value
        if rms_value != 0 else 0
    )

    row = metadata_lookup.loc[original_file]

    records.append({
        "file": original_file,
        "segment": segment_number,
        "fault_type": row["fault_type"],
        "load_hp": row["load_hp"],
        "rpm": row["rpm"],
        "sampling_frequency_hz": row["sampling_frequency_hz"],
        "mean": mean_value,
        "std": std_value,
        "rms": rms_value,
        "peak": peak_value,
        "peak_to_peak": peak_to_peak,
        "mean_absolute": mean_absolute,
        "kurtosis": kurtosis_value,
        "skewness": skewness_value,
        "crest_factor": crest_factor
    })

    if count % 100 == 0:
        print(f"Processed {count}/{len(segment_files)} segments")

features = pd.DataFrame(records)

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n-----------------------------------------")
print("Feature extraction completed.")
print("-----------------------------------------")
print(f"Total segments processed: {len(features)}")
print("Total features extracted: 9")
print(f"Feature file saved to: {OUTPUT_FILE}")
print("-----------------------------------------")
