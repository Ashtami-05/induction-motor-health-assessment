import os
import scipy.io
import numpy as np

RAW_DIR = os.path.join("..", "dataset", "raw")
OUTPUT_DIR = os.path.join("..", "dataset", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FS = 12000
SEGMENT_LENGTH = 4096
OVERLAP = 0.50
STEP = int(SEGMENT_LENGTH * (1 - OVERLAP))

dataset_info = {
    97: ("Normal", 0),
    98: ("Normal", 1),
    99: ("Normal", 2),
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

total_segments = 0

for file_number, (fault_type, load_hp) in dataset_info.items():

    filename = f"{file_number}.mat"
    filepath = os.path.join(RAW_DIR, filename)

    if not os.path.exists(filepath):
        print(f"WARNING: {filename} not found")
        continue

    data = scipy.io.loadmat(filepath)
    de_key = f"X{file_number:03d}_DE_time"

    if de_key not in data:
        print(f"WARNING: DE signal not found in {filename}")
        continue

    signal = data[de_key].flatten()

    num_segments = 1 + (len(signal) - SEGMENT_LENGTH) // STEP

    for i in range(num_segments):
        start = i * STEP
        end = start + SEGMENT_LENGTH
        segment = signal[start:end]

        output_filename = f"{file_number}_{i:04d}.npy"
        np.save(os.path.join(OUTPUT_DIR, output_filename), segment)

    total_segments += num_segments

    print(f"{filename}: {num_segments} segments | {fault_type} | {load_hp} HP")

print("-----------------------------------------")
print("Signal segmentation completed.")
print("-----------------------------------------")
print(f"Total segments created: {total_segments}")
print(f"Segment length: {SEGMENT_LENGTH} samples")
print(f"Overlap: {OVERLAP * 100:.0f}%")
print(f"Sampling frequency: {FS} Hz")
print(f"Segments saved in: {OUTPUT_DIR}")
print("-----------------------------------------")
