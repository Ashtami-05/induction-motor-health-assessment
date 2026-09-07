import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat
from pathlib import Path

# Folder containing the CWRU .mat files
raw_folder = Path("dataset/raw")

# Sampling and segmentation settings
fs = 12000
segment_length = 4096
step = 2048

# List of all CWRU recordings
mat_files = sorted(raw_folder.glob("*.mat"))

print("Number of .mat files found:", len(mat_files))

# Store frequency-domain features
results = []

# Process every .mat file
for mat_file in mat_files:

    print("Processing:", mat_file.name)

    # Load MATLAB file
    data = loadmat(mat_file)

    # Find the Drive End vibration signal
    de_keys = [key for key in data.keys() if key.endswith("_DE_time")]

    if not de_keys:
        print("No DE vibration signal found in", mat_file.name)
        continue

    signal = data[de_keys[0]].flatten()

    # Process all 4096-sample segments with 50% overlap
    for segment_number, start in enumerate(
        range(0, len(signal) - segment_length + 1, step),
        start=0
    ):

        segment = signal[start:start + segment_length]

        # FFT
        fft_values = rfft(segment)

        # Frequency values
        frequencies = rfftfreq(segment_length, 1 / fs)

        # Amplitude spectrum
        amplitude = np.abs(fft_values) / segment_length

        # Peak frequency and amplitude
        peak_index = np.argmax(amplitude)
        peak_frequency = frequencies[peak_index]
        peak_amplitude = amplitude[peak_index]

        # Spectral energy
        spectral_energy = np.sum(amplitude ** 2)

        # Spectral centroid
        amplitude_sum = np.sum(amplitude)

        if amplitude_sum > 0:
            spectral_centroid = np.sum(
                frequencies * amplitude
            ) / amplitude_sum
        else:
            spectral_centroid = 0

        # Spectral bandwidth
        if amplitude_sum > 0:
            spectral_bandwidth = np.sqrt(
                np.sum(
                    ((frequencies - spectral_centroid) ** 2) * amplitude
                ) / amplitude_sum
            )
        else:
            spectral_bandwidth = 0

        # Store results
        results.append({
            "file": mat_file.name,
            "segment": segment_number,
            "peak_frequency": peak_frequency,
            "peak_amplitude": peak_amplitude,
            "spectral_energy": spectral_energy,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth
        })

# Convert results to DataFrame
frequency_features = pd.DataFrame(results)

# Save frequency-domain features
output_file = "dataset/frequency_features.csv"
frequency_features.to_csv(output_file, index=False)

print()
print("Frequency-domain feature extraction completed.")
print("Total segments:", len(frequency_features))
print("Number of features:", 5)
print("Saved to:", output_file)