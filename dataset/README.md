# CWRU Bearing Vibration Dataset Preparation

## Project

Adaptive Health Assessment and Decision Support System for Induction Motors Using Vibration Signal Processing and Machine Learning

## Purpose

This dataset was prepared for machine-learning-based health assessment of induction motors using vibration signals. The Case Western Reserve University (CWRU) bearing vibration dataset was selected as the benchmark dataset.

## Selected Recordings

A total of 16 CWRU MATLAB (.mat) recordings were selected.

| Fault Class | MATLAB Files | Load |
|---|---|---|
| Normal | 97, 98, 99, 100 | 0, 1, 2, 3 HP |
| Inner Race | 105, 106, 107, 108 | 0, 1, 2, 3 HP |
| Ball | 118, 119, 120, 121 | 0, 1, 2, 3 HP |
| Outer Race | 130, 131, 132, 133 | 0, 1, 2, 3 HP |

## Signal Information

- Signal used: Drive-End (DE) vibration
- Sampling frequency: 12,000 Hz
- Original signals were stored in CWRU MATLAB files.
- RPM information was extracted where available.

## Signal Segmentation

Each vibration signal was divided into fixed-length segments.

- Segment length: 4,096 samples
- Sampling frequency: 12,000 Hz
- Segment duration: approximately 0.341 seconds
- Overlap: 50%
- Step size: 2,048 samples
- Total segments created: 1,521

The segments are stored as NumPy `.npy` files in:

`dataset/processed/`

## Extracted Features

Nine time-domain vibration features were calculated for every segment:

1. Mean
2. Standard deviation
3. RMS
4. Peak
5. Peak-to-peak value
6. Mean absolute value
7. Kurtosis
8. Skewness
9. Crest factor

The complete feature dataset is stored in:

`dataset/features.csv`

## Class Distribution

The original feature dataset contains:

| Fault Class | Segments |
|---|---:|
| Normal | 824 |
| Inner Race | 233 |
| Ball | 232 |
| Outer Race | 232 |
| Total | 1,521 |

Because the original dataset is imbalanced, 232 samples were selected from each class.

Balanced dataset:

- Normal: 232
- Inner Race: 232
- Ball: 232
- Outer Race: 232
- Total: 928

## Train/Test Split

The balanced dataset was divided into training and testing datasets.

Importantly, the split was performed based on the original CWRU recording file rather than randomly splitting individual segments. This helps reduce data leakage because overlapping segments from the same original recording are highly correlated.

### Training Set

- Samples: 690
- Files:
  - 97.mat
  - 98.mat
  - 100.mat
  - 105.mat
  - 106.mat
  - 108.mat
  - 119.mat
  - 120.mat
  - 121.mat
  - 130.mat
  - 131.mat
  - 132.mat

### Testing Set

- Samples: 238
- Files:
  - 99.mat
  - 107.mat
  - 118.mat
  - 133.mat

## Machine Learning Files

The ML-ready datasets are stored in:

`dataset/ml/`

Files:

- `balanced_dataset.csv` - balanced complete dataset
- `train.csv` - training dataset
- `test.csv` - testing dataset

Each CSV contains 15 columns:

- file
- segment
- fault_type
- load_hp
- rpm
- sampling_frequency_hz
- mean
- std
- rms
- peak
- peak_to_peak
- mean_absolute
- kurtosis
- skewness
- crest_factor

## Note on RPM

RPM values were available in most selected recordings. The metadata table contains missing RPM values for 98.mat and 99.mat. These values should be handled appropriately if RPM is used as a machine-learning input.

## Recommended Use

For machine-learning classification, `train.csv` should be used for model training and `test.csv` should be reserved for final evaluation.

The columns `fault_type`, `file`, and `segment` should not be used as numerical input features.

The nine vibration features can be used as the primary input features for the initial classification model.

## Prepared Dataset Summary

| Parameter | Value |
|---|---:|
| Original recordings | 16 |
| Sampling frequency | 12 kHz |
| Segment length | 4,096 samples |
| Segment overlap | 50% |
| Total segments | 1,521 |
| Balanced samples | 928 |
| Training samples | 690 |
| Testing samples | 238 |
| Fault classes | 4 |
| Extracted vibration features | 9 |

## Project Workflow

Raw CWRU recordings

↓

Metadata preparation

↓

Vibration signal segmentation

↓

Time-domain feature extraction

↓

Class balancing

↓

Recording-level train/test split

↓

Machine learning classification

↓

Health assessment and decision support

