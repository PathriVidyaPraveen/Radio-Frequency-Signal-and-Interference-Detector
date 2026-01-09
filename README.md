# Radio-Frequency-Signal-and-Interference-Detector
This is the code (Jupyter Notebook) for detection of different RF(radio frequency) signals like Wi-Fi, Blue-tooth, Zigbee using Computer Vision Techniques and MobileNet Architecture.  

In this Jupyter Notebook , I have generated synthetic Radio Frequency(RF) signals like Wi-Fi, BlueTooth , Zigbee and designed a Mo

---

Wireless Signal Classification & Interference Detection Framework
Project Overview
This repository contains a robust pipeline for the synthesis, processing, and classification of wireless radio frequency (RF) signals using Deep Learning. The project utilizes Synthetic Data Generation to simulate realistic RF environments (Wi-Fi, Bluetooth, ZigBee) and their interferences. The generated In-Phase and Quadrature (IQ) samples are converted into time-frequency spectrograms to train two distinct Convolutional Neural Networks (CNNs) based on MobileNetV3:

Majority Signal Detector: Identifies the dominant communication standard.

Interference Detector: Detects the presence and type of secondary interfering signals in the spectrum.

## 1. Synthetic RF Data Generation
The notebook implements a physics-compliant signal generator for the 2.4 GHz ISM band. It simulates baseband IQ signals, applies channel impairments, and converts them into spectrograms.

Protocols Simulated
Wi-Fi (IEEE 802.11a/g):

Technique: Orthogonal Frequency-Division Multiplexing (OFDM).

Parameters: 64-point FFT, 16-sample Cyclic Prefix (CP), QPSK modulation on data subcarriers.

Bandwidth: 20 MHz.

Bluetooth Classic (IEEE 802.15.1):

Technique: Frequency Hopping Spread Spectrum (FHSS).

Modulation: Gaussian Frequency Shift Keying (GFSK) with a modulation index of 0.32.

Hopping: Simulates 1600 hops/s across 79 channels.

ZigBee (IEEE 802.15.4):

Technique: Direct Sequence Spread Spectrum (DSSS).

Modulation: Offset Quadrature Phase Shift Keying (OQPSK) with half-sine pulse shaping.

Chip Rate: 2 Mchips/s.

Co-Channel Interference Simulation
The framework generates composite signals to simulate crowded spectral environments. It supports 7 distinct classes of spectral occupancy:

Single Source: Wi-Fi, Bluetooth, ZigBee.

Dual Source (Interference): Wi-Fi+Bluetooth, Wi-Fi+ZigBee, Bluetooth+ZigBee.

Multi-Source: Wi-Fi+Bluetooth+ZigBee.

Key Simulation Features:

AWGN Channel: Additive White Gaussian Noise is injected to simulate varying Signal-to-Noise Ratios (SNR ~10dB).

Carrier Frequency Offset (CFO): Signals are randomly placed within the frequency band to simulate asynchronous transmission.

Near-Far Effect: Random power scaling is applied to different signal components to simulate varying distances from the receiver.

Pre-processing Pipeline
STFT (Short-Time Fourier Transform): Converts time-domain IQ samples into frequency-domain representations.

Log-Scale Transformation: Conversion of magnitude to Decibels (dB) to highlight low-power features.

Data Compression: Spectrograms are stored as .npz (compressed NumPy arrays) to optimize I/O during training.

---

## 2. Deep Learning Architecture
The project employs Transfer Learning using the MobileNetV3-Small architecture, optimized for edge-deployment efficiency.

Modifications
Input Layer Adaptation: The first convolutional layer is modified to accept 1-channel input (grayscale spectrograms) instead of the standard 3-channel RGB, aggregating pre-trained weights via mean pooling.

Classification Head: The fully connected layers are adapted to output probabilities for the specific number of classes (3 for Majority, 4 for Interference).

Model 1: Majority Signal Detector
Objective: Classify the primary protocol occupying the channel.

Class Mapping (7 → 3):

Wi-Fi (includes pure Wi-Fi and Wi-Fi-dominated mixtures).

Bluetooth.

ZigBee.

Loss Function: Cross Entropy Loss with Class Balancing to handle dataset skew.

Model 2: Interference Detector
Objective: Detect the specific type of interference (OOD/Anomaly detection context).

Class Mapping (7 → 4):

None (Clean signal).

Wi-Fi (as interferer).

Bluetooth (as interferer).

ZigBee (as interferer).

Inference Logic: Uses a confidence threshold (NONE_THRESH = 0.45) to distinguish between clean and interfered signals.

---

## 3. Advanced Training Strategy
The training pipeline implements State-of-the-Art (SOTA) regularization and optimization techniques to ensure robustness and generalization.

Data Augmentation (Spectrogram-Specific)
SpecAugment: Applies random time and frequency masking to force the model to learn distributed features rather than relying on specific frequency bins or time bursts.

Random Resized Crop: Introduces scale invariance.

Gaussian Jitter: Adds noise to the spectrogram pixels to improve resilience against low SNR.

MixUp: Convex combination of pairs of examples and their labels to linearize the decision boundary and reduce overfitting.

Optimization & Hygiene
Optimizer: AdamW (Adam with decoupled weight decay) for better regularization.

Scheduler: CosineAnnealingLR with Warmup for stable convergence.

Automatic Mixed Precision (AMP): Uses torch.cuda.amp (FP16/FP32) to accelerate training and reduce VRAM usage.

Label Smoothing: Prevents the model from becoming over-confident in its predictions (calibrated probabilities).

Gradient Clipping: Prevents exploding gradients during backpropagation.

---

## 4. Deployment & Export
The trained PyTorch models (.pt) are exported to ONNX (Open Neural Network Exchange) format. This enables deployment on various platforms (TensorRT, OpenVINO, ONNX Runtime) agnostic of the training framework.

---
## 5. File Structure
Arista.ipynb: Main execution notebook containing:

Signal Generation Modules (OFDM, GFSK, DSSS).

Dataset Management (Google Drive mounting & saving).

SpectrogramDataset & InterfDataset PyTorch classes.

Training Loops (Train, Validation, Test).

ONNX Export logic.

Dataset Output:

/content/drive/MyDrive/*_dataset/npz: Compressed spectrogram data.

---

## 6. Requirements
Python 3.x

PyTorch / Torchvision (with CUDA support recommended)

SciPy (Signal processing: spectrogram, windows)

NumPy (Matrix operations)

Matplotlib (Visualization)

Scikit-Learn (Metrics: classification_report, confusion_matrix)

ONNX (Model export)
