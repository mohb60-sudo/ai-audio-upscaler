# AI Audio Upscaler Pro: Deep Dive Documentation

**Version 1.0**
**Date: November 2025**

---

## Table of Contents

1. [Project Philosophy & Goals](#1-project-philosophy--goals)
2. [System Architecture](#2-system-architecture)
3. [Codebase Structure](#3-codebase-structure)
4. [Data Pipeline](#4-data-pipeline)
5. [Model Details](#5-model-details)
6. [Training System](#6-training-system)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Web Application Architecture](#8-web-application-architecture)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Hardware & Performance](#10-hardware--performance)
11. [Troubleshooting & FAQ](#11-troubleshooting--faq)

---

## 1. Project Philosophy & Goals

The AI Audio Upscaler Pro is designed to solve the "bandwidth limitation" problem in digital audio. Standard upsampling (interpolation) can increase the sample rate but cannot restore high-frequency content lost during recording or compression. This project aims to:

1.  **Hallucinate Plausible High Frequencies**: Use deep learning to predict what the >8kHz spectrum *should* look like based on the <8kHz content.
2.  **Preserve Phase Coherence**: Ensure the reconstructed audio aligns perfectly in time with the original.
3.  **Robustness**: Handle various input degradations (MP3 compression, noise, low bandwidth) gracefully.
4.  **Accessibility**: Provide a professional-grade tool via a user-friendly Web UI.

---

## 2. System Architecture

The system employs a **Hybrid DSP-AI Approach**:

1.  **Stage 1: DSP Baseline**
    - The input audio is first upsampled to the target rate (e.g., 96kHz) using high-quality poly-phase sinc interpolation.
    - This provides a "clean" but "dull" base.

2.  **Stage 2: AI Enhancement (Dual-Model)**
    - **Waveform Model (Time Domain)**: A 1D U-Net processes the raw waveform to refine transients and phase.
    - **Spectral Model (Frequency Domain)**: A 2D U-Net processes the spectrogram to hallucinate missing harmonics and high-frequency texture.

3.  **Stage 3: Fusion**
    - The outputs are fused: Magnitude from the Spectral Model + Phase from the Waveform Model.
    - This combines the best of both worlds: crisp transients and rich spectral content.

---

## 3. Codebase Structure

### Core Package (`ai_audio_upscaler/`)

*   **`config.py`**: Defines `UpscalerConfig` dataclass, the central source of truth for all settings (sample rates, model paths, DSP methods).
*   **`dsp.py`**: Implements `DSPUpscaler` class. Wraps `torchaudio.transforms.Resample` and provides custom sinc/linear interpolation logic.
*   **`pipeline.py`**: The `AudioUpscalerPipeline` class. Orchestrates the end-to-end process: Load -> DSP -> AI -> Save. Handles chunking for long files.
*   **`hardware.py`**: Utilities for system detection (`get_system_info`), VRAM estimation, and training recommendations.

### AI Modules (`ai_audio_upscaler/ai_upscaler/`)

*   **`model.py`**: Contains `AudioSuperResNet` (Waveform U-Net).
    *   `GatedResidualBlock`: The core building block, using sigmoid gates to control information flow.
    *   `NoiseInjection`: Adds stochasticity to help generate realistic textures.
*   **`spectral_model.py`**: Contains `SpectralUNet` (Spectrogram U-Net). Standard 2D Conv-BatchNorm-ReLU architecture.
*   **`discriminator.py`**: Implements `MultiResolutionDiscriminator` for GAN training. Checks audio at multiple scales to ensure realism.
*   **`loss.py`**: Defines `MultiResolutionSTFTLoss`. Calculates spectral distance at multiple FFT sizes (512, 1024, 2048) to capture both time and frequency errors.
*   **`metrics.py`**: Implements LSD (Log-Spectral Distance) and SSIM calculation.
*   **`transforms.py`**: Data augmentations (`MP3Compression`, `BandwidthLimiter`, `QuantizationNoise`) for robust training.
*   **`inference.py`**: `AIUpscalerWrapper` handles model loading and the `process_chunk` logic.

### Application Layer

*   **`web_app/app.py`**: The Gradio interface. Handles UI layout, state management, and async task execution.
*   **`train.py`**: Main training script. Sets up `RobustAudioDataset`, `DataLoader`, and the training loop.
*   **`tuning.py`**: Optuna integration for hyperparameter search.

---

## 4. Data Pipeline

### `RobustAudioDataset` (in `train.py`)

This class is critical for training stability and performance.

1.  **Loading**: Reads audio using `torchaudio`.
2.  **Mono Conversion**: Averages channels if stereo (current limitation).
3.  **Dynamic Resampling**: Caches resamplers to avoid re-initialization overhead.
4.  **Random Cropping**: Extracts a fixed-length segment (e.g., 16384 samples).
5.  **On-the-Fly Degradation** (if `robust=True`):
    *   **Bandwidth Limiting**: Low-pass filters the audio (random cutoff 4kHz-12kHz).
    *   **Quantization**: Reduces bit depth (random 8-12 bits).
    *   **MP3 Compression**: Encodes/decodes as MP3 (random bitrate).
6.  **Input Creation**: The degraded audio is downsampled and then upsampled back to target size using DSP. This forms the input tensor.
7.  **Target Creation**: The original high-res crop is the target tensor.

### DataLoader Optimization

*   **`num_workers`**: Controls parallel CPU processes.
    *   *Windows*: Must be 0 for stability (due to `spawn` method).
    *   *Linux*: Can be 4+ for speed.
*   **`pin_memory`**: True for CUDA training (faster host-to-device transfer).
*   **`prefetch_factor`**: Determines how many batches each worker loads in advance.

---

## 5. Model Details

### Waveform Model (`AudioSuperResNet`)

*   **Input**: `(Batch, 1, Length)`
*   **Structure**: Encoder-Decoder (U-Net) with skip connections.
*   **Downsampling**: Strided convolutions.
*   **Upsampling**: Transposed convolutions.
*   **Bottleneck**: Deepest layer with highest channel count.
*   **AdaIN**: Adaptive Instance Normalization allows the model to adjust style based on global statistics.

### Spectral Model (`SpectralUNet`)

*   **Input**: `(Batch, 1, Freq, Time)` - Log-magnitude spectrogram.
*   **Structure**: 2D U-Net.
*   **Focus**: Learns the mapping from "blurry" high frequencies to "sharp" high frequencies.

### Fusion Logic

```python
# 1. Get Magnitude from Spectral Model
pred_mag = spectral_model(input_mag)

# 2. Get Phase from Waveform Model
out_stft = torch.stft(waveform_model_output)
phase = torch.angle(out_stft)

# 3. Reconstruct
final_stft = pred_mag * torch.exp(1j * phase)
output = torch.istft(final_stft)
```

---

## 6. Training System

### Mixed Precision (AMP)

*   Uses `torch.amp.autocast` to run forward pass in FP16 (half precision).
*   Uses `torch.amp.GradScaler` to scale gradients, preventing underflow.
*   **Benefit**: Reduces VRAM usage by ~40% and speeds up training by ~1.8x on Tensor Cores.

### Loss Functions

1.  **STFT Loss**: Primary metric. Ensures spectral accuracy.
2.  **L1 Loss**: Ensures waveform alignment.
3.  **Adversarial Loss (GAN)**: (Optional) Discriminator pushes generator to create realistic textures, not just minimize error.

### Validation

*   Runs every epoch on held-out data.
*   Calculates **LSD** (Log-Spectral Distance).
*   Saves checkpoint if LSD improves.

---

## 7. Inference Pipeline

### Chunking Strategy

To handle long files without running out of VRAM:

1.  **Segment**: Audio is split into chunks (e.g., 1 second).
2.  **Overlap**: Each chunk overlaps the next by ~50ms.
3.  **Process**: Model runs on each chunk independently.
4.  **Cross-Fade**: Overlapping regions are linearly cross-faded to prevent "clicks" at boundaries.

### Normalization

*   **Peak Normalization**: Ensures output doesn't clip. Defaults to -1 dBFS.
*   **Dithering**: Adds TPDF dither before saving to 16-bit/24-bit formats to prevent quantization distortion.

---

## 8. Web Application Architecture

Built with **Gradio**.

*   **Tabs**:
    *   `Studio`: Single file processing.
    *   `Batch Queue`: Directory processing.
    *   `Model Lab`: Training and Tuning.
    *   `Benchmark`: Evaluation.
*   **State**: Uses global variables for training state (stop events, progress).
*   **Concurrency**:
    *   Training runs in a separate thread/process to keep UI responsive.
    *   `yield` generators are used to stream logs to the UI textboxes.

---

## 9. Hyperparameter Tuning

Powered by **Optuna**.

*   **Objective Function**: Trains a small model for a few epochs and returns the final validation LSD.
*   **Search Space**:
    *   `base_channels`: 16, 32, 64
    *   `num_layers`: 3, 4, 5
    *   `lr`: 1e-4 to 1e-3
    *   `batch_size`: 4, 8, 16
*   **Parallelization**:
    *   Checks available VRAM.
    *   If >8GB, sets `n_jobs=2` to run two trials simultaneously.
    *   Otherwise, runs sequentially.

---

## 10. Hardware & Performance

### VRAM Usage

*   **Base Model (32ch, 4 layers)**: ~2GB VRAM (Training batch size 4).
*   **Spectral Model**: Adds ~1GB VRAM.
*   **Inference**: ~1-2GB VRAM (chunked).

### Windows vs. Linux

*   **Windows**: `DataLoader` requires `num_workers=0` due to `spawn` multiprocessing overhead and file locking issues.
*   **Linux**: Supports `num_workers > 0` efficiently.

### Optimization Tips

1.  **Use AMP**: Always keep this on.
2.  **Batch Size**: Maximize this until OOM occurs.
3.  **Chunk Size**: Reduce if OOM during inference.

---

## 11. Troubleshooting & FAQ

### Q: Why is training slow?
**A**: Ensure you are using a GPU (`cuda`). Check that "Use Mixed Precision" is ON. On Windows, `num_workers=0` is necessary but slower than Linux.

### Q: I get "CUDA Out of Memory".
**A**: Reduce `batch_size` in training settings. During inference, the system automatically handles chunking, but extremely large models might still struggle on <4GB cards.

### Q: The output sounds "metallic".
**A**: This is a common artifact of spectral reconstruction. Try:
1.  Training for more epochs.
2.  Using a larger dataset.
3.  Disabling "Spectral Recovery" (use Waveform model only) if the artifact persists.

### Q: Can I train on stereo files?
**A**: The current system converts everything to mono for training. For inference on stereo files, it processes channels independently or averages them depending on the pipeline configuration. Future updates will add true stereo training.
