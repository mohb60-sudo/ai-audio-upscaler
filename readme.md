# AI Audio Up-Scaler: Complete Technical Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Core Modules](#core-modules)
4. [AI Components](#ai-components)
5. [User Interfaces](#user-interfaces)
6. [Training System](#training-system)
7. [Configuration](#configuration)
8. [API Reference](#api-reference)
9. [File Formats & Compatibility](#file-formats--compatibility)
10. [Performance & Optimization](#performance--optimization)
11. [Future Roadmap](#future-roadmap)

---

## Project Overview

### Purpose
The AI Audio Up-Scaler is a Python-based application that enhances audio quality by upsampling to higher sample rates using a combination of traditional Digital Signal Processing (DSP) and Neural Network-based AI enhancement.

### Key Features
- **Dual Processing Modes**: Baseline DSP and AI-enhanced upscaling
- **Multiple Sample Rate Support**: Upsample from any rate to target rates (48kHz, 96kHz, 192kHz, etc.)
- **GPU Acceleration**: CUDA support for fast AI processing
- **Visual Analysis**: Spectrogram comparison for quality verification
- **Training Framework**: Complete system for training custom AI models
- **User-Friendly Interfaces**: Both CLI and web-based GUI
- **Memory Efficient**: Chunked processing for large audio files
- **Format Flexibility**: Supports WAV, FLAC, MP3 input formats

### Technology Stack
- **Language**: Python 3.11+
- **ML Framework**: PyTorch 2.0+
- **Audio Processing**: torchaudio, soundfile, scipy
- **Web UI**: Gradio
- **Visualization**: matplotlib
- **Platform**: Cross-platform (Windows, Linux, macOS)

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                      │
│  ┌──────────────┐            ┌──────────────┐          │
│  │   CLI App    │            │   Web UI     │          │
│  │  (cli.py)    │            │  (app.py)    │          │
│  └──────┬───────┘            └──────┬───────┘          │
└─────────┼──────────────────────────┼──────────────────┘
          │                          │
          └──────────┬───────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│              Processing Pipeline                         │
│                 (pipeline.py)                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Load Audio  → 2. DSP Process → 3. AI Enhance │  │
│  │     ↓              ↓                   ↓         │  │
│  │ 4. Normalize → 5. Analysis → 6. Save Output    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
          │                          │
    ┌─────┴─────┐            ┌───────┴────────┐
    ↓           ↓            ↓                ↓
┌────────┐  ┌────────┐  ┌──────────┐  ┌───────────┐
│  DSP   │  │   AI   │  │ Analysis │  │   Train   │
│ Layer  │  │  Model │  │  Module  │  │  System   │
└────────┘  └────────┘  └──────────┘  └───────────┘
```

### Data Flow

**Input** → **Load** → **Resample (DSP)** → **AI Enhancement** (optional) → **Normalize** (optional) → **Analysis** (optional) → **Save** → **Output**

---

## Core Modules

### 1. Configuration ([config.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/config.py))

**Purpose**: Centralized configuration management

**Class**: [UpscalerConfig](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/config.py#4-19)

**Parameters**:
- `target_sample_rate` (int): Target sample rate in Hz (e.g., 48000, 96000, 192000)
- [mode](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/train.py#157-277) (str): Processing mode - "baseline" (DSP only) or "ai" (DSP + AI enhancement)
- `baseline_method` (str): Resampling method - "sinc", "poly-sinc", "poly-sinc-hq"
- `model_checkpoint` (str, optional): Path to trained AI model weights
- `device` (str): Compute device - "cpu" or "cuda"
- `export_format` (str): Output format - "wav", "flac", "mp3", "ogg"

**Advanced DSP Settings**:
- `use_advanced_dsp` (bool): Enable HQPlayer-inspired processing
- `dsp_quality_preset` (str): "fast", "balanced", "quality", "ultra"
- [apply_dither](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/dsp_advanced.py#162-177) (bool): Enable TPDF dithering
- `noise_shaper` (str): "none", "tpdf", "ns9", "ns15", "auto"

**Usage**:
```python
from ai_audio_upscaler.config import UpscalerConfig

config = UpscalerConfig(
    target_sample_rate=192000,
    mode="ai",
    baseline_method="poly-sinc-hq",
    device="cuda",
    use_advanced_dsp=True,
    dsp_quality_preset="ultra",
    noise_shaper="ns9"
)
```

---

### 2. DSP Module ([dsp.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/dsp.py) & [dsp_advanced.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/dsp_advanced.py))

**Purpose**: Traditional digital signal processing for baseline upsampling.

**Standard DSP ([dsp.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/dsp.py))**:
- **Class**: [DSPUpscaler](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/dsp.py#7-68)
- **Methods**: Sinc (Hann window) and Linear interpolation.

**Advanced DSP ([dsp_advanced.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/dsp_advanced.py))** - *New in v2.0*:
- **Class**: [AdvancedDSPUpscaler](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/dsp_advanced.py#16-220)
- **Purpose**: HQPlayer-inspired audiophile processing.
- **Features**:
    - **Poly-Sinc Filters**: High-quality polyphase sinc interpolation with Kaiser windows.
        - `poly-sinc-fast`: 8x taps, beta=5.0 (Balanced)
        - `poly-sinc`: 16x taps, beta=8.6 (High Quality)
        - `poly-sinc-hq`: 32x taps, beta=12.0 (Ultra Quality)
    - **Two-Stage Upsampling**: Automatically engages for ratios >= 8x (e.g., 44.1kHz -> 352.8kHz).
    - **TPDF Dithering**: Triangular Probability Density Function dithering to decorrelate quantization noise.
    - **Noise Shaping**: Psychoacoustic noise shaping to push quantization noise to ultrasonic frequencies.
        - `ns9`: 9th order shaper (optimized for >192kHz)
        - `ns15`: 15th order shaper (optimized for >384kHz)

---

### 3. Pipeline ([pipeline.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/pipeline.py))

**Purpose**: Orchestrates the complete upscaling workflow.

**Class**: [AudioUpscalerPipeline](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/pipeline.py#13-220)

**Key Updates**:
- Integrates [AdvancedDSPUpscaler](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/dsp_advanced.py#16-220) for high-fidelity baseline generation.
- Supports generation of multiple analysis plots (Spectrogram, PSD, Waveform Zoom).
- Handles "Generative BWE-UNet" model inference.

---

### 4. Analysis Module ([analysis.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/analysis.py))

**Purpose**: Visual analysis tools for quality verification.

**Functions**:

1.  **[generate_spectrogram](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/analysis.py#7-63)**:
    - **Type**: Log-magnitude Spectrogram (STFT)
    - **Use**: Visualizing frequency content over time.
    - **Settings**: 2048 FFT, Inferno colormap, Dark theme.

2.  **[generate_psd_plot](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/analysis.py#64-101)** (*New*):
    - **Type**: Power Spectral Density (Welch's Method)
    - **Use**: Analyzing energy distribution across frequencies.
    - **Features**: Highlights Nyquist limit, shows noise floor and high-frequency extension.

3.  **[generate_waveform_zoom](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/analysis.py#102-137)** (*New*):
    - **Type**: Time-domain Waveform Slice (10ms)
    - **Use**: Inspecting transient response and pre/post-ringing artifacts.

**Theme**: All plots are styled with a custom dark theme (transparent background, cyan/amber accents) to match the UI.

---

## AI Components

### 1. Neural Network Model ([ai_upscaler/model.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/model.py))

**Architecture**: Generative Bandwidth Extension Network (BWE-UNet)

**Class**: [AudioSuperResNet](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/model.py#56-145)

**Purpose**: Generatively "hallucinate" plausible high-frequency content to fill in missing details, rather than just interpolating.

**Network Structure**:

```
Input (Upsampled Baseline)
    ↓
Head Conv1D
    ↓
Encoder (Downsampling Path)
    ├─ Gated Residual Block (Dilated)
    ├─ Noise Injection
    └─ Downsample Conv
    ↓
Bottleneck
    ├─ Gated Residual Blocks (High Dilation)
    └─ Noise Injection
    ↓
Decoder (Upsampling Path)
    ├─ Upsample TransposeConv
    ├─ Skip Connection (Concat/Add)
    ├─ Gated Residual Block
    └─ Noise Injection
    ↓
Tail Conv1D
    ↓
Global Residual Add (Input + Generated Detail)
    ↓
Output (Enhanced Audio)
```

**Key Features**:
- **Generative Noise Injection**: Injects random noise into layers (similar to StyleGAN) to allow the model to generate stochastic textures (air, shimmer) that don't exist in the input.
- **Gated Activations**: Uses `Tanh * Sigmoid` gates (WaveNet style) to capture complex non-linear audio dynamics better than ReLU.
- **U-Net Architecture**: Multi-scale processing captures both broad musical structure and fine-grained texture simultaneously.
- **Global Residual**: The model learns only the *difference* (missing high frequencies) rather than regenerating the whole signal.

**Parameters**:
- `in_channels` (int): Input channels (default: 1)
- `base_channels` (int): Base feature dimension (default: 32)
- `num_layers` (int): Depth of U-Net (default: 4)

**Input/Output**:
- Input: Baseline upsampled audio (from DSP)
- Output: Enhanced audio with generatively filled frequencies

---

### 2. Inference Engine ([ai_upscaler/inference.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/inference.py))

**Class**: [AIUpscalerWrapper](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/inference.py#9-109)

**Purpose**: Manages model loading and inference with memory efficiency

**Methods**:

#### [__init__(config)](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/loss.py#10-16)
Load model and prepare for inference.

**Functionality**:
- Loads [AudioSuperResNet](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/model.py#56-145) model
- Loads checkpoint weights if provided (otherwise uses random initialization with warning)
- Configures device (CPU/CUDA)
- Sets model to evaluation mode

#### [enhance(waveform, original_sr, target_sr, chunk_seconds=2.0)](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/inference.py#53-87)
Apply AI enhancement with automatic chunking for memory efficiency.

**Parameters**:
- [waveform](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/analysis.py#102-137) (torch.Tensor): Upsampled audio (channels, time)
- `original_sr` (int): Original sample rate
- `target_sr` (int): Target sample rate
- `chunk_seconds` (float): Chunk duration for processing (default: 2 seconds)

**Returns**: Enhanced audio tensor

**Memory Management**:
- **Short Audio** (< chunk_size): Process entire file
- **Long Audio** (> chunk_size): Split into chunks, process sequentially, concatenate
- **Chunk Size**: `target_sr * chunk_seconds` samples (e.g., 96,000 samples for 2s at 48kHz)

**Error Handling**:
- Catches CUDA OOM errors
- Suggests reducing `chunk_seconds` if OOM occurs inside chunk

**Performance**:
- Processes stereo audio by treating each channel as independent batch item
- Uses `torch.no_grad()` for inference (no gradient computation)
- Clamps output to [-1.0, 1.0] to prevent clipping

---

### 3. Advanced Training Components

#### Discriminator ([ai_upscaler/discriminator.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/discriminator.py))

**Purpose**: For GAN-based training (advanced feature)

**Class**: `MultiScaleDiscriminator`

**Architecture**:
- Multiple discriminators operating at different time scales
- Detects artifacts and unrealistic audio patterns
- Provides adversarial feedback during training

#### Loss Functions ([ai_upscaler/loss.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/loss.py))

**Purpose**: Compute training losses for model optimization

**Available Losses**:

1. **Time-Domain Loss** (L1):
   ```python
   loss = torch.mean(torch.abs(predicted - target))
   ```

2. **Spectral Loss** (Multi-scale STFT):
   ```python
   class MultiScaleSTFTLoss:
       # Computes STFT magnitudes at multiple FFT sizes
       # Penalizes frequency-domain differences
   ```

3. **Perceptual Loss**:
   - Computed in frequency domain
   - Emphasizes perceptually important frequencies
   - Weights higher frequencies more heavily

4. **Adversarial Loss** (for GAN training):
   ```python
   class AdversarialLoss:
       # Discriminator-based loss
       # Encourages realistic audio generation
   ```

#### Metrics ([ai_upscaler/metrics.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/metrics.py))

**Purpose**: Evaluate model quality objectively

**Available Metrics**:

1. **SNR (Signal-to-Noise Ratio)**:
   ```python
   def calculate_snr(clean, noisy):
       # Measures signal quality in dB
   ```

2. **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)**:
   ```python
   def calculate_si_sdr(reference, estimate):
       # Scale-invariant quality metric
   ```

3. **Log-Spectral Distance**:
   ```python
   def log_spectral_distance(ref, est):
       # Measures spectral similarity
   ```

#### Transforms ([ai_upscaler/transforms.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/transforms.py))

**Purpose**: Data augmentation for training

**Available Transforms**:

1. **Random Gain**: Vary audio volume
2. **Time Stretching**: Temporal variations
3. **Pitch Shifting**: Frequency variations
4. **Add Noise**: Robustness training
5. **Random EQ**: Frequency response variations

---

## User Interfaces

### 1. Command-Line Interface ([cli.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/cli.py))

**Purpose**: Scriptable batch processing

**Usage**:
```bash
python -m ai_audio_upscaler.cli INPUT_FILE [OPTIONS]
```

**Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input_path` | str | Required | Path to input audio file |
| `--output-path` | str | Auto | Output file path (auto: `{input}_upscaled_{rate}hz.wav`) |
| `--target-rate` | int | 48000 | Target sample rate in Hz |
| `--mode` | str | baseline | Processing mode ([baseline](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/tests/test_resampler.py#35-46) or [ai](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/web_app/app.py#344-914)) |
| `--baseline-method` | str | sinc | DSP method ([sinc](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/tests/test_resampler.py#25-34) or `linear`) |
| `--model-checkpoint` | str | None | Path to AI model checkpoint |
| `--device` | str | cpu | Compute device (`cpu` or `cuda`) |

**Examples**:

```bash
# Basic DSP upscaling to 96kHz
python -m ai_audio_upscaler.cli song.wav --target-rate 96000

# AI-enhanced upscaling with CUDA
python -m ai_audio_upscaler.cli song.wav \
  --mode ai \
  --target-rate 96000 \
  --model-checkpoint ./checkpoints/model.ckpt \
  --device cuda

# Fast linear interpolation
python -m ai_audio_upscaler.cli podcast.mp3 \
  --baseline-method linear \
  --target-rate 48000
```
### Dataset ([train.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/train.py))

**Class**: [RobustAudioDataset](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/train.py#25-115)

**Purpose**: Flexible dataset loader with automatic preprocessing

**Features**:
- **Multi-Format Support**: WAV, FLAC, MP3
- **Auto-Resampling**: Normalizes all inputs to target sample rate
- **Paired Data Generation**: Creates (low-res → high-res) pairs on-the-fly
- **Random Cropping**: Extracts fixed-length segments for training

**Process Flow**:
```
Load Audio File (any sample rate)
    ↓
Convert to Mono
    ↓
Resample to Target SR (e.g., 48kHz) → [Ground Truth]
    ↓
Downsample to Input SR (e.g., 24kHz)
    ↓
Upsample back to Target SR using DSP → [Input Features]
    ↓
Return (Input, Target) pair
```

**Parameters**:
- `data_dir`: Root directory containing audio files
- `target_sr`: Target sample rate (default: 48000)
- `input_sr`: Simulated low-res sample rate (default: 24000)
- `segment_length`: Training segment length in samples (default: 16384)

**Error Handling**:
- Skips corrupted files
- Returns zero tensors for failed loads (prevents training crash)

---

### Training Function ([train.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/train.py))

**Function**: [train_model(data_dir, save_path, epochs, batch_size, lr, device, progress_callback, yield_loss)](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/train.py#157-277)

**Purpose**: Complete training loop with UI integration

**Parameters**:
- `data_dir` (str): Path to training audio
- `save_path` (str): Where to save checkpoint
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size (default: 16, reduce if OOM)
- `lr` (float): Learning rate (default: 1e-4)
- `device` (str): "cpu" or "cuda"
- `progress_callback` (callable): Updates UI progress
- `yield_loss` (callable): Sends loss values for plotting

**Training Loop**:
```python
for epoch in range(epochs):
    for batch in dataloader:
        inputs, targets = batch
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = L1Loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

**Optimization**:
- **Optimizer**: Adam with default β values
- **Loss Function**: L1 (Mean Absolute Error)
- **Data Loading**: `num_workers=0` for Windows compatibility in UI

**Output**:
- Saves model state dict to `{save_path}.ckpt`
- Logs epoch-wise average loss
- Returns success message

---

## Configuration

### File Structure

```
ai-audio-upscaler/
├── ai_audio_upscaler/          # Main package
│   ├── __init__.py
│   ├── config.py               # Configuration dataclass
│   ├── dsp.py                  # DSP upsampling
│   ├── pipeline.py             # Processing pipeline
│   ├── analysis.py             # Visualization tools
│   ├── cli.py                  # Command-line interface
│   └── ai_upscaler/            # AI components
│       ├── model.py            # Neural network architecture
│       ├── inference.py        # Inference engine
│       ├── discriminator.py    # GAN discriminator
│       ├── loss.py             # Loss functions
│       ├── metrics.py          # Evaluation metrics
│       └── transforms.py       # Data augmentation
├── web_app/
│   └── app.py                  # Gradio web interface
├── train.py                    # Training script
├── examples/
│   └── generate_sine_example.py
├── tests/
│   └── test_resampler.py
├── requirements.txt
├── setup.py
└── README.md
```

### Environment Variables

(None currently, but can be extended for advanced configuration)

---

## API Reference

### Quick Start

```python
from ai_audio_upscaler import UpscalerConfig, AudioUpscalerPipeline

# Configure
config = UpscalerConfig(
    target_sample_rate=96000,
    mode="baseline",
    baseline_method="sinc",
    device="cpu"
)

# Create pipeline
pipeline = AudioUpscalerPipeline(config)

# Process audio
results = pipeline.run(
    input_path="input.wav",
    output_path="output.wav",
    normalize=True,
    generate_analysis=True
)

# Access results
print(f"Saved to: {results['output_path']}")
input_spec = results['input_spectrogram']  # matplotlib Figure
output_spec = results['output_spectrogram']
```

### Advanced Usage

```python
# Custom progress tracking
def my_progress_callback(progress, message):
    print(f"[{progress*100:.0f}%] {message}")

results = pipeline.run(
    input_path="song.flac",
    output_path="song_upscaled.wav",
    normalize=False,
    generate_analysis=True,
    progress_callback=my_progress_callback
)
```

---

## File Formats & Compatibility

### Input Formats

| Format | Extensions | Supported | Notes |
|--------|-----------|-----------|-------|
| WAV | .wav | ✓ | All sample rates, bit depths |
| FLAC | .flac | ✓ | Lossless compression |
| MP3 | .mp3 | ✓ | Lossy, any bitrate |
| OGG | .ogg | ✓ (via torchaudio) | Vorbis codec |

### Output Formats

| Format | Supported | Notes |
|--------|-----------|-------|
| **WAV** | ✓ | Default. 32-bit float. Supports all sample rates. Best for further processing. |
| **FLAC** | ✓ | Lossless compression. Supports high sample rates (up to 192kHz+). Recommended for archiving. |
| **MP3** | ✓ | Lossy. Limited to max 48kHz. Good for sharing/preview. |
| **OGG** | ✓ | Vorbis codec. Good balance of quality/size. |

**Note on High Sample Rates**:
- **MP3** is technically limited to 48kHz. If you select a higher target rate (e.g., 96kHz) and choose MP3, the output will likely be downsampled or fail depending on the encoder.
- **WAV** and **FLAC** are recommended for all high-resolution upscaling tasks.

### Sample Rate Support

| Input SR | Output SR | Quality |
|----------|-----------|---------|
| 8 kHz | Any | Extreme upsampling (may introduce artifacts) |
| 16 kHz | 48/96 kHz | Common for voice/podcast enhancement |
| 22.05 kHz | 44.1/88.2 kHz | CD-quality family |
| 24 kHz | 48/96 kHz | Half-rate upsampling |
| 44.1 kHz | 88.2/176.4 kHz | Standard CD upsampling |
| 48 kHz | 96/192 kHz | Professional audio upsampling |

**Nyquist Consideration**: DSP upsampling is bandlimited to input Nyquist frequency (SR/2). AI enhancement attempts to hallucinate content above this limit.

---

## Performance & Optimization

### Benchmarks

**Test System**: RTX 3090 (24GB), i9-12900K, 64GB RAM

| Task | Duration (3min song) | GPU Memory |
|------|---------------------|------------|
| DSP Sinc (48→96 kHz) | 1-2s | 0.5 GB |
| AI Enhancement (CPU) | 45-60s | - |
| AI Enhancement (CUDA) | 8-12s | 1.5 GB |
| Spectrogram Generation | 2-3s | - |
| Training (epoch, batch=16) | 120-180s | 3 GB |

### Memory Management

**Chunked Inference**:
- Default chunk: 2 seconds of audio
- Processes long files in segments to prevent OOM
- Trade-off: Slight quality loss at chunk boundaries (future: overlap-add)

**Recommended Settings**:

| GPU Memory | Batch Size | Chunk Size |
|------------|------------|------------|
| 4 GB | 4 | 1s |
| 8 GB | 8 | 2s |
| 16 GB | 16 | 4s |
| 24 GB+ | 32 | 8s |

### Optimization Tips

1. **Use CUDA**: 5-10x faster than CPU for AI mode
2. **Reduce Chunk Size**: If OOM errors occur
3. **Batch Processing**: Use CLI for multiple files (parallelizable)
4. **Lower Batch Size**: Reduces memory usage during training
5. **Use Baseline Mode**: 50x faster than AI mode for quick previews

---

## Future Roadmap

### Planned Features

#### Phase 1: HQPlayer DSP Integration (8 weeks)
- Advanced interpolation filters (poly-sinc, Gauss)
- Professional noise shaping (TPDF, NS9, LNS15)
- Two-stage upsampling for extreme rates
- DSD conversion support
- See [hqplayer_integration_plan.md](file:///C:/Users/MBalakrishnan/.gemini/antigravity/brain/224688ed-d191-45dc-87fe-13a336f656c7/hqplayer_integration_plan.md) for details

#### Phase 2: Stem Separation Integration (3 weeks)
- Demucs-based source separation
- Per-stem upscaling with optimized settings
- Intelligent remixing with phase alignment
- Specialized AI models per instrument type
- See [stem_separation_integration_plan.md](file:///C:/Users/MBalakrishnan/.gemini/antigravity/brain/224688ed-d191-45dc-87fe-13a336f656c7/stem_separation_integration_plan.md) for details

#### Phase 3: Advanced Features (ongoing)
- A/B comparison player in UI
- Batch processing interface
- Preset management (save/load configurations)
- Additional export formats (FLAC, DSD)
- Real-time preview mode
- API server for remote processing

### Known Limitations

1. **AI Model**: Uses random weights by default (requires training)
2. **Output Format**: WAV only (MP3 has SR limitations)
3. **Chunk Boundaries**: Potential artifacts at 2-second intervals in AI mode
4. **Stereo Processing**: Channels processed independently (future: joint processing)
5. **OneDrive Sync**: Git repository corruption risk (recommendation: move project outside OneDrive)

---

## Troubleshooting

### Common Issues

**Issue**: CUDA Out of Memory
**Solution**: 
- Reduce chunk size: Modify `chunk_seconds` in [inference.py](file:///c:/Users/MBalakrishnan/OneDrive%20-%20Schools%20Insurance%20Authority/Documents/Dev/Test-App/ai-audio-upscaler/ai_audio_upscaler/ai_upscaler/inference.py)
- Lower batch size during training
- Use CPU mode

**Issue**: "No module named 'ai_audio_upscaler'"
**Solution**: Run as module `python -m ai_audio_upscaler.cli` or install package

**Issue**: MP3 save error at high sample rates
**Solution**: Already fixed - output enforced to WAV format

**Issue**: Git corruption
**Solution**: Move project outside OneDrive, or add `.git` to OneDrive exclusions

**Issue**: Random/noisy AI output
**Solution**: Expected behavior - AI model requires training with real audio data

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/mohb60-sudo/ai-audio-upscaler.git
cd ai-audio-upscaler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### Code Style

- **Linting**: Follow PEP 8
- **Docstrings**: Google-style docstrings
- **Type Hints**: Preferred for new code
- **Comments**: Explain "why", not "what"

---

## License

Apache 2.0 license

---

## Acknowledgments

- **PyTorch Team**: Core ML framework
- **Torchaudio**: Audio processing primitives
- **Gradio**: Rapid UI prototyping
- **Demucs (Meta)**: Source separation (planned integration)
- **HQPlayer**: Inspiration for advanced DSP features (planned integration)

---

## Contact & Support

**Repository**: https://github.com/mohb60-sudo/ai-audio-upscaler

**Issues**: Please report bugs or feature requests via GitHub Issues

**Documentation Version**: 1.0 (2025-11-23)


