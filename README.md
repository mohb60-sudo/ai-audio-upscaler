# AI Audio Upscaler Pro

> Professional audio super-resolution powered by hybrid DSP-AI architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transform low-resolution audio into high-fidelity masters. Our hybrid approach combines traditional DSP with deep learning to hallucinate missing high-frequency content, restore clarity, and enhance sonic presence.

> [!TIP]
> **See what's possible:** Check out [SHOWCASE.md](SHOWCASE.md) for a deep dive into the technical achievements of this project.

> [!IMPORTANT]
> **Safety First:** Please read [AI_GUIDELINES.md](AI_GUIDELINES.md) for responsible AI usage.

---

## ✨ Features

- 🎯 **Hybrid Architecture**: Dual-model system (waveform + spectral) for superior quality
- ⚖️ **Hybrid Quality Control (QC)**: "Judge-Jury-Executioner" system to eliminate hallucinations
- 🌊 **Infinite Streaming**: Disk-based pipeline for processing hour-long audio files
- 📊 **Real-time Visualization**: Live VU Meters and granular progress tracking
- 🚀 **Multiple Sample Rates**: 44.1kHz to 384kHz support
- 🧠 **Auto-Tuning**: Optuna-powered hyperparameter optimization
- ⚡ **Performance Optimized**: VRAM-aware batching and mixed precision
- 🎨 **Web UI**: Beautiful Gradio interface for training, tuning, and processing
- 🔧 **Robust Training**: Handles degraded inputs (MP3, bandwidth-limited, noisy)

---

## 📋 Requirements

### Minimum (Inference)
- **OS**: Windows 10+, Linux, macOS
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GTX 1660 (6GB VRAM) or CPU
- **RAM**: 8GB
- **Storage**: 5GB

### Recommended (Training)
- **GPU**: NVIDIA RTX 4070+ (12GB+ VRAM)
- **RAM**: 32GB
- **Storage**: 50GB (SSD)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-audio-upscaler.git
cd ai-audio-upscaler

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Web UI (Recommended)

```bash
python web_app/app.py
```

Then open your browser to `http://localhost:7860`

### CLI Usage

#### Basic Upscaling

```bash
python -m ai_audio_upscaler.cli upscale \
    --input input.wav \
    --output output.wav \
    --target-rate 96000 \
    --mode ai \
    --model checkpoints/best_model.ckpt
```

#### Batch Processing

```bash
python -m ai_audio_upscaler.cli batch \
    --input-dir ./audio_files \
    --output-dir ./upscaled \
    --target-rate 96000 \
    --format flac
```

#### Training a Model

```bash
python train.py \
    --data-dir ./training_data \
    --save-path models/my_model.ckpt \
    --epochs 50 \
    --batch-size 8 \
    --use-spectral \
    --device cuda
```

---

## 🎨 Web UI Guide

### Studio Tab (Single File Processing)

1. **Upload Audio**: Drag and drop or click to browse
2. **Configure Settings**:
   - **Target Sample Rate**: 96000Hz recommended
   - **Processing Mode**: 
     - `Baseline`: Fast DSP-only (sinc/poly-phase)
     - `AI`: Neural enhancement (requires trained model)
   - **Interpolation Filter**: `Poly-Sinc (HQ)` for best quality
3. **Select Model**: Choose from trained checkpoints (AI mode only)
4. **Advanced Options**:
   - Peak Limiter: Normalize to -1 dBFS
   - TPDF Dither: Add shaped noise for quantization
5. **Click "🔴 RENDER MASTER"**
6. **Preview Results**: Listen to output, view spectrograms

### Batch Queue Tab

Process multiple files with identical settings:

1. **Upload Files**: Select multiple WAV/FLAC files
2. **Set Destination**: Choose output directory
3. **Configure Settings**: Same as Studio tab
4. **Click "▶ START BATCH JOB"**
5. **Monitor Progress**: Real-time log of successes/failures

### Model Lab Tab (Training)

#### Dataset Preparation

1. **Prepare Audio Files**:
   - Format: WAV or FLAC (lossless)
   - Sample Rate: 96kHz+ recommended
   - Duration: 30s-5min per file
   - Quantity: 200+ files minimum
2. **Organize in Folder**: All training files in single directory

#### Training Workflow

1. **Select Dataset**: Browse to your audio folder
2. **Scan Dataset**: Verify file count and formats
3. **Configure Architecture**:
   - **Base Channels**: 32 (balanced), 64 (high quality)
   - **Num Layers**: 4 (standard), 5 (complex audio)
4. **Set Training Parameters**:
   - **Epochs**: 50-100 for production models
   - **Batch Size**: 4-16 (depends on VRAM)
   - **Learning Rate**: 1e-4 (default, rarely needs changing)
5. **Enable Options**:
   - ✅ **Robust Training**: Handles degraded inputs
   - ✅ **Spectral Recovery**: Hybrid model (slower but better)
   - ✅ **Use Mixed Precision**: ~1.8x speedup (recommended)
6. **Performance Settings**:
   - **CPU Workers**: 0 (Windows), 2-4 (Linux)
   - **Use Mixed Precision (AMP)**: ON (default)
7. **Click "🚀 START / RESUME TRAINING"**
8. **Monitor Progress**:
   - Status updates every 5 batches
   - Loss plot updates per epoch
   - Best model saved automatically

#### Auto-Tuning (Optuna)

Automatically find best hyperparameters:

1. **Expand "🤖 Auto-Tuning" Accordion**
2. **Set Number of Trials**: 10 recommended (2-4 hours)
3. **Click "✨ START AUTO-TUNING"**
4. **Wait for Completion**: Best parameters auto-populate
5. **Train Final Model**: Use discovered settings

**Console Output** (watch terminal):
```
============================================================
Trial 0: Testing Hyperparameters
  - base_channels: 32
  - num_layers: 4
  - lr: 0.000234
  - batch_size: 8
============================================================
```

### Benchmark Tab

Evaluate model quality objectively:

1. **Select Ground Truth Folder**: High-res test files
2. **Select Model**: Checkpoint to evaluate
3. **Click "▶ RUN BENCHMARK"**
4. **Review Metrics**:
   - **LSD** (Log-Spectral Distance): Lower = Better
     - < 2.0: Excellent
     - 2.0-3.0: Good
     - \> 4.0: Poor
   - **SSIM** (Structural Similarity): Higher = Better

---

## 🔧 CLI Reference

### Commands

```bash
# Upscale single file
python -m ai_audio_upscaler.cli upscale [OPTIONS]

# Batch process directory
python -m ai_audio_upscaler.cli batch [OPTIONS]

# Train model
python train.py [OPTIONS]

# Run hyperparameter tuning
python tuning.py [OPTIONS]
```

### Upscale Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input` | Path | Required | Input audio file path |
| `--output` | Path | Required | Output file path |
| `--target-rate` | Int | 96000 | Target sample rate (Hz) |
| `--mode` | Choice | `baseline` | `baseline` or `ai` |
| `--model` | Path | None | Model checkpoint (AI mode) |
| `--baseline-method` | Choice | `sinc` | `sinc`, `linear`, `poly-sinc-hq` |
| `--device` | Choice | `cuda` | `cuda` or `cpu` |
| `--normalize` | Flag | False | Peak normalize to -1dB |
| `--format` | Choice | `wav` | `wav`, `flac`, `mp3`, `ogg` |

### Training Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data-dir` | Path | Required | Training dataset directory |
| `--save-path` | Path | Required | Model save location |
| `--epochs` | Int | 100 | Number of training epochs |
| `--batch-size` | Int | 8 | Samples per batch |
| `--lr` | Float | 1e-4 | Learning rate |
| `--base-channels` | Int | 32 | Network width (16/32/64) |
| `--num-layers` | Int | 4 | Network depth (3-6) |
| `--use-gan` | Flag | False | Enable adversarial training |
| `--robust-training` | Flag | False | Augment with degradations |
| `--use-spectral` | Flag | False | Enable spectral recovery |
| `--num-workers` | Int | 0 | Parallel data loaders |
| `--use-amp` | Flag | True | Mixed precision (FP16) |
| `--device` | Choice | `cuda` | `cuda` or `cpu` |

### Example Workflows

#### Quick Test (DSP Only)

```bash
python -m ai_audio_upscaler.cli upscale \
    --input test.wav \
    --output test_96k.wav \
    --target-rate 96000 \
    --mode baseline \
    --baseline-method poly-sinc-hq
```

#### Production Upscaling (AI)

```bash
python -m ai_audio_upscaler.cli upscale \
    --input master.flac \
    --output master_192k.flac \
    --target-rate 192000 \
    --mode ai \
    --model checkpoints/production_model_best.ckpt \
    --normalize \
    --format flac \
    --device cuda
```

#### Train Custom Model

```bash
python train.py \
    --data-dir /path/to/audio/dataset \
    --save-path models/custom_model.ckpt \
    --epochs 100 \
    --batch-size 8 \
    --base-channels 64 \
    --num-layers 5 \
    --robust-training \
    --use-spectral \
    --use-amp \
    --device cuda
```

---

## 📊 Understanding Metrics

### Log-Spectral Distance (LSD)

Measures perceptual difference in frequency domain (lower = better):

- **LSD < 1.5**: Exceptional quality (near-perfect reconstruction)
- **LSD 1.5-2.5**: Excellent (production-ready)
- **LSD 2.5-4.0**: Good (clear improvement over baseline)
- **LSD > 4.0**: Minimal improvement (needs more training)

### Training Loss

- **Generator Loss**: Should decrease steadily (2.0-4.0 typical)
- **Plateau**: After 20-30 epochs is normal
- **Instability**: Sudden spikes indicate learning rate too high

---

## 🛠️ Troubleshooting

### Common Issues

#### "CUDA Out of Memory"

**Solution**: Reduce batch size or use `--num-workers 0`

```bash
python train.py ... --batch-size 4 --num-workers 0
```

#### "DataLoader worker exited unexpectedly" (Windows)

**Solution**: Set workers to 0 in Performance Settings or CLI:

```bash
python train.py ... --num-workers 0
```

#### "No audio files found"

**Solution**: Ensure dataset contains WAV/FLAC files (not subdirectories)

#### Low GPU Utilization

**Solutions**:
1. Increase `num_workers` (Linux: 2-4, Windows: 0)
2. Enable Mixed Precision (AMP)
3. Increase batch size if VRAM allows

#### Poor Quality Results

**Checks**:
1. Train for more epochs (50-100)
2. Enable spectral recovery (`--use-spectral`)
3. Use robust training for degraded inputs
4. Verify dataset quality (lossless, >96kHz)

---

## 📁 Project Structure

```
ai-audio-upscaler/
├── ai_audio_upscaler/          # Core package
│   ├── config.py               # Configuration dataclass
│   ├── dsp.py                  # DSP upsampling methods
│   ├── hardware.py             # GPU/CPU detection
│   ├── pipeline.py             # Inference pipeline
│   └── ai_upscaler/            # Neural network modules
│       ├── model.py            # Waveform U-Net
│       ├── discriminator.py    # GAN discriminator
│       ├── loss.py             # Multi-resolution STFT loss
│       ├── metrics.py          # LSD, SSIM calculation
│       ├── inference.py        # Model loading & chunking
│       └── transforms.py       # Data augmentations
├── web_app/
│   └── app.py                  # Gradio web interface
├── train.py                    # Training script
├── tuning.py                   # Optuna hyperparameter search
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── WHITEPAPER.md               # Technical documentation
└── DEEPDIVE.md                 # Exhaustive guide
```

---

## 🎓 Best Practices

### Dataset Preparation

1. **Use Lossless Formats**: WAV (24-bit) or FLAC only
2. **High Sample Rates**: 96kHz or 192kHz source material
3. **Diverse Content**: Mix of instruments, voices, genres
4. **Consistent Quality**: Avoid mixing studio and live recordings
5. **Sufficient Quantity**: 200+ files minimum, 500+ ideal

### Training Strategy

1. **Start Small**: 10 epochs to verify setup
2. **Monitor Validation**: LSD should decrease steadily
3. **Enable Spectral Recovery**: For best quality (slower)
4. **Use Robust Training**: If processing degraded inputs
5. **Checkpoint Often**: Training saves best model automatically

### Inference Optimization

1. **Use GPU**: 5-10x faster than CPU
2. **Batch Similar Files**: Same sample rate for efficiency
3. **FLAC for Archival**: Lossless compression saves space
4. **Normalize Output**: Ensures consistent loudness

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- PyTorch team for excellent deep learning framework
- Gradio for beautiful UI components
- Optuna for hyperparameter optimization
- Audio research papers that inspired this work

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-audio-upscaler/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-audio-upscaler/discussions)
- **Email**: your.email@example.com

---

**Made with ❤️ for audiophiles and engineers**
