# AI Audio Up-Scaler Prototype

A Python-based audio upscaler that uses both traditional DSP resampling and a prototype Neural Network (Super Resolution) approach to enhance audio sample rates.

## Features

- **Baseline Mode**: High-quality bandlimited sinc interpolation (via `torchaudio`).
- **AI Mode**: Experimental Neural Super-Resolution (1D ResNet) to hallucinate high frequencies.
- **CLI**: Simple command-line interface.
- **Web UI**: Interactive browser-based demo using Gradio.

## Installation

1. **Prerequisites**: Python 3.11+ installed.
2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Test Audio
Create a sample low-res file (16kHz) to test with:
```bash
python examples/generate_sine_example.py
```
This creates `examples/input_16k.wav`.

### 2. CLI Usage

**Baseline Upscaling (DSP only)**
Upscale to 48kHz using Sinc interpolation:
```bash
python -m ai_audio_upscaler.cli examples/input_16k.wav --target-rate 48000 --mode baseline
```

**AI Upscaling (Neural Net)**
*Note: Without a trained checkpoint, this uses random weights and will produce noise/artifacts. It demonstrates the pipeline connectivity.*
```bash
python -m ai_audio_upscaler.cli examples/input_16k.wav --target-rate 48000 --mode ai
```

### 3. Web UI
Launch the graphical interface:
```bash
python web_app/app.py
```
Open the URL provided in the terminal (usually `http://127.0.0.1:7860`).

### 4. Training (Skeleton)
To train the model (requires a dataset of .wav files):
```bash
python train.py --dataset-path /path/to/wavs --epochs 10
```

## Project Structure
- `ai_audio_upscaler/`: Core package.
  - `dsp.py`: Traditional resampling logic.
  - `pipeline.py`: Main processing flow.
  - `ai_upscaler/`: Neural network modules.
- `web_app/`: Gradio UI.
- `tests/`: Unit tests.

## Limitations
- The AI model is currently initialized with random weights. You must train it on a high-res audio dataset to get actual quality improvements.
- The model is a simple 1D ResNet prototype.
