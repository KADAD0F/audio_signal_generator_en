# audio_signal_generator_en

This script generates and saves various types of audio signals in popular formats (WAV, MP3, CSV), with optional visualization and full stereo support. It automatically detects the runtime environment (including Termux on Android), installs missing dependencies, and adapts to system limitations.

---

## 🧩 Supported Signal Types

- **Sine wave (`sin`)** — pure tone at a specified frequency.  
- **Amplitude Modulation (`am`)** — carrier wave modulated in amplitude.  
- **Pulse wave (`pulse`)** — rectangular pulses with configurable duty cycle (e.g., square wave).  
- **White noise (`noise`)**:
  - `uniform` — uniformly distributed samples;
  - `normal` — Gaussian (normal) distribution.
- **Chirp / Frequency-Modulated signal (`chm`)**:
  - `linear` — linear frequency sweep;
  - `quadratic` — quadratic sweep;
  - `hyperbolic` — hyperbolic sweep.
- **Multi-signal mode (`multi`)** — combine and sum multiple signals of different types into one output.

---

## 📁 Output Formats

- **WAV** — lossless, 16-bit PCM, compatible with all audio software.
- **MP3** — compressed format (requires `ffmpeg` and `pydub`).
- **CSV** — plain-text format for analysis in Excel, Python, MATLAB, etc.
- **Visualization** — plot of the signal (first 10,000 samples for performance) in PNG, SVG, or PDF.

---

## ⚙️ Key Features

### 🌐 Automatic Environment Detection
- Detects **Termux** (Android terminal) and defaults to shared storage (`~/storage/shared`).
- If write access is missing in Termux, prompts to run `termux-setup-storage` and retries.
- Falls back to local home directory if external storage is unavailable.

### 🔌 Smart Dependency Management
- Checks for required Python libraries (`numpy`, `jax`, `pydub`, `librosa`, `matplotlib`, etc.).
- If missing:
  - Verifies internet connectivity;
  - Asks for user permission to install;
  - Installs via `pip` with a progress bar for heavy packages (e.g., `jax`, `librosa`);
  - Supports fallbacks: `jax.numpy` if `numpy` is unavailable, `plotly` or `plotext` if `matplotlib` fails.

### 🛡️ Safety & Reliability
- **Filename sanitization**: removes invalid characters and limits length to 255.
- **Signal normalization**: peaks are scaled to **0.99** to prevent clipping.
- **Disk space check**: warns if free space is insufficient for large signals.
- **Graceful interruption**: handles `Ctrl+C` during installation or generation.

### 📊 Flexible User Input
- Default values for all parameters.
- Input validation (e.g., frequency ≤ Nyquist limit = `sample_rate / 2`).
- Stereo support: configure left/right channels independently or identically.

---

## 🖥️ Usage Examples

### Example 1: Generate a Sine Wave
```text
Select signal type (1–6 or name): sin
Signal duration (seconds) [0.001-∞]: 3
Sample rate (Hz) [default: 44100]: 
Number of channels (1/2) [default: 1]: 1
Frequency (Hz) [0.0–22050.0]: 440
Amplitude [0.0–1.0]: 0.8
Output filename (without extension): tone_440hz

✅ WAV saved to /home/user/tone_440hz.wav
CSV saved to /home/user/tone_440hz.csv
Save signal visualization? (y/n): y
Visualization format (png/svg/pdf) [default: png]: png
✅ Visualization saved: /home/user/tone_440hz_visualization.png
```

### Example 2: Stereo Gaussian Noise in Termux
```text
📁 Termux: using shared storage — /data/data/com.termux/files/home/storage/shared
...
Select signal type: noise
Signal duration: 5
Number of channels: 2
Noise type (1=uniform, 2=normal) [1]: 2
Amplitude: 1.0
Use different signals for left/right channels? (y/n): n

✅ WAV saved to /storage/emulated/0/tone_440hz.wav
```

### Example 3: Multi-Signal (Sine + Noise)
```text
Select signal type: multi
...
Signal type: sin → added
Signal type: noise → added
Signal type: [empty] → finished

Signal normalized (peak amplitude: 1.0234)
✅ WAV saved to combined_signal.wav
```

---

## ⚙️ Requirements

### 🐍 Base
- **Python 3.6+** (uses f-strings, type hints, modern stdlib).

### 🌐 Internet Access
- Required **only once** to install missing dependencies via `pip`.

### 📦 System Dependencies
- **`ffmpeg`** — needed **only for MP3 export**.  
  Install via:
  - **Ubuntu/Debian**: `sudo apt install ffmpeg`
  - **Fedora/RHEL**: `sudo dnf install ffmpeg`
  - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html ) and add to `PATH`
  - **macOS**: `brew install ffmpeg`
  - **Termux**: `pkg install ffmpeg`

> 💡 Note: FFmpeg is **not required** for WAV or CSV output.

### 📚 Python Libraries (auto-installed if missing)

#### 1. **Numerical Arrays** (required)
- **`numpy`** — primary backend for signal generation.
- **Fallback**: **`jax`** (with `jax.numpy`) — compatible API, but installation may take **10–60 minutes** on Termux.

#### 2. **Audio I/O**
- **`pydub`** — for MP3 export (requires `ffmpeg`).
- **Fallbacks**:
  - **`librosa`** — for advanced audio loading (heavy, but robust).
  - **`soundfile`** — lightweight alternative for WAV/FLAC.

#### 3. **Visualization** (optional)
- **`matplotlib`** — default (uses `Agg` backend for headless environments like Termux).
- **Fallbacks**:
  - **`plotly`** — interactive browser-based plots.
  - **`plotext`** — terminal-native plotting (lightweight, no GUI needed).

> All packages are installed via `pip`. The script handles interruptions and provides clear error messages.

### 📱 Android (Termux) Notes
- First run triggers `termux-setup-storage` if external storage access is missing.
- Heavy packages (`jax`, `librosa`) may compile slowly or fail due to ARM limitations and lack of prebuilt wheels.

---

## 📜 License & Author

- **Author**: KADAD0F  
- **License**: MIT  
