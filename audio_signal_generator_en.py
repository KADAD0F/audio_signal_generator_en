#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import platform
import subprocess
import time
import shutil
import wave
import csv
import re

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def sanitize_filename(name: str) -> str:
    """Remove or replace invalid filename characters."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name.strip())
    name = name.rstrip('. ')  # Windows doesn't allow trailing dots or spaces
    if not name:
        return "output"
    return name[:255]  # Enforce filesystem filename length limits

def is_yes(user_input: str) -> bool:
    """
    Check if user input is affirmative.
    Supports both English ('y', 'yes') and Russian ('да', 'д') responses.
    """
    return user_input.strip().lower() in ('y', 'yes', 'да', 'д')

def is_termux() -> bool:
    """
    Detect if the script is running inside Termux (Android terminal emulator).
    Termux uses a special HOME path that includes 'com.termux'.
    """
    return 'com.termux' in os.environ.get('HOME', '')

def check_internet() -> bool:
    """
    Check for internet connectivity by pinging well-known hosts.
    Returns True if at least one host responds.
    """
    hosts = ['yandex.ru', 'google.com']
    for host in hosts:
        try:
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            command = ['ping', param, '1', host]
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            if result.returncode == 0:
                return True
        except (subprocess.SubprocessError, OSError, TimeoutError):
            continue
    return False

def check_ffmpeg() -> bool:
    """
    Verify that ffmpeg is available in the system PATH.
    Required for MP3 export via pydub.
    """
    try:
        subprocess.run(['ffmpeg', '-version'],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def determine_output_directory() -> str:
    """
    Determine a suitable output directory with write permissions.
    - In Termux: tries ~/storage/shared first (external storage).
      If unavailable, prompts user to run termux-setup-storage or fall back to $HOME.
    - Elsewhere: uses the current working directory.
    """
    if not is_termux():
        return os.getcwd()

    home = os.environ.get('HOME', '')
    if not home:
        print("⚠️  Could not determine home directory. Using current working directory.")
        return os.getcwd()

    shared_dir = os.path.join(home, 'storage', 'shared')
    local_dir = home

    def can_write_to(path: str) -> bool:
        """Test if we can write to a given directory."""
        if not os.path.exists(path):
            return False
        test_file = os.path.join(path, '.write_test_ffmpeg_signal')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except (OSError, IOError):
            return False

    if can_write_to(shared_dir):
        print(f"📁 Termux: using shared storage — {shared_dir}")
        return shared_dir

    print(f"❌ No write access to {shared_dir}")
    print("\nIn Termux, external storage access requires explicit permission.")

    while True:
        choice = input(
            "Try to request permission via termux-setup-storage? (y/n): "
        ).strip().lower()

        if choice in ('y', 'yes', 'да', 'д'):
            print("Running termux-setup-storage... Follow the on-screen prompts.")
            print("Once complete, press Enter to continue.")
            try:
                subprocess.run(['termux-setup-storage'], check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                print("⚠️  Failed to launch termux-setup-storage.")
                break

            input()  # Wait for user confirmation

            if can_write_to(shared_dir):
                print(f"✅ Permission granted. Files will be saved to: {shared_dir}")
                return shared_dir
            else:
                print("❌ Permission not granted. Try again or choose local storage.")
                continue

        elif choice in ('n', 'no', 'нет', 'н'):
            print(f"📁 Using local Termux directory: {local_dir}")
            return local_dir
        else:
            print("Please enter 'y' or 'n'.")

    # Fallback if loop exits without success
    print(f"📁 Fallback: saving to local directory {local_dir}")
    return local_dir

# ==============================================================================
# DEPENDENCY INSTALLATION
# ==============================================================================

def install_package_heavy(package_name: str) -> bool:
    """
    Install a "heavy" package (e.g., jax, librosa) with a progress indicator.
    Shows either a percentage bar or spinner animation.
    Handles Ctrl+C gracefully.
    """
    print(f"\nInstalling {package_name}... ", end='', flush=True)
    start_time = time.time()
    install_cmd = [sys.executable, '-m', 'pip', 'install', package_name]

    process = subprocess.Popen(
        install_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )

    bar_length = 30
    last_update = 0
    spinner = '|/-\\'

    def parse_line(line: str):
        """Extract percentage from pip output (if present)."""
        if '%' in line:
            parts = line.split()
            for part in parts:
                if part.endswith('%'):
                    try:
                        return float(part.rstrip('%'))
                    except ValueError:
                        continue
        return None

    try:
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            current_time = time.time()
            elapsed = current_time - start_time

            if current_time - last_update < 0.1:
                continue
            last_update = current_time

            percent = parse_line(line)

            if percent is not None:
                filled = int(bar_length * percent / 100)
                bar = '█' * filled + ' ' * (bar_length - filled)
                print(f"\rInstalling {package_name}... [{bar}] {percent:.0f}% ({elapsed:.0f}s) ",
                      end='', flush=True)
            else:
                spin_char = spinner[int(elapsed) % len(spinner)]
                print(f"\rInstalling {package_name}... {spin_char} ({elapsed:.0f}s) ",
                      end='', flush=True)

        process.wait()

    except KeyboardInterrupt:
        print("\n\n⚠️  Installation interrupted by user (Ctrl+C). Terminating pip...")
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return False

    if process.returncode != 0:
        print(f"\n\n❌ Installation of {package_name} failed.")
        print(f"Error code: {process.returncode}")
        print("Try installing manually:")
        print(f"  {sys.executable} -m pip install {package_name}")
        return False

    print(f"\n\n✅ Successfully installed {package_name}!")
    return True

def install_library(package_name: str, is_heavy: bool = False) -> bool:
    """
    Generic library installer with user confirmation.
    Uses lightweight or heavy installation based on `is_heavy`.
    """
    if not check_internet():
        print(f"❌ No internet connection. Cannot install {package_name}.")
        return False

    install = input(f"{package_name} is not installed. Install now? (y/n): ")
    if not is_yes(install):
        print(f"Installation of {package_name} skipped.")
        return False

    if is_heavy and is_termux():
        print(f"\n⚠️  Termux: installing {package_name} may take 10–60 minutes!")
        print("Consider using `pkg` if available (e.g., `pkg install python-{package}`).")
        print("Proceeding with pip...")

    if is_heavy:
        return install_package_heavy(package_name)

    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', package_name
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ {package_name} installed.")
        return True
    except Exception as e:
        print(f"Failed to install {package_name}: {e}")
        return False

def get_library(import_func, package_names: list, is_heavy: bool = False):
    """
    Attempt to import a library. If missing, try installing alternatives.
    Exits if all options fail.
    """
    try:
        return import_func()
    except ImportError:
        pass

    for package in package_names:
        if install_library(package, is_heavy):
            try:
                return import_func()
            except ImportError:
                continue

    print(f"\n❌ Failed to install any of: {', '.join(package_names)}")
    print("Manual installation recommended:")
    for p in package_names:
        print(f"  pip install {p}")
    sys.exit(1)

# ==============================================================================
# CORE LIBRARY SETUP
# ==============================================================================

def get_numpy_or_alternative():
    """
    Try importing NumPy. Fall back to JAX if unavailable.
    Both provide compatible array APIs.
    """
    def try_numpy():
        import numpy as np
        return np

    def try_jax():
        import jax.numpy as jnp
        return jnp

    try:
        return try_numpy()
    except ImportError:
        pass

    if install_library('jax', is_heavy=True):
        try:
            return try_jax()
        except ImportError:
            pass

    print("\n❌ Failed to install either NumPy or JAX.")
    sys.exit(1)

def get_audio_library():
    """
    Return an audio loading/generation wrapper.
    Tries pydub (for MP3), then librosa or soundfile (for analysis).
    """
    def try_pydub():
        from pydub import AudioSegment
        return AudioSegment

    def try_librosa():
        import librosa
        class LibrosaWrapper:
            @staticmethod
            def from_file(path):
                y, sr = librosa.load(path, sr=None)
                return (y, sr)
        return LibrosaWrapper

    def try_soundfile():
        import soundfile as sf
        class SoundfileWrapper:
            @staticmethod
            def from_file(path):
                data, samplerate = sf.read(path)
                return (data, samplerate)
        return SoundfileWrapper

    try:
        aud = try_pydub()
        if not check_ffmpeg():
            print("\n⚠️  ffmpeg not found. pydub may not work properly.")
        return aud
    except ImportError:
        pass

    if install_library('librosa', is_heavy=True):
        try:
            return try_librosa()
        except ImportError:
            pass

    if install_library('soundfile', is_heavy=False):
        try:
            return try_soundfile()
        except ImportError:
            pass

    print("\n❌ No audio libraries could be installed.")
    sys.exit(1)

def get_plotting_library():
    """
    Return a plotting backend: matplotlib (non-interactive), plotly, or plotext.
    All wrapped to share a simple interface.
    """
    def try_matplotlib():
        import matplotlib
        matplotlib.use('Agg')  # Non-GUI backend (safe for Termux/servers)
        import matplotlib.pyplot as plt
        return plt

    def try_plotly():
        import plotly.graph_objects as go
        class PlotlyWrapper:
            def plot(self, x, y, title=""):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y))
                fig.update_layout(title=title)
                fig.show()
        return PlotlyWrapper()

    def try_plotext():
        import plotext as plt
        class PlotextWrapper:
            def plot(self, x, y, title=""):
                plt.clear_data()
                plt.plot(x, y)
                plt.title(title)
                plt.show()
        return PlotextWrapper()

    try:
        return try_matplotlib()
    except ImportError:
        pass

    if install_library('plotly', is_heavy=False):
        try:
            return try_plotly()
        except ImportError:
            pass

    if install_library('plotext', is_heavy=False):
        try:
            return try_plotext()
        except ImportError:
            pass

    print("\n❌ Failed to install any plotting library.")
    sys.exit(1)

# ==============================================================================
# USER INPUT HANDLING
# ==============================================================================

def get_input(prompt, default=None, min_val=None, max_val=None, type_func=float):
    """
    Robust input helper with validation, defaults, and range checks.
    Accepts int or float (with comma-to-dot conversion).
    """
    while True:
        hint = f" [{min_val}-{max_val}]" if min_val is not None and max_val is not None else ""
        if default is not None:
            hint += f" (default: {default})"
        user_input = input(f"{prompt}{hint}: ")
        if user_input == '' and default is not None:
            return default

        if type_func is float:
            user_input = user_input.replace(',', '.')

        try:
            value = type_func(user_input)
            if min_val is not None and value < min_val:
                print(f"Error: value must be ≥ {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"Error: value must be ≤ {max_val}.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_disk_space(path):
    """
    Return free disk space in bytes (or None on error).
    Used to warn about insufficient storage.
    """
    try:
        total, used, free = shutil.disk_usage(path)
        return free
    except Exception:
        return None

# ==============================================================================
# SIGNAL GENERATION UTILITIES
# ==============================================================================

def normalize_signal(np, signal, max_amplitude=0.99):
    """
    Scale signal so its peak amplitude doesn't exceed `max_amplitude`.
    Prevents clipping in audio exports.
    """
    max_abs = np.max(np.abs(signal))
    if max_abs > max_amplitude:
        return signal / max_abs * max_amplitude
    return signal

def generate_sin(np, duration, sample_rate, freq, amplitude):
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def generate_am(np, duration, sample_rate, carrier_freq, mod_freq, mod_depth, amplitude):
    """Generate an amplitude-modulated (AM) signal."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    mod = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
    return amplitude * mod * np.sin(2 * np.pi * carrier_freq * t)

def generate_pulse(np, duration, sample_rate, pulse_freq, duty_cycle, amplitude):
    """Generate a pulse wave (square wave with configurable duty cycle)."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    period = 1 / pulse_freq
    phase = (t % period) / period
    return amplitude * np.where(phase < duty_cycle, 1, -1)

def generate_noise(np, duration, sample_rate, amplitude, noise_type='uniform'):
    """Generate white noise (uniform or Gaussian)."""
    num_samples = int(sample_rate * duration)
    if noise_type == 'uniform':
        noise = np.random.uniform(-amplitude, amplitude, num_samples)
    elif noise_type == 'normal':
        noise = np.random.normal(0, amplitude, num_samples)
    else:
        raise ValueError("Unsupported noise type")

    return normalize_signal(np, noise, amplitude)

def generate_chm(np, duration, sample_rate, start_freq, end_freq, chm_type, amplitude):
    """
    Generate a frequency-modulated (chirp) signal with various sweep types:
    - linear
    - quadratic
    - hyperbolic
    """
    if duration <= 0:
        raise ValueError("Duration must be positive")
    if start_freq <= 0 or end_freq <= 0:
        raise ValueError("Frequencies must be positive for hyperbolic chirp")

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    T = duration

    if chm_type == 'linear':
        phase = 2 * np.pi * (start_freq * t + (end_freq - start_freq) * t**2 / (2 * T))
    elif chm_type == 'quadratic':
        phase = 2 * np.pi * (start_freq * t + (end_freq - start_freq) * t**3 / (3 * T**2))
    elif chm_type == 'hyperbolic':
        if abs(end_freq - start_freq) < 1e-12:
            phase = 2 * np.pi * start_freq * t
        else:
            inv_f0 = 1.0 / start_freq
            inv_f1 = 1.0 / end_freq
            a = (inv_f1 - inv_f0) / T
            b = inv_f0
            denom = a * t + b
            if np.any(denom <= 0):
                raise ValueError("Invalid parameters: instantaneous frequency becomes non-positive")
            if abs(a) < 1e-15:
                phase = 2 * np.pi * start_freq * t
            else:
                phase = 2 * np.pi * (np.log(denom) - np.log(b)) / a
    else:
        raise ValueError("Unsupported chirp type")

    return amplitude * np.sin(phase)

# ==============================================================================
# SIGNAL PARAMETER PROMPTS & GENERATION
# ==============================================================================

def get_signal_parameters(np, signal_type, sample_rate, stereo=False):
    """
    Prompt user for signal-specific parameters.
    Supports stereo mode (separate settings per channel).
    """
    params = {}
    if stereo:
        params['stereo'] = True
        print("\nLeft channel settings:")

    if signal_type == 'sin':
        params['freq'] = get_input("Frequency (Hz)", min_val=0.0, max_val=sample_rate/2)
        params['amplitude'] = get_input("Amplitude", min_val=0.0, max_val=1.0)
    elif signal_type == 'am':
        params['carrier_freq'] = get_input("Carrier frequency (Hz)", min_val=0.0, max_val=sample_rate/2)
        params['mod_freq'] = get_input("Modulation frequency (Hz)", min_val=0.0, max_val=sample_rate/2)
        params['mod_depth'] = get_input("Modulation depth", min_val=0.0, max_val=1.0)
        params['amplitude'] = get_input("Amplitude", min_val=0.0, max_val=1.0)
    elif signal_type == 'pulse':
        params['pulse_freq'] = get_input("Pulse frequency (Hz)", min_val=0.0, max_val=sample_rate/2)
        params['duty_cycle'] = get_input("Duty cycle", min_val=0.0, max_val=1.0)
        params['amplitude'] = get_input("Amplitude", min_val=0.0, max_val=1.0)
    elif signal_type == 'noise':
        noise_map = {
            '1': 'uniform', 'uniform': 'uniform',
            '2': 'normal', 'normal': 'normal'
        }
        noise_type_choice = input("Noise type (1=uniform, 2=normal) [1]: ").strip().lower()
        params['noise_type'] = noise_map.get(noise_type_choice, 'uniform')
        params['amplitude'] = get_input("Amplitude", min_val=0.0, max_val=1.0)
    elif signal_type == 'chm':
        params['start_freq'] = get_input("Start frequency (Hz)", min_val=0.0, max_val=sample_rate/2)
        params['end_freq'] = get_input("End frequency (Hz)", min_val=0.0, max_val=sample_rate/2)
        print("\nAvailable chirp types:")
        print("1. linear      – Linear sweep")
        print("2. quadratic   – Quadratic sweep")
        print("3. hyperbolic  – Hyperbolic sweep")
        chm_map = {
            '1': 'linear', 'linear': 'linear',
            '2': 'quadratic', 'quadratic': 'quadratic',
            '3': 'hyperbolic', 'hyperbolic': 'hyperbolic'
        }
        chm_type_choice = input("Select chirp type (1–3 or name): ").strip().lower()
        params['chm_type'] = chm_map.get(chm_type_choice, 'linear')
        params['amplitude'] = get_input("Amplitude", min_val=0.0, max_val=1.0)

    if stereo and signal_type != 'multi':
        print("\nRight channel settings (leave blank to copy left channel):")
        right_params = get_signal_parameters(np, signal_type, sample_rate, stereo=False)
        params['right_params'] = right_params

    return params

def generate_signal(np, signal_type, duration, sample_rate, channels, **kwargs):
    """
    Generate a mono or stereo signal of the specified type.
    In stereo mode, channels can differ if right_params are provided.
    """
    if signal_type == 'sin':
        signal = generate_sin(np, duration, sample_rate, kwargs['freq'], kwargs['amplitude'])
    elif signal_type == 'am':
        signal = generate_am(np, duration, sample_rate, kwargs['carrier_freq'],
                          kwargs['mod_freq'], kwargs['mod_depth'], kwargs['amplitude'])
    elif signal_type == 'pulse':
        signal = generate_pulse(np, duration, sample_rate, kwargs['pulse_freq'],
                             kwargs['duty_cycle'], kwargs['amplitude'])
    elif signal_type == 'noise':
        signal = generate_noise(np, duration, sample_rate, kwargs['amplitude'], kwargs['noise_type'])
    elif signal_type == 'chm':
        signal = generate_chm(np, duration, sample_rate, kwargs['start_freq'],
                           kwargs['end_freq'], kwargs['chm_type'], kwargs['amplitude'])
    else:
        raise ValueError("Unknown signal type")

    signal = normalize_signal(np, signal)

    if channels == 2:
        if 'stereo' in kwargs and kwargs['stereo'] and 'right_params' in kwargs:
            right_signal = generate_signal(np, signal_type, duration, sample_rate, 1, **kwargs['right_params'])
            right_signal = normalize_signal(np, right_signal)
            return np.column_stack((signal, right_signal))
        else:
            return np.column_stack((signal, signal))

    return signal

def generate_multi(np, duration, sample_rate, channels):
    """
    'Multi' mode: user adds multiple signals that are summed into one output.
    Supports stereo.
    """
    signals = []
    stereo_mode = channels == 2

    print("\nAdd signals (leave blank to finish):")
    while True:
        signal_type = input("Signal type (sin, am, pulse, noise, chm): ").strip().lower()
        if not signal_type:
            break
        if signal_type not in ['sin', 'am', 'pulse', 'noise', 'chm']:
            print("Invalid type. Allowed: sin, am, pulse, noise, chm")
            continue

        is_stereo = False
        if stereo_mode:
            stereo_choice = input("Use different signals for left/right channels? (y/n): ")
            is_stereo = is_yes(stereo_choice)

        try:
            params = get_signal_parameters(np, signal_type, sample_rate, is_stereo)
            params['stereo'] = is_stereo
            signal = generate_signal(np, signal_type, duration, sample_rate, channels, **params)
            signals.append(signal)
            print(f"Added {signal_type} signal.\n")
        except Exception as e:
            print(f"Error generating signal: {e}")
            continue

    if not signals:
        raise ValueError("No signals added")

    if channels == 1:
        combined = np.sum(signals, axis=0)
    else:
        left = np.sum([s[:, 0] for s in signals], axis=0)
        right = np.sum([s[:, 1] for s in signals], axis=0)
        combined = np.column_stack((left, right))

    combined = normalize_signal(np, combined)
    max_abs = np.max(np.abs(combined))
    if max_abs > 0.99:
        print(f"Signal normalized (peak amplitude: {max_abs:.4f})")

    return combined

# ==============================================================================
# OUTPUT FUNCTIONS
# ==============================================================================

def save_wav(np, filename, sample_rate, data, channels):
    """Save signal as 16-bit PCM WAV."""
    data = normalize_signal(np, data)

    if hasattr(np, 'int16'):
        data_int16 = np.int16(data * 32767)
    else:
        data_int16 = (data * 32767).to(dtype=np.int16)

    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data_int16.tobytes())

def save_csv(filename, data, channels):
    """Save signal data to CSV for external analysis."""
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if channels == 1:
                writer.writerow(['index', 'value'])
                for i, value in enumerate(data):
                    writer.writerow([i, value])
            else:
                writer.writerow(['index', 'left', 'right'])
                for i, (left, right) in enumerate(data):
                    writer.writerow([i, left, right])
    except Exception as e:
        print(f"Warning: failed to save CSV: {e}")

def save_mp3(np, AudioSegment, filename, sample_rate, data, channels):
    """Export to MP3 using pydub + ffmpeg."""
    data = normalize_signal(np, data)

    if hasattr(np, 'int16'):
        data_int16 = np.int16(data * 32767)
    else:
        data_int16 = (data * 32767).to(dtype=np.int16)

    audio = AudioSegment(
        data_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=channels
    )
    audio.export(filename, format='mp3')

def save_visualization(np, signal, output_dir, base_filename):
    """
    Save a plot of the signal (first 10k samples for performance).
    Supports PNG, SVG, PDF.
    """
    plt = get_plotting_library()
    if plt is None:
        return False

    try:
        plt.figure(figsize=(12, 5))

        if signal.ndim > 1:
            n_samples = min(10000, len(signal))
            print(f"Plotting first {n_samples} samples out of {len(signal)} for speed")
            plt.plot(signal[:n_samples, 0], 'b', label='Left channel')
            plt.plot(signal[:n_samples, 1], 'r', label='Right channel')
            plt.legend()
        else:
            n_samples = min(10000, len(signal))
            print(f"Plotting first {n_samples} samples out of {len(signal)} for speed")
            plt.plot(signal[:n_samples])

        plt.title('Generated Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.grid(True, linestyle='--', alpha=0.7)

        formats = ['png', 'svg', 'pdf']
        fmt = input(f"Visualization format ({'/'.join(formats)}) [default: png]: ").strip().lower()
        fmt = fmt if fmt in formats else 'png'
        viz_path = os.path.join(output_dir, f"{base_filename}_visualization.{fmt}")

        plt.savefig(viz_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✅ Visualization saved: {viz_path}")
        return True
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return False

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """
    Main program flow:
    1. Load NumPy (or fallback)
    2. Determine output directory
    3. Prompt for signal type & parameters
    4. Generate signal
    5. Save in selected formats
    6. (Optional) Save visualization
    """
    np = get_numpy_or_alternative()
    if np is None:
        print("This script requires a numerical array library (e.g., NumPy). Exiting.")
        sys.exit(1)

    output_dir = determine_output_directory()

    print("\n" + "="*50)
    print("AVAILABLE SIGNAL TYPES")
    print("="*50)
    print("1. sin      – Sine wave")
    print("2. am       – Amplitude modulation")
    print("3. pulse    – Pulse wave")
    print("4. noise    – White noise")
    print("   • uniform  – Uniform distribution")
    print("   • normal   – Gaussian distribution")
    print("5. chm      – Frequency-modulated chirp")
    print("   • linear      – Linear sweep")
    print("   • quadratic   – Quadratic sweep")
    print("   • hyperbolic  – Hyperbolic sweep")
    print("6. multi    – Multi-signal mode (add & sum multiple signals)")
    print("="*50)

    signal_map = {
        '1': 'sin', 'sin': 'sin',
        '2': 'am', 'am': 'am',
        '3': 'pulse', 'pulse': 'pulse',
        '4': 'noise', 'noise': 'noise',
        '5': 'chm', 'chm': 'chm',
        '6': 'multi', 'multi': 'multi'
    }

    signal_type = input("\nSelect signal type (1–6 or name): ").strip().lower()
    if signal_type not in signal_map:
        print("Error: invalid signal type")
        return
    signal_type = signal_map[signal_type]
    print(f"Selected: {signal_type}")

    duration = get_input("Signal duration (seconds)", min_val=0.001)
    sample_rate = get_input("Sample rate (Hz)", default=44100, min_val=1)
    channels = get_input("Number of channels (1/2)", default=1, min_val=1, max_val=2, type_func=int)

    # Estimate memory/disk usage
    num_samples = int(duration * sample_rate * channels)
    estimated_size = num_samples * 4  # ~4 bytes per float32

    free_space = get_disk_space(output_dir)
    if free_space and free_space < estimated_size * 1.5:
        print(f"\n⚠️  Insufficient disk space!")
        print(f"Required: {estimated_size / (1024*1024):.1f} MB")
        print(f"Available: {free_space / (1024*1024):.1f} MB")
        proceed = input("Continue anyway? (y/n): ")
        if not is_yes(proceed):
            print("Generation cancelled.")
            return

    if estimated_size > 500 * 1024 * 1024:
        print(f"\n⚠️  Warning: this will use ~{estimated_size / (1024*1024):.1f} MB of RAM")
        print("This may slow down or freeze your system.")
        proceed = input("Continue? (y/n): ")
        if not is_yes(proceed):
            print("Generation cancelled.")
            return

    raw_name = input("Output filename (without extension): ")
    output_filename = sanitize_filename(raw_name)

    try:
        if signal_type == 'multi':
            signal = generate_multi(np, duration, sample_rate, channels)
        else:
            params = get_signal_parameters(np, signal_type, sample_rate, channels == 2)
            params['stereo'] = channels == 2
            signal = generate_signal(np, signal_type, duration, sample_rate, channels, **params)

        print("\nChoose output format(s):")
        print("1. WAV")
        print("2. MP3 (requires ffmpeg + pydub)")
        print("3. Both")
        format_choice = get_input("Your choice", default=1, min_val=1, max_val=3, type_func=int)

        if format_choice in [1, 3]:
            output_wav = os.path.join(output_dir, output_filename + ".wav")
            save_wav(np, output_wav, sample_rate, signal, channels)
            print(f"\n✅ WAV saved to {output_wav}")

        if format_choice in [2, 3]:
            if not check_ffmpeg():
                print("⚠️  ffmpeg not found. Install ffmpeg to export MP3.")
                print("   Ubuntu/Debian: sudo apt install ffmpeg")
                print("   Windows: https://ffmpeg.org/download.html")
                print("   macOS: brew install ffmpeg")
            else:
                try:
                    from pydub import AudioSegment
                    output_mp3 = os.path.join(output_dir, output_filename + ".mp3")
                    save_mp3(np, AudioSegment, output_mp3, sample_rate, signal, channels)
                    print(f"✅ MP3 saved to {output_mp3}")
                except ImportError:
                    print("MP3 export unavailable: pydub not installed")

        output_csv = os.path.join(output_dir, output_filename + ".csv")
        save_csv(output_csv, signal, channels)
        print(f"CSV saved to {output_csv}")

        save_viz = input("\nSave signal visualization? (y/n): ")
        if is_yes(save_viz):
            save_visualization(np, signal, output_dir, output_filename)

    except Exception as e:
        print(f"❌ Error: {e}")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user (Ctrl+C). Exiting...")
        sys.exit(0)

# Author: KADAD0F
# License: MIT
