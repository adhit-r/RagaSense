#!/usr/bin/env python3
"""
Generate Test Audio

This script generates a simple test audio file with a known raga pattern
that can be used for testing the audio analysis endpoint.
"""
import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

def generate_raga_audio(output_file, duration=10, sample_rate=44100):
    """
    Generate a simple audio file with a raga-like pattern.
    
    Args:
        output_file (str): Path to save the output WAV file
        duration (float): Duration of the audio in seconds
        sample_rate (int): Sample rate in Hz
    """
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Define the notes for Yaman raga (C major scale for simplicity)
    # Frequencies for one octave in C major (C4 to B4)
    notes = {
        'S': 261.63,  # C4
        'R': 293.66,  # D4
        'G': 329.63,  # E4
        'M': 349.23,  # F4
        'P': 392.00,  # G4
        'D': 440.00,  # A4
        'N': 493.88,  # B4
        "S'": 523.25  # C5
    }
    
    # Simple arohana (ascending) and avarohana (descending) pattern for Yaman
    pattern = ['S', 'R', 'G', 'M', 'P', 'D', 'N', "S'"]
    
    # Generate the audio signal
    audio = np.array([], dtype=np.float32)
    note_duration = 0.5  # seconds per note
    samples_per_note = int(sample_rate * note_duration)
    
    # Arohana (ascending)
    for note in pattern:
        freq = notes[note]
        t_note = np.linspace(0, note_duration, samples_per_note, endpoint=False)
        note_wave = 0.5 * np.sin(2 * np.pi * freq * t_note)
        # Apply a simple envelope to avoid clicks
        envelope = np.ones_like(note_wave)
        attack = int(0.05 * sample_rate)  # 50ms attack
        release = int(0.1 * sample_rate)  # 100ms release
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        audio = np.concatenate([audio, note_wave * envelope])
    
    # Avarohana (descending)
    for note in reversed(pattern[:-1]):  # Skip the top S' coming down
        freq = notes[note]
        t_note = np.linspace(0, note_duration, samples_per_note, endpoint=False)
        note_wave = 0.5 * np.sin(2 * np.pi * freq * t_note)
        # Apply a simple envelope to avoid clicks
        envelope = np.ones_like(note_wave)
        attack = int(0.05 * sample_rate)  # 50ms attack
        release = int(0.1 * sample_rate)  # 100ms release
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        audio = np.concatenate([audio, note_wave * envelope])
    
    # Repeat the pattern to fill the duration
    pattern_duration = len(pattern) * 2 * note_duration - note_duration
    repeat_count = int(duration / pattern_duration) + 1
    audio = np.tile(audio, repeat_count)
    audio = audio[:int(duration * sample_rate)]
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save as WAV file
    wavfile.write(output_file, sample_rate, audio_int16)
    print(f"Generated test audio file: {output_file}")
    
    # Plot a spectrogram
    plt.figure(figsize=(12, 6))
    f, t, Sxx = signal.spectrogram(audio, fs=sample_rate, nperseg=1024)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of Generated Raga Yaman')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    
    # Save the spectrogram
    spectrogram_file = os.path.splitext(output_file)[0] + '_spectrogram.png'
    plt.savefig(spectrogram_file)
    print(f"Saved spectrogram: {spectrogram_file}")
    plt.close()

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "test_raga_yaman.wav")
    generate_raga_audio(output_file, duration=15)
