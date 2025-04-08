#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FFT and IFFT Visualization Example.

This script demonstrates the usage of Fast Fourier Transform (FFT) and 
Inverse Fast Fourier Transform (IFFT) through a practical example.
It generates a signal composed of multiple sine waves, performs FFT to 
analyze the frequency components, and then reconstructs the signal using IFFT.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def generate_signal(t, frequencies, amplitudes):
    """
    Generate a composite signal from multiple sine waves.
    
    Args:
        t (numpy.ndarray): Time array.
        frequencies (list): List of frequencies in Hz for each component.
        amplitudes (list): List of amplitudes for each component.
    
    Returns:
        numpy.ndarray: The composite signal.
    """
    signal = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
    return signal

def perform_fft(signal, sampling_rate):
    """
    Perform Fast Fourier Transform on a signal.
    
    Args:
        signal (numpy.ndarray): Input signal to transform.
        sampling_rate (float): Sampling rate of the signal in Hz.
    
    Returns:
        tuple: Frequency array and FFT magnitude.
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result) / n  # Normalize
    
    # For real signals, the FFT is symmetric, so we take only the first half
    magnitude = magnitude[:n//2]
    
    # Calculate corresponding frequencies
    freq = np.fft.fftfreq(n, d=1/sampling_rate)[:n//2]
    
    return freq, magnitude

def perform_ifft(fft_result):
    """
    Perform Inverse Fast Fourier Transform.
    
    Args:
        fft_result (numpy.ndarray): FFT result to transform back.
    
    Returns:
        numpy.ndarray: Time-domain signal.
    """
    return np.fft.ifft(fft_result).real

def visualize_results(t, original_signal, reconstructed_signal, freq, magnitude, experiment_id):
    """
    Visualize the original signal, FFT results, and reconstructed signal.
    
    Args:
        t (numpy.ndarray): Time array.
        original_signal (numpy.ndarray): Original time-domain signal.
        reconstructed_signal (numpy.ndarray): Reconstructed signal after IFFT.
        freq (numpy.ndarray): Frequency array.
        magnitude (numpy.ndarray): FFT magnitude.
        experiment_id (str): Identifier for the experiment.
    
    Returns:
        None
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot original signal
    axs[0].plot(t, original_signal)
    axs[0].set_title(f'Original Signal - {experiment_id}')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    
    # Plot frequency spectrum (fixed stem function call)
    axs[1].stem(freq, magnitude)
    axs[1].set_title(f'Frequency Spectrum - {experiment_id}')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].grid(True)
    
    # Plot reconstructed signal
    axs[2].plot(t, reconstructed_signal)
    axs[2].set_title(f'Reconstructed Signal - {experiment_id}')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Amplitude')
    axs[2].grid(True)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Save the figure with experiment ID and date in the filename
    plt.savefig(f'output/fft_visualization_{experiment_id}.png')
    plt.show()

def main():
    """
    Main function demonstrating FFT and IFFT with visualization.
    
    This function generates a signal, performs FFT, reconstructs the signal with IFFT,
    and visualizes the results.
    """
    # Create experiment ID with date
    date_str = datetime.now().strftime('%Y%m%d')
    experiment_id = f'EXP001_{date_str}_visualize_results'
    
    print(f"Starting FFT/IFFT demonstration - {experiment_id}")
    
    # Parameters
    duration = 1.0  # seconds
    sampling_rate = 1000.0  # Hz
    
    # Create time array
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    
    # Generate a test signal (sum of sines)
    frequencies = [5, 50, 120]  # Hz
    amplitudes = [1.0, 0.5, 0.3]
    original_signal = generate_signal(t, frequencies, amplitudes)
    
    # Add some noise
    noise = np.random.normal(0, 0.1, len(t))
    noisy_signal = original_signal + noise
    
    # Perform FFT
    freq, magnitude = perform_fft(noisy_signal, sampling_rate)
    
    # Perform IFFT on the full FFT result
    fft_result = np.fft.fft(noisy_signal)
    reconstructed_signal = perform_ifft(fft_result)
    
    # Visualize results
    visualize_results(t, noisy_signal, reconstructed_signal, freq, magnitude, experiment_id)
    
    # Save results to CSV
    results = np.column_stack((t, original_signal, noisy_signal, reconstructed_signal))
    header = "time,original_signal,noisy_signal,reconstructed_signal"
    np.savetxt(f'output/fft_data_{experiment_id}.csv', results, delimiter=',', header=header)
    
    print(f"FFT/IFFT demonstration completed - {experiment_id}")
    print(f"Results saved to output/fft_data_{experiment_id}.csv")
    print(f"Visualization saved to output/fft_visualization_{experiment_id}.png")

if __name__ == "__main__":
    main()