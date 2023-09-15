# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 09:00:26 2023

@author: Benny Simatupang
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

print("Nama: Benny Simatupang")
print("NRP: 5009211051")


t = np.linspace(0, 1, 1000, endpoint=False)

signal_freq1 = 5  
signal_freq2 = 20  
noise_amplitude = 0.5

clean_signal = np.sin(2 * np.pi * signal_freq1 * t) + np.sin(2 * np.pi * signal_freq2 * t)
noisy_signal = clean_signal + noise_amplitude * np.random.normal(size=len(t))

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title("Noisy Signal")
plt.plot(t, noisy_signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

fft_result = np.fft.fft(noisy_signal)
freq = np.fft.fftfreq(len(t))

plt.subplot(2, 2, 2)
plt.title("Frequency Domain (Magnitude)")
plt.plot(freq, np.abs(fft_result))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

cutoff_frequency = 10  # Cutoff frequency in Hz
order = 4  # Filter order
b, a = signal.butter(order, cutoff_frequency / (0.5 * len(t)), btype='low')
filtered_signal = signal.lfilter(b, a, noisy_signal)

plt.subplot(2, 2, 3)
plt.title("Filtered Signal")
plt.plot(t, filtered_signal)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

filtered_fft_result = np.fft.fft(filtered_signal)

plt.subplot(2, 2, 4)
plt.title("Filtered Signal Frequency Domain (Magnitude)")
plt.plot(freq, np.abs(filtered_fft_result))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
