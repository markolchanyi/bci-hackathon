import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the third Excel file into a DataFrame
file_path = '/Users/markolchanyi/Desktop/training_data/mark_tongue_left_right_60k_samples_session_3.xlsx'  # Update this to your actual file path
df = pd.read_excel(file_path)

# Parameters
sampling_rate = 500  # Hz
channel = 'Channel_1'  # Choose which channel to plot, e.g., Channel_1

# Convert the signal from the specified channel into an MNE-compatible data structure
data = df[[channel]].values.T  # Transpose to get shape (n_channels, n_times)
info = mne.create_info(ch_names=[channel], sfreq=sampling_rate, ch_types=['eeg'])
raw = mne.io.RawArray(data, info)

# Creating artificial events to segment the continuous data into epochs
events = mne.make_fixed_length_events(raw, id=1, duration=0.5)  # Duration in seconds

# Create epochs
epochs = mne.Epochs(raw, events, tmin=0, tmax=0.5, baseline=None, preload=True)

# Define frequencies of interest
fmin, fmax = 1, 50  # Frequency range in Hz
freqs = np.linspace(fmin, fmax, int((fmax - fmin) * 2) + 1)  # Define frequencies from fmin to fmax

# Compute the multitaper FFT spectrogram
power = mne.time_frequency.tfr_multitaper(epochs, freqs=freqs, n_cycles=2, time_bandwidth=4.0, return_itc=False)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Spectrogram plot
power.plot([0], baseline=None, mode='logratio', axes=axs[0], show=False, colorbar=False)
axs[0].set_title(f'Multitaper FFT Spectrogram of {channel}')
axs[0].set_ylabel('Frequency (Hz)')

# KeyPress events plot
axs[1].plot(df['Timestamp'], df['KeyPress'], color='tab:red', marker='o', linestyle='None', markersize=2)
axs[1].set_ylabel('KeyPress')
axs[1].set_xlabel('Time (s)')
axs[1].set_title('KeyPress Events')
axs[1].set_ylim(-0.5, 2.5)  # Assuming KeyPress values are 0, 1, 2
axs[1].set_yticks([0, 1, 2])  # Set y-ticks to match KeyPress values

plt.tight_layout()
plt.show()
