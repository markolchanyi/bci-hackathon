import numpy as np
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
from scipy.signal import stft
import time

def plot_spectrogram(data, fs=500, window='hann', nperseg=None, noverlap=None):
    """
    Plot a spectrogram of the data.

    Parameters:
    - data: 2D array of shape (channels, samples)
    - fs: Sampling frequency of the EEG data
    - window: Type of window function used for STFT
    - nperseg: Length of each segment for STFT (automatically determined if None)
    - noverlap: Number of points to overlap between segments (automatically determined if None)
    """
    channels, samples = data.shape

    for i in range(channels):
        channel_data = data[i, :]

        # Dynamically adjust nperseg and noverlap if not provided
        if nperseg is None or nperseg > samples:
            nperseg = samples
        if noverlap is None or noverlap >= nperseg:
            noverlap = max(1, nperseg // 2 - 1)

        f, t, Zxx = stft(channel_data, fs, window=window, nperseg=nperseg, noverlap=noverlap)
        plt.figure()
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.title(f'STFT Magnitude - Channel {i+1}')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity')
    plt.show()

def main():
    print("Looking for an EEG stream...")
    streams = resolve_stream('source_id', '102801-0077 500')

    # Create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    # Define the duration for which to collect data before plotting
    collection_duration = 1  # in seconds
    fs = 500  # Sampling rate for the EEG device
    num_channels = 11  # Adjust this based on the actual number of channels in your EEG data
    buffer_length = collection_duration * fs  # Number of samples to collect per channel
    data_buffer = np.empty((num_channels, 0))

    start_time = time.time()
    while True:
        # Get a new sample
        sample, timestamp = inlet.pull_sample()
        # Append the new sample to the data buffer
        data_buffer = np.hstack((data_buffer, np.array(sample).reshape(-1, 1)))  # Reshape sample and append
        if data_buffer.shape[1] > buffer_length:
            data_buffer = data_buffer[:, -buffer_length:]  # Keep only the latest 'buffer_length' samples

        # Check if enough data has been collected
        if time.time() - start_time > collection_duration:
            # Plot the spectrogram of the collected data
            plot_spectrogram(data_buffer, fs=fs)
            # Reset the start time and clear the buffer for the next collection period
            start_time = time.time()
            data_buffer = np.empty((num_channels, 0))

if __name__ == '__main__':
    main()
