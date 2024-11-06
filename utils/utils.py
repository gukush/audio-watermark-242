from matplotlib import pyplot as plt
import torchaudio
import numpy as np
import torch

# Source: https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)


def plot_waveform(audio_array, sample_rate, filename="plot.png"):
    """
    Plots the waveform of an audio signal from a numpy array and saves it to a PNG file.

    Parameters:
    - audio_array (numpy.ndarray): The audio signal array.
    - sample_rate (int): The sample rate of the audio signal in Hz.
    - filename (str): The filename to save the plot as a PNG image.
    """
    # Calculate time array based on the sample rate and length of audio data
    time = np.linspace(0, len(audio_array) / sample_rate, num=len(audio_array))

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio_array, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_waveform_old(waveform, sample_rate):
    if isinstance(waveform,torch.Tensor):
        waveform = waveform.numpy()
    elif isinstance(waveform, np.ndarray):
        pass

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    return figure