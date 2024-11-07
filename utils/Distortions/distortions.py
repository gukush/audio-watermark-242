import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, fftconvolve


class Distortions:
    # Universal dictionary for encoding all modifications
    POS_DISTORTIONS_DICT = {
        'pitch_shift': '1',
        'time_stretch': '2',
        'domain_filter': '3',
        'sampling_rate': '4',
        'bit_depth': '5'

        ####'5'
    }

    def __init__(self, file_path):
        # Load the audio file
        self.file_path = file_path
        self.output_filename = self.file_path  # changes with adding distortions
        self.y, self.sr = librosa.load(self.file_path, sr=None)
    
    def reverberation(self, decay_f, delay_f):
        delay = int(delay_f * self.sr)
        decay = decay_f

        echo_data = np.zeros_like(self.y)
        
        for i in range(delay, len(self.y)):
            echo_data[i] = self.y[i] + decay * self.y[i - delay]
        
        # Normalization
        #max_val = np.max(np.abs(echo_data))
        #echo_data = echo_data / max_val * 0.5
        self.y = echo_data

    def pitch_shift(self, n_steps):
        """
        Changes the pitch (tone?) of voice, positive values make the pitch higher by parsed number of
        steps, negative makes the pitch lower
        """
        self.y = librosa.effects.pitch_shift(self.y, sr = self.sr, n_steps = n_steps)

    def time_stretch(self, rate):
        """
        Changes the speed of the audio without affecting pitch.

        :param rate: Stretch factor (values > 1 speed up, values < 1 slow down).
        :return: None
        """
        self.y = librosa.effects.time_stretch(self.y, rate = rate)

    def domain_filter(self, cutoff, filter_type='low', order=5):
        """
        Applies a Butterworth filter to the audio.

        :param cutoff: Frequency cutoff for the filter.
        :param filter_type: Type of filter ('low' for low-pass, 'high' for high-pass).
        :param order: Filter order (higher = sharper cutoff).
        :return: None
        """
        nyquist = 0.5 * self.sr
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
        self.y = lfilter(b, a, self.y)

    def set_sampling_rate(self, new_sr):
        """
        Changes the sampling rate of the audio by resampling.

        :param new_sr: The new sampling rate to apply.
        :return: None
        """
        if new_sr != self.sr:
            self.y = librosa.resample(self.y, orig_sr=self.sr, target_sr=new_sr)
            self.sr = new_sr

    def reduce_bit_depth(self, bit_depth):
        """
        Reduces the bit depth of the audio by quantizing it.

        :param bit_depth: The new bit depth to apply (e.g., 8, 12, 16).
        :return: None
        """
        max_val = np.max(np.abs(self.y))  # Normalize to avoid overflow
        y_normalized = self.y / max_val  # Scale signal to -1 to 1

        # Calculate the number of levels for quantization
        num_levels = 2 ** bit_depth
        y_quantized = np.round(y_normalized * (num_levels / 2)) / (num_levels / 2)
        print(y_quantized, max_val, self.y)
        # Scale back to the original amplitude range
        self.y = y_quantized * max_val

    def adjust_file_name(self, attributes):
        """
        Encode applied distortions in the output file name
        """
        original_name = self.file_path.split("\\")[-1]
        name, ext = original_name.split(".")

        # Access values in POS_DISTORTIONS_DICT based on keys in effects_to_apply
        applied_effect_keys = list(attributes.keys())  # Get the list of keys
        applied_effect_values = [Distortions.POS_DISTORTIONS_DICT[key] for key in applied_effect_keys if
                                 key in Distortions.POS_DISTORTIONS_DICT]
        if applied_effect_values:
            # Join suffixes to create the new filename
            suffix_str = "_".join(applied_effect_values)
            self.output_filename = f"{name}_distorted_{suffix_str}.{ext}"

    def apply_effects(self, attributes):
        """
        Applies selected distortions based on provided attributes.

        :param attributes: Dictionary of attributes to apply.
                           Supported keys are 'pitch_shift', 'time_stretch', 'domain_filter'.
        """
        if 'pitch_shift' in attributes:
            n_steps = attributes['pitch_shift']
            print(f"Applying pitch shift by {n_steps} semitones.")
            self.pitch_shift(n_steps)

        if 'time_stretch' in attributes:
            rate = attributes['time_stretch']
            print(f"Applying time stretch with rate {rate}.")
            self.time_stretch(rate)

        if 'domain_filter' in attributes:
            # Expecting all filter parameters to be provided in the dictionary
            filter_params = attributes['domain_filter']
            cutoff = filter_params['cutoff']  # Required parameter
            filter_type = filter_params['filter_type']  # Optional, default 'low'
            order = filter_params.get('order', 5)  # Optional, default 5
            print(f"Applying {filter_type} filter with cutoff {cutoff} Hz and order {order}.")
            self.domain_filter(cutoff, filter_type, order)
        
        if 'reverberation' in attributes:
            filter_params = attributes['reverberation']
            decay = filter_params['decay']
            delay = filter_params['delay']
            print(f"Applying reverberation with {decay} decay and {delay} delay.")
            self.reverberation(decay, delay)

        if 'sampling_rate' in attributes:
            new_sr = attributes['sampling_rate']
            print(f"Changing sampling rate to {new_sr} Hz.")
            self.set_sampling_rate(new_sr)

        if 'bit_depth' in attributes:
            bit_depth = attributes['bit_depth']
            print(f"Reducing bit depth to {bit_depth} bits.")
            self.reduce_bit_depth(bit_depth)

        self.adjust_file_name(attributes)  # Change output file name

    def plot_wave(self, filename, x_start=0, x_end=100, x_label='Time', y_label='Y-axis'):
        """
        Plots the distorted data and saves it as an image.

        Parameters:
        - y: numpy array, the distorted data to plot.
        - x_start: float, starting value for the x-axis (default is 0).
        - x_end: float, ending value for the x-axis (default is 100).
        - x_label: str, label for the x-axis (default is 'Time').
        - y_label: str, label for the y-axis (default is 'Y-axis').
        - filename: str, name of the file to save the plot as (default is 'result_plot.png').
        """
        # Create the x-axis values to match the length of y
        x = np.linspace(x_start, x_end, len(self.y))

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(x, self.y, label='Distorted Data')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title('Distorted Data Plot')
        plt.legend()

        # Save the plot
        plt.savefig(filename)
        plt.close()  # Close the plot to free memory

        print(f"Plot saved as {filename}")

    def save_audio(self, output_path):
        """
        Saves the processed audio to a file.

        :param output_path: Path to save the processed audio file.
        """
        output_path = f"{output_path}//{self.output_filename}"
        sf.write(output_path, self.y, self.sr)

