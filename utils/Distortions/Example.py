from distortions import Distortions

if __name__ == '__main__':
    # Example usage
    file_path = "unofficial_corpus\\voice-british-1.wav" #TO BE SET
    output_path = 'Src'         #TO BE SET

    # Create an instance of the Distortions class
    distort = Distortions(file_path)
    distort.plot_wave("basic_plot.png")

    # Define the desired effects as a dictionary of attributes
    effects_to_apply = {
        "pitch_shift": 2,              # Shift pitch by +2 semitones
        "time_stretch": 0.8,           # Stretch time by a factor of 0.8 (slows down)
        "domain_filter": {                    # Apply a filter with specified parameters
            "cutoff": 1000,
            "filter_type": "high",
            "order": 5
        },
        "sampling_rate": 14100,
        "bit_depth": 4
    }

    # Apply selected effects
    distort.apply_effects(effects_to_apply)
    distort.plot_wave("modified_wave.png")

    # Save the modified audio
    distort.save_audio(output_path)
