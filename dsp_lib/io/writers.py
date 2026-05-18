import numpy as np
import soundfile as sf
import os

def write_wav(filename, audio_array, sample_rate=44100, subtype='PCM_24', normalize=False):
    """
    Writes a NumPy array to a WAV file with selectable encoding formats.
    
    Parameters:
        filename (str): The output file path (e.g., 'output.wav').
        audio_array (np.ndarray): The audio data array.
        sample_rate (int): The sample rate in Hz.
        subtype (str): The encoding format. Options: 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT'.
        normalize (bool): If True, scales the audio to peak at 0 dBFS.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    # 1. Validate the requested format subtype
    valid_subtypes = ['PCM_16', 'PCM_24', 'PCM_32', 'FLOAT']
    subtype = subtype.upper()
    
    if subtype not in valid_subtypes:
        print(f"Warning: Invalid format '{subtype}'. Defaulting to 'PCM_24'.")
        subtype = 'PCM_24'

    # 2. Enforce the strict 4.0 second maximum duration limit
    max_samples = int(4.0 * sample_rate)
    if len(audio_array) > max_samples:
        print(f"Warning: Array exceeds limits. Truncating before writing to {filename}.")
        audio_array = audio_array[:max_samples]

    # 3. Handle clipping and optional normalization
    max_amplitude = np.max(np.abs(audio_array))
    
    if normalize and max_amplitude > 0:
        audio_array = audio_array / max_amplitude
    elif max_amplitude > 1.0 and subtype != 'FLOAT':
        # Floating point WAVs can technically exceed 1.0, but integer PCM will clip hard
        print(f"CLIPPING DETECTED: Values in {filename} exceed 1.0. Audio will distort.")

    # 4. Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # 5. Write to disk
    try:
        sf.write(filename, audio_array, sample_rate, subtype=subtype)
        print(f"Successfully wrote: {filename} (Format: {subtype})")
        return True
    except Exception as e:
        print(f"Error writing {filename}: {e}")
        return False