import numpy as np
import soundfile as sf
import os
from scipy import signal

def get_file_metadata(filename):
    """
    Reads the header of an audio file to extract information without loading the audio data.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: The file {filename} was not found.")
        
    info = sf.info(filename)
    
    metadata = {
        "filename": os.path.basename(filename),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
        "original_duration_sec": info.frames / info.samplerate,
        "frames": info.frames
    }
    
    return metadata

def read_wav(filename, force_mono=True, target_sr=None, normalize=False):
    """
    Reads an audio file of any length and handles resampling/normalization.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: The file {filename} was not found.")
        
    info = sf.info(filename)
    original_sr = info.samplerate
    
    # Read the entire file into memory
    audio_array, _ = sf.read(filename, dtype='float64')
    
    if force_mono and len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
        
    final_sr = original_sr
    if target_sr is not None and original_sr != target_sr:
        audio_array = signal.resample_poly(audio_array, target_sr, original_sr)
        final_sr = target_sr
        
    if normalize:
        max_amplitude = np.max(np.abs(audio_array))
        if max_amplitude > 0:
            audio_array = audio_array / max_amplitude
            
    return audio_array, final_sr