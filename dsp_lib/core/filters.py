import numpy as np
from scipy import signal

# ==========================================
# OFFLINE FILTERS (Zero Phase Distortion)
# Best for processing complete arrays 
# ==========================================

def apply_lowpass_filter(audio_array, cutoff_freq, sample_rate=44100, order=4):
    """Applies a Butterworth low-pass filter (offline/zero-phase)."""
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, audio_array)

def apply_highpass_filter(audio_array, cutoff_freq, sample_rate=44100, order=4):
    """Applies a Butterworth high-pass filter (offline/zero-phase)."""
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, audio_array)

def apply_bandpass_filter(audio_array, lowcut, highcut, sample_rate=44100, order=4):
    """
    Applies a Butterworth band-pass filter.
    Passes frequencies between lowcut and highcut.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    return signal.filtfilt(b, a, audio_array)

def apply_notch_filter(audio_array, notch_freq, sample_rate=44100, quality_factor=30.0):
    """
    Applies an IIR notch filter.
    Surgically removes a specific frequency (e.g., 50Hz/60Hz mains hum).
    """
    nyquist = 0.5 * sample_rate
    normalized_notch = notch_freq / nyquist
    b, a = signal.iirnotch(normalized_notch, quality_factor)
    return signal.filtfilt(b, a, audio_array)


# ==========================================
# REAL-TIME STREAMING FILTERS
# Best for live GUI audio processing
# ==========================================

class RealTimeFilter:
    """
    A stateful filter designed to process chunks of live audio sequentially.
    Maintains the filter memory (state) between chunks to prevent clicking.
    """
    def __init__(self, filter_type='lowpass', cutoff=1000.0, sample_rate=44100, order=4, q_factor=30.0):
        self.sample_rate = sample_rate
        self.nyquist = 0.5 * sample_rate
        self.filter_type = filter_type.lower()
        
        # Calculate filter coefficients based on the type
        if self.filter_type == 'lowpass':
            self.b, self.a = signal.butter(order, cutoff / self.nyquist, btype='low')
        elif self.filter_type == 'highpass':
            self.b, self.a = signal.butter(order, cutoff / self.nyquist, btype='high')
        elif self.filter_type == 'notch':
            self.b, self.a = signal.iirnotch(cutoff / self.nyquist, q_factor)
        else:
            raise ValueError("Unsupported filter type. Use 'lowpass', 'highpass', or 'notch'.")
            
        # Initialize filter state history (zi) with zeros
        self.zi = signal.lfilter_zi(self.b, self.a)

    def process_chunk(self, audio_chunk):
        """
        Processes a single block of audio and updates the internal filter state.
        
        Parameters:
            audio_chunk (np.ndarray): The incoming live audio block.
            
        Returns:
            np.ndarray: The filtered audio block.
        """
        # lfilter uses the previous state (self.zi) to compute the next block,
        # and returns the filtered audio alongside the new updated state (zf).
        filtered_chunk, zf = signal.lfilter(self.b, self.a, audio_chunk, zi=self.zi)
        self.zi = zf  # Save the state for the next chunk
        
        return filtered_chunk