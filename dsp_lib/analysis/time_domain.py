import numpy as np
from scipy.signal import hilbert

def calculate_peak(audio_array, in_dbfs=False):
    """
    Calculates the maximum absolute peak amplitude of the signal.
    
    Parameters:
        audio_array (np.ndarray): 1D audio array.
        in_dbfs (bool): If True, returns the value in Decibels Full Scale.
        
    Returns:
        float: The peak amplitude.
    """
    peak = np.max(np.abs(audio_array))
    
    if in_dbfs:
        # Prevent log10(0) if the array is complete silence
        peak = max(peak, 1e-10)
        return 20 * np.log10(peak)
        
    return peak

def calculate_rms(audio_array, in_dbfs=False):
    """
    Calculates the Root Mean Square (RMS) energy, representing the average loudness.
    
    Parameters:
        audio_array (np.ndarray): 1D audio array.
        in_dbfs (bool): If True, returns the value in Decibels Full Scale.
        
    Returns:
        float: The RMS value.
    """
    # Square the array, find the mean, and take the square root
    rms = np.sqrt(np.mean(audio_array**2))
    
    if in_dbfs:
        rms = max(rms, 1e-10)
        return 20 * np.log10(rms)
        
    return rms

def calculate_zero_crossing_rate(audio_array, sample_rate=44100, return_hz=False):
    """
    Calculates how often the signal crosses the zero-amplitude line.
    Useful for distinguishing between noisy and tonal sounds.
    
    Parameters:
        audio_array (np.ndarray): 1D audio array.
        sample_rate (int): Sample rate in Hz.
        return_hz (bool): If True, returns the estimated frequency of zero crossings.
                          If False, returns the raw count of crossings.
                          
    Returns:
        float or int: The zero-crossing metric.
    """
    # Find where the sign of the current sample differs from the next sample
    zero_crossings = np.where(np.diff(np.signbit(audio_array)))[0]
    count = len(zero_crossings)
    
    if return_hz:
        duration_sec = len(audio_array) / sample_rate
        # Divide by 2 because a full wave cycle crosses zero twice
        return (count / duration_sec) / 2.0
        
    return count

def extract_envelope(audio_array):
    """
    Extracts the amplitude envelope of a signal using the Analytic Signal (Hilbert Transform).
    This traces the smooth outer boundary of the waveform over time.
    
    Parameters:
        audio_array (np.ndarray): 1D audio array.
        
    Returns:
        np.ndarray: A 1D array representing the envelope contour.
    """
    # The Hilbert transform creates a complex analytic signal
    analytic_signal = hilbert(audio_array)
    
    # The magnitude of the analytic signal is the true amplitude envelope
    amplitude_envelope = np.abs(analytic_signal)
    
    return amplitude_envelope