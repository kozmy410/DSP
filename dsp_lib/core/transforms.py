import numpy as np
from scipy import signal

def apply_window(audio_array, window_type='hann', **kwargs):
    """
    Applies a windowing function to mitigate spectral leakage.
    
    Parameters:
        audio_array (np.ndarray): 1D audio array.
        window_type (str): 'hann', 'hamming', 'blackman', 'kaiser', or 'rectangular'.
        **kwargs: Additional parameters for specific windows (e.g., 'beta' for kaiser).
        
    Returns:
        np.ndarray: Windowed audio array.
    """
    n_samples = len(audio_array)
    window_type = window_type.lower()
    
    if window_type == 'hann':
        window = np.hanning(n_samples)
    elif window_type == 'hamming':
        window = np.hamming(n_samples)
    elif window_type == 'blackman':
        window = np.blackman(n_samples)
    elif window_type == 'kaiser':
        # Beta determines the shape. 14.0 provides excellent side-lobe rejection
        beta = kwargs.get('beta', 14.0)
        window = np.kaiser(n_samples, beta)
    else:
        window = np.ones(n_samples) # Rectangular (no alteration)
        
    return audio_array * window

def compute_fft(audio_array, sample_rate=44100, window_type='hann', fft_type='rfft', return_complex=False, **kwargs):
    """
    Computes the Fast Fourier Transform to extract frequency data.
    
    Parameters:
        audio_array (np.ndarray): 1D audio array.
        sample_rate (int): Sample rate in Hz.
        window_type (str): Windowing function to apply before FFT.
        fft_type (str): 'rfft' for Real FFT (standard audio), 'fft' for Complex FFT.
        return_complex (bool): If True, returns the raw complex data instead of magnitude and phase.
        **kwargs: Passed down to the apply_window function.
        
    Returns:
        If return_complex is True:
            tuple: (frequencies_array, complex_spectrum)
        Else:
            tuple: (frequencies_array, magnitude_db_array, phase_array)
    """
    # Apply the window, passing any extra arguments (like beta) down the chain
    windowed_audio = apply_window(audio_array, window_type, **kwargs)
    n_samples = len(windowed_audio)
    fft_type = fft_type.lower()
    
    if fft_type == 'rfft':
        fft_result = np.fft.rfft(windowed_audio)
        frequencies = np.fft.rfftfreq(n_samples, d=1.0/sample_rate)
        
        magnitude = np.abs(fft_result) / (n_samples / 2.0)
        magnitude[0] = magnitude[0] / 2.0
        
    elif fft_type == 'fft':
        fft_result = np.fft.fft(windowed_audio)
        frequencies = np.fft.fftfreq(n_samples, d=1.0/sample_rate)
        
        magnitude = np.abs(fft_result) / n_samples
        
    else:
        raise ValueError(f"Unsupported fft_type: '{fft_type}'. Use 'rfft' or 'fft'.")
    
    # Return the raw data if we intend to run an Inverse FFT later
    if return_complex:
        return frequencies, fft_result
        
    # Convert linear magnitude to Decibels (dBFS)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
    
    # Extract phase in radians (-pi to pi)
    phase = np.angle(fft_result) 
    
    return frequencies, magnitude_db, phase

def compute_ifft(complex_spectrum, n_samples=None, fft_type='rfft'):
    """
    Computes the Inverse Fast Fourier Transform to reconstruct time-domain audio.
    
    Parameters:
        complex_spectrum (np.ndarray): The raw complex array from compute_fft.
        n_samples (int): The target length of the time-domain array (fixes odd/even length issues).
        fft_type (str): Must match the type used to generate the spectrum ('rfft' or 'fft').
        
    Returns:
        np.ndarray: Reconstructed time-domain audio array.
    """
    fft_type = fft_type.lower()
    
    if fft_type == 'rfft':
        # irfft automatically returns real numbers
        audio_array = np.fft.irfft(complex_spectrum, n=n_samples)
    elif fft_type == 'fft':
        # ifft returns complex numbers, but physical audio is strictly real
        audio_array = np.real(np.fft.ifft(complex_spectrum, n=n_samples))
    else:
        raise ValueError("Unsupported fft_type. Use 'rfft' or 'fft'.")
        
    return audio_array

def compute_stft(audio_array, sample_rate=44100, window_type='hann', nperseg=2048, noverlap=1024):
    """
    Computes the Short-Time Fourier Transform for spectrogram generation.
    
    Parameters:
        audio_array (np.ndarray): 1D audio array.
        sample_rate (int): Sample rate in Hz.
        window_type (str): Windowing function for each chunk.
        nperseg (int): Number of samples per chunk (determines frequency resolution).
        noverlap (int): Number of overlapping samples (determines time resolution).
        
    Returns:
        tuple: (frequencies, times, complex_stft_matrix)
    """
    frequencies, times, Zxx = signal.stft(
        audio_array, 
        fs=sample_rate, 
        window=window_type, 
        nperseg=nperseg, 
        noverlap=noverlap
    )
    
    return frequencies, times, Zxx