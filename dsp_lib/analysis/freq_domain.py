import numpy as np

def calculate_spectral_centroid(frequencies, magnitudes):
    """
    Calculates the Spectral Centroid, which indicates where the "center of mass"
    of the spectrum is located. Correlates with the perceived "brightness" of a sound.
    
    Parameters:
        frequencies (np.ndarray): Array of frequency bins (from FFT).
        magnitudes (np.ndarray): Array of linear magnitudes (from FFT, NOT in dB).
        
    Returns:
        float: The spectral centroid in Hz.
    """
    # Prevent division by zero if the array is completely silent
    magnitude_sum = np.sum(magnitudes)
    if magnitude_sum == 0:
        return 0.0
        
    # Centroid is the weighted mean of the frequencies
    centroid = np.sum(frequencies * magnitudes) / magnitude_sum
    return centroid

def calculate_spectral_rolloff(frequencies, magnitudes, rolloff_percent=0.85):
    """
    Calculates the Spectral Rolloff point.
    This is the frequency below which a specified percentage of the total spectral energy lies.
    
    Parameters:
        frequencies (np.ndarray): Array of frequency bins.
        magnitudes (np.ndarray): Array of linear magnitudes.
        rolloff_percent (float): The percentage threshold (typically 0.85 or 0.95).
        
    Returns:
        float: The rolloff frequency in Hz.
    """
    total_energy = np.sum(magnitudes)
    if total_energy == 0:
        return 0.0
        
    # Calculate the cumulative sum of the magnitudes
    cumulative_energy = np.cumsum(magnitudes)
    
    # Find the index where the cumulative energy exceeds the threshold
    threshold = rolloff_percent * total_energy
    rolloff_index = np.where(cumulative_energy >= threshold)[0][0]
    
    return frequencies[rolloff_index]

def find_fundamental_frequency(frequencies, magnitudes):
    """
    Finds the frequency bin with the maximum magnitude.
    For monophonic signals, this is typically the fundamental frequency (F0).
    """
    if np.sum(magnitudes) == 0:
        return 0.0
        
    peak_index = np.argmax(magnitudes)
    return frequencies[peak_index]

def calculate_thd(frequencies, magnitudes, fundamental_freq=None, num_harmonics=5):
    """
    Calculates the Total Harmonic Distortion (THD) of a signal.
    It measures the ratio of the RMS amplitude of a set of higher harmonics 
    to the RMS amplitude of the fundamental frequency.
    
    Parameters:
        frequencies (np.ndarray): Array of frequency bins.
        magnitudes (np.ndarray): Array of linear magnitudes.
        fundamental_freq (float): The fundamental frequency in Hz. If None, it is auto-detected.
        num_harmonics (int): How many harmonic multiples to include in the distortion calculation.
        
    Returns:
        float: The THD as a percentage.
    """
    if fundamental_freq is None:
        fundamental_freq = find_fundamental_frequency(frequencies, magnitudes)
        
    if fundamental_freq == 0:
        return 0.0

    # We need to find the exact array index (bin) for a given frequency
    df = frequencies[1] - frequencies[0] # Frequency resolution (bin width)
    
    # Extract fundamental magnitude
    f0_index = int(round(fundamental_freq / df))
    if f0_index >= len(magnitudes):
        return 0.0
    f0_mag = magnitudes[f0_index]
    
    if f0_mag == 0:
        return 0.0

    # Calculate the sum of squares for the harmonics
    harmonic_squares_sum = 0.0
    for i in range(2, num_harmonics + 2):
        harmonic_freq = fundamental_freq * i
        h_index = int(round(harmonic_freq / df))
        
        # Ensure we don't look past the Nyquist limit
        if h_index < len(magnitudes):
            harmonic_squares_sum += magnitudes[h_index]**2
            
    # THD formula: sqrt(H2^2 + H3^2 + ... + Hn^2) / H1
    thd = np.sqrt(harmonic_squares_sum) / f0_mag
    
    return thd * 100.0 # Return as percentage