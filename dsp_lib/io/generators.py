import numpy as np

def generate_sine_wave(frequency=440.0, sample_rate=44100, duration=1.0):
    """
    Synthesizes a pure sine wave.
    
    Parameters:
        frequency (float): The frequency of the tone in Hz (default: 440Hz, A4).
        sample_rate (int): The sample rate in Hz (default: 44100 CD Quality).
        duration (float): The length of the tone in seconds.
        
    Returns:
        tuple: (audio_array, sample_rate)
    """
    # Enforce a strict 4.0 second maximum duration for the buffer
    duration = min(float(duration), 4.0)
    
    # Create an array of time values
    # endpoint=False ensures we don't overlap the final sample if looping
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate the sine wave: y(t) = A * sin(2 * pi * f * t)
    # We keep amplitude (A) at 1.0 (0 dBFS) for a normalized float array
    audio_array = np.sin(2 * np.pi * frequency * t)
    
    return audio_array, sample_rate

def generate_white_noise(sample_rate=44100, duration=1.0):
    """
    Synthesizes uniform white noise, useful for testing filter responses.
    """
    duration = min(float(duration), 4.0)
    num_samples = int(sample_rate * duration)
    
    # Generate random values between -1.0 and 1.0
    audio_array = np.random.uniform(-1.0, 1.0, num_samples)
    
    return audio_array, sample_rate

def generate_multi_tone(frequencies, amplitudes=None, sample_rate=44100, duration=1.0):
    """
    Synthesizes and combines multiple pure tones into a single complex wave.
    
    Parameters:
        frequencies (list): A list of frequencies in Hz (e.g., [440.0, 880.0]).
        amplitudes (list): A list of linear amplitudes (0.0 to 1.0) for each frequency. 
                           If None, all frequencies are weighted equally.
        sample_rate (int): The sample rate in Hz.
        duration (float): The length of the tone in seconds.
        
    Returns:
        tuple: (audio_array, sample_rate)
    """
    duration = min(float(duration), 4.0)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    if amplitudes is None:
        amplitudes = [1.0 / len(frequencies)] * len(frequencies)
        
    # Initialize an empty array of zeros
    audio_array = np.zeros_like(t)
    
    # Generate and accumulate each frequency
    for freq, amp in zip(frequencies, amplitudes):
        audio_array += amp * np.sin(2 * np.pi * freq * t)
        
    # Normalize the final array to prevent digital clipping (keeping it strictly between -1.0 and 1.0)
    max_val = np.max(np.abs(audio_array))
    if max_val > 1.0:
        audio_array /= max_val
        
    return audio_array, sample_rate

def generate_sweep(start_freq=20.0, end_freq=20000.0, sample_rate=44100, duration=2.0):
    """
    Synthesizes a linear frequency sweep (chirp). Essential for testing filter responses.
    """
    duration = min(float(duration), 4.0)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Calculate the instantaneous frequency at each point in time
    instantaneous_frequency = start_freq + (end_freq - start_freq) * (t / (2 * duration))
    
    audio_array = np.sin(2 * np.pi * instantaneous_frequency * t)
    
    return audio_array, sample_rate

def generate_square_wave(frequency=440.0, sample_rate=44100, duration=1.0):
    """
    Synthesizes a square wave. Rich in odd harmonics, good for testing distortion.
    """
    duration = min(float(duration), 4.0)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # A square wave is simply the mathematical sign (+1 or -1) of a sine wave
    audio_array = np.sign(np.sin(2 * np.pi * frequency * t))
    
    return audio_array, sample_rate

def generate_impulse(sample_rate=44100, duration=0.1):
    """
    Synthesizes a single-sample impulse (Dirac delta). 
    Used to capture the impulse response (IR) of filters or reverbs.
    """
    duration = min(float(duration), 4.0)
    num_samples = int(sample_rate * duration)
    
    audio_array = np.zeros(num_samples)
    audio_array[0] = 1.0  # The single click at the very first sample
    
    return audio_array, sample_rate