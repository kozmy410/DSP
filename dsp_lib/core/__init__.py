"""
Core Package for Custom DSP Library
Handles fundamental mathematical processing including offline and real-time filtering,
and frequency domain transforms (FFT, iFFT, STFT).
"""

# Import key functions and classes from the filter module
from .filters import (
    apply_lowpass_filter,
    apply_highpass_filter,
    apply_bandpass_filter,
    apply_notch_filter,
    RealTimeFilter
)

# Import key functions from the transforms module
from .transforms import (
    apply_window,
    compute_fft,
    compute_ifft,
    compute_stft
)

# Define the public API of this package
__all__ = [
    # Filter tools
    "apply_lowpass_filter",
    "apply_highpass_filter",
    "apply_bandpass_filter",
    "apply_notch_filter",
    "RealTimeFilter",
    
    # Transform tools
    "apply_window",
    "compute_fft",
    "compute_ifft",
    "compute_stft"
]