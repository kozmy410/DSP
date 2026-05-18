"""
Analysis Package for Custom DSP Library
Provides tools for extracting mathematical features from audio signals in both 
the time domain (raw waveform) and the frequency domain (FFT data).
"""

# Import Time Domain tools
from .time_domain import (
    calculate_peak,
    calculate_rms,
    calculate_zero_crossing_rate,
    extract_envelope
)

# Import Frequency Domain tools
from .freq_domain import (
    calculate_spectral_centroid,
    calculate_spectral_rolloff,
    find_fundamental_frequency,
    calculate_thd
)

# Import Advanced Analysis tools
from .advanced import (
    find_harmonics,
    calculate_imd,
    measure_linearity_metrics  # <-- Added here
)

# Define the public API of this package
__all__ = [
    # Time Domain
    "calculate_peak",
    "calculate_rms",
    "calculate_zero_crossing_rate",
    "extract_envelope",
    
    # Frequency Domain
    "calculate_spectral_centroid",
    "calculate_spectral_rolloff",
    "find_fundamental_frequency",
    "calculate_thd",
    
    # Advanced
    "find_harmonics",
    "calculate_imd",
    "measure_linearity_metrics"  # <-- And added here
]