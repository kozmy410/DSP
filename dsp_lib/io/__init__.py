"""
I/O Package for Custom DSP Library
Handles audio generation, reading, writing, hardware probing, and recording.
"""

# Import key functions from generators
from .generators import (
    generate_sine_wave,
    generate_white_noise,
    generate_multi_tone,
    generate_sweep,
    generate_square_wave,
    generate_impulse
)

# Import key functions from readers and writers
from .readers import read_wav, get_file_metadata
from .writers import write_wav

# Import key functions from hardware configuration
from .hardware import (
    get_all_devices,
    generate_hardware_config,
    load_hardware_config
)

# Import key functions and classes from recorders
from .recorders import record_fixed_duration, ContinuousRecorder

# Define the public API of this package using __all__
# This dictates exactly what gets imported if someone uses `from dsp_lib.io import *`
__all__ = [
    "generate_sine_wave",
    "generate_white_noise",
    "generate_multi_tone",
    "generate_sweep",
    "generate_square_wave",
    "generate_impulse",
    "read_wav",
    "get_file_metadata",
    "write_wav",
    "get_all_devices",
    "generate_hardware_config",
    "load_hardware_config",
    "record_fixed_duration",
    "ContinuousRecorder"
]