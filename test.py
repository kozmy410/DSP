import numpy as np
import matplotlib.pyplot as plt
from dsp_lib.io.generators import generate_multi_tone
from dsp_lib.core.transforms import compute_fft
from dsp_lib.analysis.advanced import measure_linearity_metrics
from dsp_lib.gui.plotters import create_layout, plot_imd_stems, plot_linearity_curve

print("Testing Hardware Linearity Visualizations...")

# 1. Simulate a dual-tone test through nonlinear hardware
f1, f2 = 1000.0, 1200.0
# Simulating the fundamental outputs and the generated 3rd order IMD products
freqs = [f1, f2, 800.0, 1400.0] 
amps = [0.8, 0.8, 0.05, 0.05] 

audio, sr = generate_multi_tone(frequencies=freqs, amplitudes=amps, duration=1.0)
f_bins, complex_spec = compute_fft(audio, sample_rate=sr, fft_type='rfft', return_complex=True)
linear_mags = np.abs(complex_spec) / (len(audio) / 2.0)

# 2. Extract the exact power metrics using our new function
metrics = measure_linearity_metrics(f_bins, linear_mags, f1, f2)

# 3. Simulate a swept amplitude test for the compression curve
input_sweep_db = np.linspace(-40, 0, 10)
# Simulating hardware that perfectly tracks input until -10dB, then hard-clips (compresses)
output_sweep_db = np.array([x if x < -10 else -10 + (x + 10)*0.2 for x in input_sweep_db])

# 4. Draw the plots side-by-side for the GUI layout
fig, axes = create_layout('horizontal', figsize=(14, 6))

# Left side: The Stem Plot showing the dB delta
plot_imd_stems(metrics, ax=axes[0])

# Right side: The Compression Curve showing where the hardware stops being linear
plot_linearity_curve(input_sweep_db, output_sweep_db, ax=axes[1])

plt.tight_layout()
plt.show()