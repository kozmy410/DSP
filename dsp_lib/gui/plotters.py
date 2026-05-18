import numpy as np
import matplotlib.pyplot as plt

def create_layout(mode='standalone', figsize=(10, 6)):
    """
    Creates a Matplotlib Figure and Axes layout.
    
    Parameters:
        mode (str): 'standalone' (1 plot), 'vertical' (2 rows), 'horizontal' (2 columns).
        figsize (tuple): Width and height of the figure.
        
    Returns:
        tuple: (Figure, Axes or array of Axes)
    """
    if mode == 'standalone' or mode == 'overlapped':
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, ax
    elif mode == 'vertical':
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        return fig, axes
    elif mode == 'horizontal':
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        return fig, axes
    else:
        raise ValueError("Mode must be 'standalone', 'overlapped', 'vertical', or 'horizontal'.")

def apply_styling(ax, title="", xlabel="", ylabel="", show_grid=True):
    """Internal helper to consistently style the plots."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.6)
    else:
        ax.grid(False)

def plot_waveform(audio_array, sample_rate, ax=None, color='blue', linewidth=1.0, alpha=1.0, 
                  show_grid=True, label=None):
    """
    Plots a time-domain waveform.
    """
    if ax is None:
        fig, ax = create_layout('standalone')
        
    # Generate time axis dynamically based on array length
    duration = len(audio_array) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_array), endpoint=False)
    
    ax.plot(time_axis, audio_array, color=color, linewidth=linewidth, alpha=alpha, label=label)
    
    apply_styling(ax, title="Time Domain Waveform", xlabel="Time (Seconds)", ylabel="Amplitude", show_grid=show_grid)
    
    if label:
        ax.legend()
        
    return ax

def plot_spectrum(frequencies, magnitudes_db, ax=None, color='red', linewidth=1.0, alpha=1.0, 
                  show_grid=True, label=None, xlim=(20, 20000), ylim=(-100, 0),
                  harmonics_data=None, imd_freqs=None):
    """
    Plots a frequency-domain spectrum with optional overlays for advanced metrics.
    
    Parameters:
        frequencies (np.ndarray): Frequency bins.
        magnitudes_db (np.ndarray): dBFS magnitudes.
        ax (matplotlib.axes.Axes): The axes to plot on.
        harmonics_data (list of dicts): Output from analysis.advanced.find_harmonics.
        imd_freqs (list of floats): Frequencies of intermodulation products to highlight.
    """
    if ax is None:
        fig, ax = create_layout('standalone')
        
    ax.plot(frequencies, magnitudes_db, color=color, linewidth=linewidth, alpha=alpha, label=label)
    
    # Overlay Harmonics if provided
    if harmonics_data:
        for h in harmonics_data:
            ax.axvline(x=h['frequency'], color='orange', linestyle=':', linewidth=1.5, alpha=0.8)
            ax.text(h['frequency'], ylim[1] - 5, f" H{h['harmonic_number']}", color='orange', fontsize=8)

    # Overlay IMD Products if provided
    if imd_freqs:
        for imd_f in imd_freqs:
            ax.axvline(x=imd_f, color='purple', linestyle='-.', linewidth=1.5, alpha=0.8)
            ax.text(imd_f, ylim[1] - 15, " IMD", color='purple', fontsize=8)

    # Use a logarithmic scale for frequencies (standard for audio)
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    apply_styling(ax, title="Frequency Spectrum", xlabel="Frequency (Hz)", ylabel="Magnitude (dBFS)", show_grid=show_grid)
    
    if label or harmonics_data or imd_freqs:
        ax.legend(loc="upper right")
        
    return ax

def plot_spectrogram(frequencies, times, Zxx, ax=None, cmap='viridis', show_grid=False):
    """
    Plots a 2D Heatmap Spectrogram (Time vs Frequency vs Magnitude).
    """
    if ax is None:
        fig, ax = create_layout('standalone')
        
    # Convert STFT complex matrix to dBFS
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
    
    # Draw the heatmap (pcolormesh is highly optimized for GUI rendering)
    mesh = ax.pcolormesh(times, frequencies, magnitude_db, shading='gouraud', cmap=cmap)
    
    ax.set_ylim(20, 20000)
    
    apply_styling(ax, title="Spectrogram", xlabel="Time (Seconds)", ylabel="Frequency (Hz)", show_grid=show_grid)
    
    return ax, mesh

def plot_imd_stems(metrics, ax=None, show_grid=True):
    if ax is None:
        fig, ax = create_layout('standalone')
        
    freqs = [metrics['f1_hz'], metrics['f2_hz']]
    powers = [metrics['f1_db'], metrics['f2_db']]
    colors = ['blue', 'blue']
    labels = ['F1', 'F2']
    
    if metrics['f3_hz'] > 0:
        freqs.append(metrics['f3_hz'])
        powers.append(metrics['f3_db'])
        colors.append('blue')
        labels.append('F3')

    freqs.extend([metrics['imd2_hz'], metrics['imd3_hz']])
    powers.extend([metrics['imd2_db'], metrics['imd3_db']])
    colors.extend(['green', 'purple'])
    labels.extend(['Worst IMD2 (Even)', 'Worst IMD3 (Odd)'])
    
    for f, p, c, l in zip(freqs, powers, colors, labels):
        markerline, stemlines, baseline = ax.stem([f], [p], bottom=-100, label=l)
        plt.setp(stemlines, 'color', c, 'linewidth', 2.5)
        plt.setp(markerline, 'color', c, 'markersize', 8)
        ax.text(f, p + 2, f"{p:.1f} dB", ha='center', color=c, fontweight='bold')
        
    ax.hlines(y=metrics['p_fund_avg'], xmin=min(freqs), xmax=max(freqs), colors='blue', linestyles='dashed', alpha=0.3)
    ax.set_ylim(-100, max(powers) + 15)
    
    span = max(freqs) - min(freqs)
    ax.set_xlim(min(freqs) - (span * 0.2), max(freqs) + (span * 0.2))
    
    apply_styling(ax, title=f"OIP2 (Even): {metrics['oip2_db']:.1f} dB | OIP3 (Odd): {metrics['oip3_db']:.1f} dB", 
                  xlabel="Frequency (Hz)", ylabel="Power (dBFS)", show_grid=show_grid)
    ax.legend()
    return ax

def plot_linearity_curve(input_db, output_db, ax=None, show_grid=True):
    """
    Plots the AM/AM transfer curve to visualize hardware compression/clipping.
    """
    if ax is None:
        fig, ax = create_layout('standalone')
        
    # Plot the ideal perfectly linear theoretical response (Slope = 1)
    ideal_out = input_db - (input_db[0] - output_db[0])
    ax.plot(input_db, ideal_out, color='green', linestyle='--', label="Ideal Linear Hardware")
    
    # Plot the actual measured hardware response
    ax.plot(input_db, output_db, color='red', linewidth=2.5, marker='o', label="Measured Hardware")
    
    # Fill the area between ideal and actual to highlight the compression zone
    ax.fill_between(input_db, ideal_out, output_db, color='red', alpha=0.1, label="Gain Compression")
    
    apply_styling(ax, title="Hardware Linearity / Gain Compression", 
                  xlabel="Input Amplitude (dB)", ylabel="Output Amplitude (dB)", show_grid=show_grid)
    ax.legend()
    return ax