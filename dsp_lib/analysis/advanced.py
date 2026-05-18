import numpy as np

def find_harmonics(frequencies, magnitudes, fundamental_freq, num_harmonics=10, tolerance_hz=5.0):
    """
    Locates and extracts the magnitudes of the harmonic series for a given fundamental.
    
    Parameters:
        frequencies (np.ndarray): Array of frequency bins.
        magnitudes (np.ndarray): Array of linear magnitudes.
        fundamental_freq (float): The fundamental frequency in Hz.
        num_harmonics (int): Number of harmonics to look for.
        tolerance_hz (float): How far off the exact mathematical frequency the peak can be.
        
    Returns:
        list of dicts: Each dict contains 'harmonic_number', 'frequency', and 'magnitude'.
    """
    df = frequencies[1] - frequencies[0]
    harmonics_data = []
    
    for i in range(1, num_harmonics + 1):
        target_freq = fundamental_freq * i
        
        # Calculate search window indices
        lower_bound = max(0, int((target_freq - tolerance_hz) / df))
        upper_bound = min(len(magnitudes), int((target_freq + tolerance_hz) / df) + 1)
        
        if lower_bound >= len(magnitudes):
            break
            
        # Find the peak within the tolerance window
        window_mags = magnitudes[lower_bound:upper_bound]
        if len(window_mags) > 0:
            local_peak_idx = np.argmax(window_mags)
            actual_idx = lower_bound + local_peak_idx
            
            harmonics_data.append({
                "harmonic_number": i,
                "frequency": frequencies[actual_idx],
                "magnitude": magnitudes[actual_idx]
            })
            
    return harmonics_data

def calculate_imd(frequencies, magnitudes, f1, f2, max_order=3, tolerance_hz=5.0):
    """
    Calculates Intermodulation Distortion (IMD) for a dual-tone signal.
    It looks for 2nd and 3rd order sum and difference products.
    
    Parameters:
        frequencies (np.ndarray): Array of frequency bins.
        magnitudes (np.ndarray): Array of linear magnitudes.
        f1 (float): The first fundamental frequency in Hz.
        f2 (float): The second fundamental frequency in Hz.
        max_order (int): The maximum IMD order to calculate (2 or 3).
        tolerance_hz (float): Tolerance for finding the frequency bins.
        
    Returns:
        float: The IMD as a percentage.
    """
    df = frequencies[1] - frequencies[0]
    
    def get_mag_at_freq(target_freq):
        if target_freq <= 0:
            return 0.0
        lower = max(0, int((target_freq - tolerance_hz) / df))
        upper = min(len(magnitudes), int((target_freq + tolerance_hz) / df) + 1)
        if lower >= len(magnitudes) or len(magnitudes[lower:upper]) == 0:
            return 0.0
        return np.max(magnitudes[lower:upper])

    # 1. Extract the magnitude of the two fundamentals
    mag_f1 = get_mag_at_freq(f1)
    mag_f2 = get_mag_at_freq(f2)
    
    fundamentals_rms = np.sqrt(mag_f1**2 + mag_f2**2)
    if fundamentals_rms == 0:
        return 0.0

    imd_products = []

    # 2. Calculate 2nd Order Products (f1 + f2, |f1 - f2|)
    if max_order >= 2:
        imd_products.extend([
            f1 + f2,
            abs(f1 - f2)
        ])
        
    # 3. Calculate 3rd Order Products (2*f1 +/- f2, 2*f2 +/- f1)
    if max_order >= 3:
        imd_products.extend([
            abs(2*f1 - f2), 2*f1 + f2,
            abs(2*f2 - f1), 2*f2 + f1
        ])

    # 4. Extract magnitudes for all IMD products and calculate their RMS sum
    imd_squares_sum = 0.0
    for prod_freq in set(imd_products): # Use set to remove duplicates if any
        prod_mag = get_mag_at_freq(prod_freq)
        imd_squares_sum += prod_mag**2

    # 5. IMD formula: sqrt(sum of squared IMD products) / sqrt(sum of squared fundamentals)
    imd_ratio = np.sqrt(imd_squares_sum) / fundamentals_rms
    
    return imd_ratio * 100.0

def measure_linearity_metrics(frequencies, magnitudes, f1, f2, f3=None, tolerance_hz=5.0):
    """
    Measures absolute power of fundamentals vs Even (IMD2) and Odd (IMD3) products.
    Calculates OIP2 and OIP3. Supports 2 or 3 tones.
    """
    df = frequencies[1] - frequencies[0]
    
    def get_power_db(target_freq):
        if target_freq <= 0: return -100.0
        lower = max(0, int((target_freq - tolerance_hz) / df))
        upper = min(len(magnitudes), int((target_freq + tolerance_hz) / df) + 1)
        if lower >= len(magnitudes) or len(magnitudes[lower:upper]) == 0:
            return -100.0 
        peak_mag = np.max(magnitudes[lower:upper])
        return 20 * np.log10(max(peak_mag, 1e-10))

    # 1. Fundamental Power (Average)
    p_f1 = get_power_db(f1)
    p_f2 = get_power_db(f2)
    funds_p = [p_f1, p_f2]
    if f3:
        p_f3 = get_power_db(f3)
        funds_p.append(p_f3)
    p_fund_avg = sum(funds_p) / len(funds_p)

    # 2. 2nd Order Products (Even)
    imd2_freqs = [abs(f1 - f2), f1 + f2]
    if f3:
        imd2_freqs.extend([abs(f2 - f3), abs(f1 - f3), f1 + f3, f2 + f3])
        
    p_imd2_list = [get_power_db(f) for f in imd2_freqs]
    p_imd2_max = max(p_imd2_list)
    best_imd2_freq = imd2_freqs[p_imd2_list.index(p_imd2_max)]

    # 3. 3rd Order Products (Odd)
    imd3_freqs = [abs(2*f1 - f2), abs(2*f2 - f1)]
    if f3:
        imd3_freqs.extend([abs(f1 + f2 - f3), abs(f1 - f2 + f3), abs(f2 + f3 - f1)])
        
    p_imd3_list = [get_power_db(f) for f in imd3_freqs]
    p_imd3_max = max(p_imd3_list)
    best_imd3_freq = imd3_freqs[p_imd3_list.index(p_imd3_max)]

    # 4. Calculate Output Intercept Points
    oip2 = p_fund_avg + (p_fund_avg - p_imd2_max)
    oip3 = p_fund_avg + (p_fund_avg - p_imd3_max) / 2.0

    return {
        "f1_hz": f1, "f1_db": p_f1,
        "f2_hz": f2, "f2_db": p_f2,
        "f3_hz": f3 if f3 else 0.0, "f3_db": p_f3 if f3 else -100.0,
        "imd2_hz": best_imd2_freq, "imd2_db": p_imd2_max,
        "imd3_hz": best_imd3_freq, "imd3_db": p_imd3_max,
        "oip2_db": oip2,
        "oip3_db": oip3,
        "p_fund_avg": p_fund_avg
    }