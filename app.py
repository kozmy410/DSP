import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QSplitter, 
                             QScrollArea, QGroupBox, QCheckBox, QLabel, QFileDialog, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# --- Import Custom DSP Library ---
from dsp_lib.gui.widgets import LabeledComboBox, DynamicParamForm, FileItemWidget
from dsp_lib.io import (generate_hardware_config, read_wav, write_wav, 
                        generate_sine_wave, generate_white_noise, generate_multi_tone, 
                        generate_sweep, generate_square_wave, generate_impulse)
from dsp_lib.core import (compute_fft, compute_stft, apply_lowpass_filter, 
                          apply_highpass_filter, apply_bandpass_filter, apply_notch_filter)
from dsp_lib.analysis import (calculate_peak, calculate_rms, calculate_zero_crossing_rate,
                              calculate_spectral_centroid, calculate_spectral_rolloff,
                              calculate_thd, find_fundamental_frequency, find_harmonics, measure_linearity_metrics)
from dsp_lib.gui.plotters import (plot_waveform, plot_spectrum, plot_spectrogram, 
                                  plot_imd_stems, plot_linearity_curve)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP Hardware Linearity Analyzer")
        self.resize(1300, 850)
        
        self.current_audio = None
        self.current_sr = None
        self.current_filename = None
        
        os.makedirs("outputs", exist_ok=True)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.init_config_tab()
        self.init_generation_tab()
        self.init_extraction_tab()
        self.init_analysis_tab()
        
        self.on_gen_type_changed()
        self.on_filter_changed()
        self.on_plot_type_changed()

    # ==========================================
    # TAB 1: CONFIGURATION (Unchanged)
    # ==========================================
    def init_config_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group_box = QGroupBox("Hardware Configuration")
        form_layout = QVBoxLayout(group_box)
        
        self.api_combo = LabeledComboBox("Host API:", ["Pending Scan..."])
        self.input_combo = LabeledComboBox("Input Device:", ["Pending Scan..."])
        self.output_combo = LabeledComboBox("Output Device:", ["Pending Scan..."])
        self.exclusive_mode = QCheckBox("Enable Exclusive Mode (Bypass OS Mixer)")
        
        scan_btn = QPushButton("Scan Hardware")
        scan_btn.clicked.connect(self.scan_hardware)
        
        form_layout.addWidget(self.api_combo)
        form_layout.addWidget(self.input_combo)
        form_layout.addWidget(self.output_combo)
        form_layout.addWidget(self.exclusive_mode)
        form_layout.addWidget(scan_btn)
        
        layout.addWidget(group_box)
        layout.addStretch()
        self.tabs.addTab(tab, "1. Config")

    def scan_hardware(self):
        try:
            config = generate_hardware_config("hardware_config.json")
            self.api_combo.combo.clear()
            self.input_combo.combo.clear()
            self.output_combo.combo.clear()
            
            for idx, api in config["host_apis"].items():
                self.api_combo.combo.addItem(api["name"])
            for dev in config["devices"]["inputs"]:
                rates = dev.get('supported_rates', [])
                max_sr = max(rates) if rates else "Unknown"
                self.input_combo.combo.addItem(f"{dev['name']} (Max SR: {max_sr})")
            for dev in config["devices"]["outputs"]:
                self.output_combo.combo.addItem(dev["name"])
                
            QMessageBox.information(self, "Success", "Hardware scan complete!")
        except Exception as e:
            QMessageBox.critical(self, "Hardware Error", str(e))

    # ==========================================
    # TAB 2: DATA GENERATION (Expanded)
    # ==========================================
    def init_generation_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        control_panel = QVBoxLayout()
        self.gen_type_combo = LabeledComboBox("Signal Type:", [
            "Sine Wave", "White Noise", "Multi-tone", "Frequency Sweep", "Square Wave", "Impulse"
        ])
        self.gen_type_combo.combo.currentIndexChanged.connect(self.on_gen_type_changed)
        
        self.dynamic_form = DynamicParamForm()
        
        generate_btn = QPushButton("Generate & Send to Extraction")
        generate_btn.clicked.connect(self.generate_data)
        
        control_panel.addWidget(self.gen_type_combo)
        control_panel.addWidget(self.dynamic_form)
        control_panel.addWidget(generate_btn)
        control_panel.addStretch()
        
        layout.addLayout(control_panel, 1)
        layout.addWidget(QWidget(), 2) 
        self.tabs.addTab(tab, "2. Data Generation")

    def on_gen_type_changed(self):
        sig_type = self.gen_type_combo.combo.currentText()
        schema = {"Duration (s)": {"type": "float", "default": "1.0"}}
        
        if sig_type in ["Sine Wave", "Square Wave"]:
            schema["Frequency (Hz)"] = {"type": "float", "default": "1000.0"}
        elif sig_type == "Multi-tone":
            schema["Frequencies (CSV)"] = {"type": "str", "default": "1000, 1500"}
            schema["Amplitudes (CSV or 'auto')"] = {"type": "str", "default": "auto"}
        elif sig_type == "Frequency Sweep":
            schema["Start Freq (Hz)"] = {"type": "float", "default": "20.0"}
            schema["End Freq (Hz)"] = {"type": "float", "default": "20000.0"}
            
        self.dynamic_form.update_form(schema)

    def generate_data(self):
        sig_type = self.gen_type_combo.combo.currentText()
        vals = self.dynamic_form.get_values()
        
        try:
            dur = float(vals.get("Duration (s)", "1.0") or 1.0)
            
            if sig_type == "Sine Wave":
                freq = float(vals.get("Frequency (Hz)", "1000.0") or 1000.0)
                audio, sr = generate_sine_wave(frequency=freq, duration=dur)
                filename = f"outputs/gen_sine_{int(freq)}.wav"
                
            elif sig_type == "Square Wave":
                freq = float(vals.get("Frequency (Hz)", "1000.0") or 1000.0)
                audio, sr = generate_square_wave(frequency=freq, duration=dur)
                filename = f"outputs/gen_square_{int(freq)}.wav"
                
            elif sig_type == "White Noise":
                audio, sr = generate_white_noise(duration=dur)
                filename = "outputs/gen_noise.wav"
                
            elif sig_type == "Impulse":
                audio, sr = generate_impulse(duration=dur)
                filename = "outputs/gen_impulse.wav"
                
            elif sig_type == "Multi-tone":
                # Safely extract frequencies (default to "1000" if left empty)
                freqs_raw = vals.get("Frequencies (CSV)", "1000").strip()
                if freqs_raw == "":
                    freqs_raw = "1000"
                freqs = [float(x.strip()) for x in freqs_raw.split(",")]
                
                # Safely extract amplitudes (default to "auto" if left empty)
                amps_raw = vals.get("Amplitudes (CSV or 'auto')", "auto").strip().lower()
                if amps_raw == "":
                    amps_raw = "auto"
                    
                amps = None if amps_raw == "auto" else [float(x.strip()) for x in amps_raw.split(",")]
                
                audio, sr = generate_multi_tone(frequencies=freqs, amplitudes=amps, duration=dur)
                filename = "outputs/gen_multitone.wav"
                
            elif sig_type == "Frequency Sweep":
                start = float(vals.get("Start Freq (Hz)", "20.0") or 20.0)
                end = float(vals.get("End Freq (Hz)", "20000.0") or 20000.0)
                audio, sr = generate_sweep(start_freq=start, end_freq=end, duration=dur)
                filename = "outputs/gen_sweep.wav"
                
            write_wav(filename, audio, sr)
            self.load_audio_file(filename) 
            self.tabs.setCurrentIndex(2)   
            
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please ensure all inputs are formatted correctly.")
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"An error occurred:\n{e}")

    # ==========================================
    # TAB 3: DATA EXTRACTION (Unchanged)
    # ==========================================
    def init_extraction_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        left_panel = QVBoxLayout()
        load_btn = QPushButton("Load File(s)")
        load_btn.clicked.connect(self.browse_file)
        
        self.file_scroll = QScrollArea()
        self.file_scroll.setWidgetResizable(True)
        self.file_list_widget = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_widget)
        self.file_list_layout.addStretch()
        self.file_scroll.setWidget(self.file_list_widget)
        
        left_panel.addWidget(load_btn)
        left_panel.addWidget(self.file_scroll)
        
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("<b>Auto-Generated Full Duration Waveform</b>"))
        
        self.ext_fig, self.ext_ax = plt.subplots(figsize=(6, 4))
        self.ext_canvas = FigureCanvas(self.ext_fig)
        right_panel.addWidget(self.ext_canvas)
        
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 2)
        self.tabs.addTab(tab, "3. Data Extraction")

    def browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "Audio Files (*.wav)")
        if filepath:
            self.load_audio_file(filepath)

    def load_audio_file(self, filepath):
        try:
            audio, sr = read_wav(filepath, force_mono=True)
            self.current_audio = audio
            self.current_sr = sr
            self.current_filename = os.path.basename(filepath)
            
            file_widget = FileItemWidget(self.current_filename, sr, len(audio))
            self.file_list_layout.insertWidget(0, file_widget) 
            
            self.ext_ax.clear()
            plot_waveform(self.current_audio, self.current_sr, ax=self.ext_ax, color='teal')
            self.ext_fig.tight_layout()
            self.ext_canvas.draw() 
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    # ==========================================
    # TAB 4: ANALYSIS & PLOTTING (Expanded)
    # ==========================================
    def init_analysis_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left Panel: Settings
        settings_panel = QVBoxLayout()
        settings_panel.addWidget(QLabel("<b>Data Pre-Processing</b>"))
        
        self.trim_start = DynamicParamForm()
        self.trim_start.update_form({"Trim Start (ms)": {"type": "float", "default": "0.0"}})
        settings_panel.addWidget(self.trim_start)
        
        self.filter_combo = LabeledComboBox("Filter:", ["None", "Lowpass", "Highpass", "Bandpass", "Notch"])
        self.filter_combo.combo.currentIndexChanged.connect(self.on_filter_changed)
        self.filter_params = DynamicParamForm()
        
        settings_panel.addWidget(self.filter_combo)
        settings_panel.addWidget(self.filter_params)
        
        settings_panel.addWidget(QLabel("<b>Plot & Analysis Customization</b>"))
        self.plot_type = LabeledComboBox("Display:", ["Waveform", "Spectrum", "Spectrogram", "IMD Stem", "Linearity Curve"])
        self.plot_type.combo.currentIndexChanged.connect(self.on_plot_type_changed)
        
        self.window_combo = LabeledComboBox("Window (FFT):", ["hann", "hamming", "blackman", "kaiser", "rectangular"])
        self.advanced_params = DynamicParamForm() # For f1, f2, kaiser beta
        
        self.toggle_harmonics = QCheckBox("Show Harmonic Markers (Spectrum)")
        
        render_btn = QPushButton("Analyze & Render Plot")
        render_btn.clicked.connect(self.render_analysis)
        
        settings_panel.addWidget(self.plot_type)
        settings_panel.addWidget(self.window_combo)
        settings_panel.addWidget(self.advanced_params)
        settings_panel.addWidget(self.toggle_harmonics)
        settings_panel.addWidget(render_btn)
        settings_panel.addStretch()
        
        # Right Panel: Plot and Metrics Display
        right_panel = QVBoxLayout()
        self.ana_fig, self.ana_ax = plt.subplots()
        self.ana_canvas = FigureCanvas(self.ana_fig)
        
        self.metrics_display = QTextEdit()
        self.metrics_display.setReadOnly(True)
        self.metrics_display.setMaximumHeight(150)
        self.metrics_display.setPlaceholderText("Mathematical Metrics will appear here after analysis...")
        
        right_panel.addWidget(self.ana_canvas, stretch=4)
        right_panel.addWidget(self.metrics_display, stretch=1)
        
        layout.addLayout(settings_panel, 1)
        layout.addLayout(right_panel, 3)
        self.tabs.addTab(tab, "4. Analysis & Plotting")

    def on_filter_changed(self):
        f_type = self.filter_combo.combo.currentText()
        schema = {}
        if f_type in ["Lowpass", "Highpass"]:
            schema = {"Cutoff (Hz)": {"type": "float", "default": "1000.0"}}
        elif f_type == "Bandpass":
            schema = {"Low Cut (Hz)": {"type": "float", "default": "500.0"},
                      "High Cut (Hz)": {"type": "float", "default": "2000.0"}}
        elif f_type == "Notch":
            schema = {"Notch Freq (Hz)": {"type": "float", "default": "50.0"}}
        self.filter_params.update_form(schema)

    def on_plot_type_changed(self):
        p_type = self.plot_type.combo.currentText()
        schema = {}
        if p_type in ["IMD Stem", "Linearity Curve"]:
            schema = {"F1 (Hz)": {"type": "float", "default": "1000.0"},
                      "F2 (Hz)": {"type": "float", "default": "1200.0"}}
        self.advanced_params.update_form(schema)

    def render_analysis(self):
        if self.current_audio is None:
            QMessageBox.warning(self, "Warning", "Please generate or load audio first!")
            return
            
        # 1. Trimming
        trim_raw = self.trim_start.get_values().get("Trim Start (ms)", "0.0")
        try:
            trim_ms = float(trim_raw) if trim_raw.strip() != "" else 0.0
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Trim Start must be a valid number.")
            return
            
        trim_samples = int((trim_ms / 1000.0) * self.current_sr)
        working_audio = self.current_audio[trim_samples:]
        
        if len(working_audio) == 0:
            QMessageBox.warning(self, "Warning", "Trim length exceeds audio duration.")
            return

        # 2. Filtering
        f_type = self.filter_combo.combo.currentText()
        f_vals = self.filter_params.get_values()
        try:
            if f_type == "Lowpass":
                cut = float(f_vals.get("Cutoff (Hz)", 1000))
                working_audio = apply_lowpass_filter(working_audio, cut, self.current_sr)
            elif f_type == "Highpass":
                cut = float(f_vals.get("Cutoff (Hz)", 1000))
                working_audio = apply_highpass_filter(working_audio, cut, self.current_sr)
            elif f_type == "Bandpass":
                low = float(f_vals.get("Low Cut (Hz)", 500))
                high = float(f_vals.get("High Cut (Hz)", 2000))
                working_audio = apply_bandpass_filter(working_audio, low, high, self.current_sr)
            elif f_type == "Notch":
                notch = float(f_vals.get("Notch Freq (Hz)", 50))
                working_audio = apply_notch_filter(working_audio, notch, self.current_sr)
        except Exception as e:
            QMessageBox.warning(self, "Filter Error", str(e))
            return

        # 3. Compute Base Metrics
        peak = calculate_peak(working_audio, in_dbfs=True)
        rms = calculate_rms(working_audio, in_dbfs=True)
        zcr = calculate_zero_crossing_rate(working_audio, self.current_sr, return_hz=True)
        
        metrics_text = f"<b>Time Domain:</b> Peak: {peak:.2f} dBFS | RMS: {rms:.2f} dBFS | Zero-Crossings: {zcr:.1f} Hz\n"

        # 4. Plotting & Freq Domain Logic
        self.ana_ax.clear()
        plot_choice = self.plot_type.combo.currentText()
        
        if plot_choice == "Waveform":
            plot_waveform(working_audio, self.current_sr, ax=self.ana_ax, color='blue')
            
        elif plot_choice == "Spectrogram":
            freqs, times, Zxx = compute_stft(working_audio, self.current_sr)
            plot_spectrogram(freqs, times, Zxx, ax=self.ana_ax)
            
        else:
            # Need FFT for Spectrum, IMD, and Linearity
            win_type = self.window_combo.combo.currentText()
            # If Kaiser, pass default beta=14.0 for now, could add to dynamic form later
            kwargs = {'beta': 14.0} if win_type == 'kaiser' else {}
            
            f_bins, mag_db, _ = compute_fft(working_audio, self.current_sr, window_type=win_type, **kwargs)
            linear_mags = 10**(mag_db/20.0) # Approx back to linear for specific THD/IMD math if needed
            
            if plot_choice == "Spectrum":
                harmonics_data = None
                if self.toggle_harmonics.isChecked():
                    f0 = find_fundamental_frequency(f_bins, linear_mags)
                    harmonics_data = find_harmonics(f_bins, linear_mags, f0)
                    
                    centroid = calculate_spectral_centroid(f_bins, linear_mags)
                    thd = calculate_thd(f_bins, linear_mags, f0)
                    metrics_text += f"<br><b>Freq Domain:</b> F0: {f0:.1f} Hz | Centroid: {centroid:.1f} Hz | THD: {thd:.2f}%"

                plot_spectrum(f_bins, mag_db, ax=self.ana_ax, color='red', harmonics_data=harmonics_data)
                
            elif plot_choice in ["IMD Stem", "Linearity Curve"]:
                adv_vals = self.advanced_params.get_values()
                
                # Safely extract F1 (default to 1000.0 if empty)
                f1_raw = str(adv_vals.get("F1 (Hz)", "1000")).strip()
                f1 = float(f1_raw) if f1_raw != "" else 1000.0
                
                # Safely extract F2 (default to 1200.0 if empty)
                f2_raw = str(adv_vals.get("F2 (Hz)", "1200")).strip()
                f2 = float(f2_raw) if f2_raw != "" else 1200.0

        self.metrics_display.setHtml(metrics_text)
        self.ana_fig.tight_layout()
        self.ana_canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())