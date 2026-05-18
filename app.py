import sys
import os
import csv
import numpy as np
import scipy.signal as sp
import sounddevice as sd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                             QVBoxLayout, QHBoxLayout, QPushButton, QSplitter, 
                             QGroupBox, QCheckBox, QLabel, QFileDialog, QMessageBox, 
                             QTextEdit, QTreeWidget, QTreeWidgetItem, QLineEdit,
                             QColorDialog, QDoubleSpinBox, QComboBox, QDialog, QMenu, QAction)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# --- Import Custom DSP Library ---
from dsp_lib.gui.widgets import LabeledComboBox, DynamicParamForm
from dsp_lib.io import (generate_hardware_config, read_wav, write_wav, 
                        generate_sine_wave, generate_white_noise, generate_multi_tone, 
                        generate_sweep, generate_square_wave, generate_impulse,
                        record_fixed_duration) 
from dsp_lib.core import (compute_fft, compute_stft, apply_lowpass_filter, 
                          apply_highpass_filter, apply_bandpass_filter, apply_notch_filter)
from dsp_lib.analysis import (calculate_peak, calculate_rms, calculate_zero_crossing_rate,
                              calculate_spectral_centroid, calculate_spectral_rolloff,
                              calculate_thd, find_fundamental_frequency, find_harmonics, measure_linearity_metrics)
from dsp_lib.gui.plotters import (plot_waveform, plot_spectrum, plot_spectrogram, 
                                  plot_imd_stems, plot_linearity_curve)

# ==========================================
# WINDOW & DIALOG CLASSES
# ==========================================
class PopoutPlotWindow(QMainWindow):
    """A standalone window to display a specific plot."""
    def __init__(self, title, plot_type, plot_kwargs):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(800, 600)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        plot_kwargs['ax'] = self.ax
        if plot_type == "Waveform":
            plot_waveform(**plot_kwargs)
        elif plot_type == "Spectrum":
            if "Linear" in title:
                self.ax.plot(plot_kwargs.get('frequencies'), plot_kwargs.get('magnitudes_db'), label=plot_kwargs.get('label'))
                self.ax.set_xscale('log')
                self.ax.set_title("Frequency Spectrum (Linear Magnitude)")
                self.ax.grid(True, linestyle='--', alpha=0.6)
                self.ax.legend()
            else:
                plot_spectrum(**plot_kwargs)
        elif plot_type == "Spectrogram":
            plot_spectrogram(**plot_kwargs)
        elif plot_type == "IMD Stem":
            plot_imd_stems(**plot_kwargs)
            
        self.fig.tight_layout()
        self.canvas.draw()

class PlotCustomizerDialog(QDialog):
    """A dialog to change line colors, styles, and widths for the active canvas."""
    def __init__(self, fig, canvas, parent=None):
        super().__init__(parent)
        self.fig = fig
        self.canvas = canvas
        self.setWindowTitle("Customize Plot Lines")
        self.layout = QVBoxLayout(self)
        
        self.lines = []
        for ax in self.fig.axes:
            self.lines.extend(ax.get_lines())
            
        if not self.lines:
            self.layout.addWidget(QLabel("No customizable lines found on the current plot."))
            return
            
        for i, line in enumerate(self.lines):
            row = QHBoxLayout()
            label = QLabel(line.get_label() or f"Line {i+1}")
            label.setMinimumWidth(150)
            
            color_btn = QPushButton("Color")
            color_btn.setStyleSheet(f"background-color: {line.get_color()}")
            color_btn.clicked.connect(lambda checked, l=line, btn=color_btn: self.choose_color(l, btn))
            
            style_combo = QComboBox()
            style_combo.addItems(['-', '--', '-.', ':', 'None'])
            style_combo.setCurrentText(line.get_linestyle())
            style_combo.currentTextChanged.connect(lambda text, l=line: l.set_linestyle(text))
            
            width_spin = QDoubleSpinBox()
            width_spin.setRange(0.5, 10.0)
            width_spin.setSingleStep(0.5)
            width_spin.setValue(line.get_linewidth())
            width_spin.valueChanged.connect(lambda val, l=line: l.set_linewidth(val))
            
            row.addWidget(label)
            row.addWidget(color_btn)
            row.addWidget(style_combo)
            row.addWidget(width_spin)
            self.layout.addLayout(row)
            
        apply_btn = QPushButton("Apply Changes & Re-Render")
        apply_btn.clicked.connect(self.apply_changes)
        self.layout.addSpacing(10)
        self.layout.addWidget(apply_btn)
        
    def choose_color(self, line, btn):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_color = color.name()
            line.set_color(hex_color)
            btn.setStyleSheet(f"background-color: {hex_color}")
            
    def apply_changes(self):
        self.canvas.draw()
        self.accept()

# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSP Hardware Linearity Analyzer - Scalable GUI")
        self.resize(1400, 900)
        
        self.current_audio = None
        self.current_sr = None
        self.current_filename = None
        self.report_export_data = [] 
        
        os.makedirs("outputs", exist_ok=True)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.main_splitter)

        # ==========================================
        # LEFT PANEL: Persistent File & Hierarchy Tree
        # ==========================================
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        load_btn = QPushButton("Load File(s) from Disk")
        load_btn.clicked.connect(self.browse_file)
        self.left_layout.addWidget(load_btn)
        
        # Audio Playback Block
        playback_layout = QHBoxLayout()
        play_btn = QPushButton("▶ Play File")
        play_btn.clicked.connect(self.play_audio)
        stop_btn = QPushButton("⏹ Stop")
        stop_btn.clicked.connect(self.stop_audio)
        playback_layout.addWidget(play_btn)
        playback_layout.addWidget(stop_btn)
        self.left_layout.addLayout(playback_layout)
        
        # Live Recording Block
        record_layout = QHBoxLayout()
        self.rec_dur_input = QLineEdit()
        self.rec_dur_input.setPlaceholderText("Rec Dur (s)")
        self.rec_dur_input.setText("4.0")
        rec_btn = QPushButton("🔴 Record Laptop Mic")
        rec_btn.clicked.connect(self.record_audio)
        record_layout.addWidget(self.rec_dur_input)
        record_layout.addWidget(rec_btn)
        self.left_layout.addLayout(record_layout)
        
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel("Loaded Files & Generated Plots")
        
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.open_tree_context_menu)
        
        self.file_tree.itemClicked.connect(self.on_tree_item_clicked)
        self.file_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        self.left_layout.addWidget(self.file_tree)
        
        self.popout_windows = []

        # ==========================================
        # RIGHT PANEL: The Tabs
        # ==========================================
        self.tabs = QTabWidget()
        
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setSizes([300, 1100])

        self.init_config_tab()
        self.init_generation_tab()
        self.init_extraction_tab()
        self.init_analysis_tab()
        self.init_report_tab()  
        
        self.on_gen_type_changed()
        self.on_filter_changed()
        self.on_plot_type_changed()

    def play_audio(self):
        if self.current_audio is not None and self.current_sr is not None:
            sd.stop() 
            sd.play(self.current_audio, self.current_sr)
        else:
            QMessageBox.warning(self, "Playback", "Please select a file from the list first.")

    def stop_audio(self):
        sd.stop()

    def open_tree_context_menu(self, position):
        item = self.file_tree.itemAt(position)
        if not item: return
        
        menu = QMenu()
        delete_action = QAction("❌ Unload / Delete", self)
        delete_action.triggered.connect(lambda: self.delete_tree_item(item))
        menu.addAction(delete_action)
        menu.exec_(self.file_tree.viewport().mapToGlobal(position))

    def delete_tree_item(self, item):
        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            index = self.file_tree.indexOfTopLevelItem(item)
            self.file_tree.takeTopLevelItem(index)
            
            data = item.data(0, Qt.UserRole)
            if data and data.get("filename") == self.current_filename:
                self.current_audio = None
                self.current_filename = None
                self.current_sr = None
                self.stop_audio()

    def record_audio(self):
        try:
            # Removed the 4.0 second maximum cap
            dur = float(self.rec_dur_input.text() or 4.0)
            if dur <= 0:
                raise ValueError("Duration must be a positive number.")
                
            QMessageBox.information(self, "Recording", f"A {dur}s recording will start when you click OK. Play the audio from your phone now.")
            
            audio, sr = record_fixed_duration(dur, sample_rate=48000, channels=1, device_index=None)
            
            # --- New: Save recording directly to disk ---
            filename = f"Live_Record_{np.random.randint(1000, 9999)}.wav"
            filepath = os.path.join("outputs", filename)
            write_wav(filepath, audio, sr)
            
            self.current_audio = audio
            self.current_sr = sr
            self.current_filename = filename
            
            file_item = QTreeWidgetItem(self.file_tree)
            file_item.setText(0, f"{self.current_filename}")
            file_item.setData(0, Qt.UserRole, {"type": "file", "audio": audio, "sr": sr, "filename": self.current_filename})
            
            base_graph_item = QTreeWidgetItem(file_item)
            base_graph_item.setText(0, "Raw Waveform (Unmodified)")
            base_graph_item.setData(0, Qt.UserRole, {
                "type": "graph",
                "plot_type": "Waveform",
                "plot_kwargs": {'audio_array': audio, 'sample_rate': sr, 'color': 'teal', 'label': 'Full Raw Waveform'}
            })
            
            self.file_tree.expandItem(file_item)
            
            self.tabs.setCurrentIndex(2) 
            
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Ensure your laptop microphone is enabled.\n\n{str(e)}")

    def on_tree_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data and data.get("type") == "file":
            self.current_audio = data["audio"]
            self.current_sr = data["sr"]
            self.current_filename = data["filename"]

    def on_tree_item_double_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data and data.get("type") == "graph":
            parent = item.parent()
            fname = parent.text(0) if parent else self.current_filename
            title = f"{fname} - {item.text(0)}"
            popout = PopoutPlotWindow(title, data["plot_type"], data["plot_kwargs"])
            popout.show()
            self.popout_windows.append(popout)

    def add_graph_to_tree(self, name, plot_type, plot_kwargs):
        root = self.file_tree.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            data = item.data(0, Qt.UserRole)
            if data and data.get("filename") == self.current_filename:
                graph_item = QTreeWidgetItem(item)
                graph_item.setText(0, name)
                graph_item.setData(0, Qt.UserRole, {
                    "type": "graph",
                    "plot_type": plot_type,
                    "plot_kwargs": plot_kwargs
                })
                item.setExpanded(True)
                break

    # ==========================================
    # TAB 1: CONFIGURATION
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
    # TAB 2: DATA GENERATION
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
        
        generate_btn = QPushButton("Generate & Send to Workspace")
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
        schema = {
            "Duration (s)": {"type": "float", "default": "4.0"}, # Max limit label removed
            "Sample Rate (Hz)": {"type": "float", "default": "48000.0"}
        }
        
        if sig_type in ["Sine Wave", "Square Wave"]:
            schema["Frequency (Hz)"] = {"type": "float", "default": "1000.0"}
        elif sig_type == "Multi-tone":
            schema["Frequencies (CSV)"] = {"type": "str", "default": "1000, 1100, 2000"}
            schema["Amplitudes (CSV or 'auto')"] = {"type": "str", "default": "auto"}
        elif sig_type == "Frequency Sweep":
            schema["Start Freq (Hz)"] = {"type": "float", "default": "20.0"}
            schema["End Freq (Hz)"] = {"type": "float", "default": "20000.0"}
            
        self.dynamic_form.update_form(schema)

    def generate_data(self):
        sig_type = self.gen_type_combo.combo.currentText()
        vals = self.dynamic_form.get_values()
        
        try:
            # Min cap removed
            dur = float(vals.get("Duration (s)", "4.0") or 4.0)
            sr_input = int(float(vals.get("Sample Rate (Hz)", "48000.0") or 48000.0))
            
            if sig_type == "Sine Wave":
                freq = float(vals.get("Frequency (Hz)", "1000.0") or 1000.0)
                audio, sr = generate_sine_wave(frequency=freq, sample_rate=sr_input, duration=dur)
                filename = f"outputs/gen_sine_{int(freq)}.wav"
                
            elif sig_type == "Square Wave":
                freq = float(vals.get("Frequency (Hz)", "1000.0") or 1000.0)
                audio, sr = generate_square_wave(frequency=freq, sample_rate=sr_input, duration=dur)
                filename = f"outputs/gen_square_{int(freq)}.wav"
                
            elif sig_type == "White Noise":
                audio, sr = generate_white_noise(sample_rate=sr_input, duration=dur)
                filename = "outputs/gen_noise.wav"
                
            elif sig_type == "Impulse":
                audio, sr = generate_impulse(sample_rate=sr_input, duration=dur)
                filename = "outputs/gen_impulse.wav"
                
            elif sig_type == "Multi-tone":
                freqs_raw = vals.get("Frequencies (CSV)", "1000").strip()
                freqs_raw = "1000" if freqs_raw == "" else freqs_raw
                freqs = [float(x.strip()) for x in freqs_raw.split(",")]
                
                amps_raw = vals.get("Amplitudes (CSV or 'auto')", "auto").strip().lower()
                amps_raw = "auto" if amps_raw == "" else amps_raw
                amps = None if amps_raw == "auto" else [float(x.strip()) for x in amps_raw.split(",")]
                
                audio, sr = generate_multi_tone(frequencies=freqs, amplitudes=amps, sample_rate=sr_input, duration=dur)
                filename = "outputs/gen_multitone.wav"
                
            elif sig_type == "Frequency Sweep":
                start = float(vals.get("Start Freq (Hz)", "20.0") or 20.0)
                end = float(vals.get("End Freq (Hz)", "20000.0") or 20000.0)
                audio, sr = generate_sweep(start_freq=start, end_freq=end, sample_rate=sr_input, duration=dur)
                filename = "outputs/gen_sweep.wav"
                
            write_wav(filename, audio, sr)
            self.load_audio_file(filename) 
            
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please ensure all inputs are formatted correctly.")
        except Exception as e:
            QMessageBox.critical(self, "Generation Error", f"An error occurred:\n{e}")

    # ==========================================
    # TAB 3: DATA EXTRACTION
    # ==========================================
    def init_extraction_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        layout.addWidget(QLabel("<b>Auto-Generated Full Duration Waveform (Raw View)</b>"))
        
        self.ext_fig, self.ext_ax = plt.subplots(figsize=(6, 4))
        self.ext_canvas = FigureCanvas(self.ext_fig)
        self.ext_toolbar = NavigationToolbar(self.ext_canvas, self) 
        
        layout.addWidget(self.ext_toolbar)
        layout.addWidget(self.ext_canvas)
        
        self.tabs.addTab(tab, "3. Data Extraction")

    def browse_file(self):
        filepaths, _ = QFileDialog.getOpenFileNames(self, "Open WAV File(s)", "", "Audio Files (*.wav)")
        for filepath in filepaths:
            self.load_audio_file(filepath)

    def load_audio_file(self, filepath):
        try:
            audio, sr = read_wav(filepath, force_mono=True)
            self.current_audio = audio
            self.current_sr = sr
            self.current_filename = os.path.basename(filepath)
            
            duration = len(audio) / sr
            file_item = QTreeWidgetItem(self.file_tree)
            file_item.setText(0, f"{self.current_filename}")
            file_item.setToolTip(0, f"SR: {sr} Hz | Dur: {duration:.2f}s | Size: {len(audio)} frames")
            
            file_item.setData(0, Qt.UserRole, {
                "type": "file", 
                "audio": audio, 
                "sr": sr, 
                "filename": self.current_filename
            })
            
            base_graph_item = QTreeWidgetItem(file_item)
            base_graph_item.setText(0, "Raw Waveform (Unmodified)")
            
            base_graph_item.setData(0, Qt.UserRole, {
                "type": "graph",
                "plot_type": "Waveform",
                "plot_kwargs": {'audio_array': audio, 'sample_rate': sr, 'color': 'teal', 'label': 'Full Raw Waveform'}
            })
            self.file_tree.expandItem(file_item)
            
            self.ext_ax.clear()
            plot_waveform(self.current_audio, self.current_sr, ax=self.ext_ax, color='teal')
            self.ext_fig.tight_layout()
            self.ext_canvas.draw() 
            
            self.tabs.setCurrentIndex(2)   
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    # ==========================================
    # TAB 4: DATA ANALYSIS & PLOTTING
    # ==========================================
    def init_analysis_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        settings_panel = QVBoxLayout()
        
        settings_panel.addWidget(QLabel("<b>Data Clip Range (s)</b>"))
        self.clip_range = DynamicParamForm()
        self.clip_range.update_form({"Start Time (s)": {"type": "float", "default": "0.5"}, 
                                     "End Time (s)": {"type": "float", "default": "1.5"}})
        settings_panel.addWidget(self.clip_range)
        
        settings_panel.addWidget(QLabel("<b>Filter Processing</b>"))
        self.filter_combo = LabeledComboBox("Filter Type:", ["None", "Lowpass", "Highpass", "Bandpass", "Notch"])
        self.filter_combo.combo.currentIndexChanged.connect(self.on_filter_changed)
        self.filter_params = DynamicParamForm()
        settings_panel.addWidget(self.filter_combo)
        settings_panel.addWidget(self.filter_params)
        
        settings_panel.addWidget(QLabel("<b>Plot Configuration</b>"))
        self.plot_type = LabeledComboBox("Display Element:", ["Waveform", "Spectrum", "Spectrogram", "IMD Stem", "Linearity Curve", "Filter Characteristics"])
        self.plot_type.combo.currentIndexChanged.connect(self.on_plot_type_changed)
        
        self.window_combo = LabeledComboBox("Window (FFT):", ["blackman", "hann", "hamming", "kaiser", "rectangular"])
        self.y_axis_mode = LabeledComboBox("Y-Axis Scale:", ["dBFS", "Linear Magnitude"])
        self.plot_mode = LabeledComboBox("Render Mode:", ["Replace", "Overlap", "Side-by-Side"])
        
        customize_btn = QPushButton("Customize Plot Lines")
        customize_btn.clicked.connect(self.open_customizer)
        
        self.advanced_params = DynamicParamForm() 
        self.toggle_harmonics = QCheckBox("Show Harmonic Markers (Spectrum)")
        
        render_btn = QPushButton("Analyze & Render Plot")
        render_btn.clicked.connect(self.render_analysis)
        
        settings_panel.addWidget(self.plot_type)
        settings_panel.addWidget(self.plot_mode)
        settings_panel.addWidget(customize_btn)
        settings_panel.addWidget(self.y_axis_mode)
        settings_panel.addWidget(self.window_combo)
        settings_panel.addWidget(self.advanced_params)
        settings_panel.addWidget(self.toggle_harmonics)
        settings_panel.addWidget(render_btn)
        settings_panel.addStretch()
        
        right_panel = QVBoxLayout()
        self.ana_fig = plt.figure(figsize=(8, 6))
        self.ana_canvas = FigureCanvas(self.ana_fig)
        self.ana_toolbar = NavigationToolbar(self.ana_canvas, self) 
        
        self.metrics_display = QTextEdit()
        self.metrics_display.setReadOnly(True)
        self.metrics_display.setMaximumHeight(100)
        self.metrics_display.setPlaceholderText("Mathematical Metrics will appear here after analysis...")
        
        right_panel.addWidget(self.ana_toolbar)
        right_panel.addWidget(self.ana_canvas, stretch=5)
        right_panel.addWidget(self.metrics_display, stretch=1)
        
        layout.addLayout(settings_panel, 1)
        layout.addLayout(right_panel, 3)
        self.tabs.addTab(tab, "4. Analysis & Plotting")

    def open_customizer(self):
        dialog = PlotCustomizerDialog(self.ana_fig, self.ana_canvas, self)
        dialog.exec_()

    def on_filter_changed(self):
        f_type = self.filter_combo.combo.currentText()
        schema = {}
        if f_type in ["Lowpass", "Highpass"]:
            schema = {"Cutoff (Hz)": {"type": "float", "default": "3000.0"}}
        elif f_type == "Bandpass":
            schema = {"Low Cut (Hz)": {"type": "float", "default": "500.0"},
                      "High Cut (Hz)": {"type": "float", "default": "2000.0"}}
        elif f_type == "Notch":
            schema = {"Notch Freq (Hz)": {"type": "float", "default": "50.0"}}
        self.filter_params.update_form(schema)

    def on_plot_type_changed(self):
        p_type = self.plot_type.combo.currentText()
        schema = {}
        if p_type in ["IMD Stem", "Linearity Curve", "Filter Characteristics"]:
            schema = {"F1 (Hz)": {"type": "float", "default": "1000.0"},
                      "F2 (Hz)": {"type": "float", "default": "1100.0"},
                      "F3 (Hz) (Optional)": {"type": "float", "default": "2000.0"}}
        self.advanced_params.update_form(schema)

    def render_analysis(self):
        if self.current_audio is None:
            QMessageBox.warning(self, "Warning", "Please generate or load an audio file from the side panel first!")
            return
            
        f_type = self.filter_combo.combo.currentText()
        f_vals = self.filter_params.get_values()
        plot_choice = self.plot_type.combo.currentText()
        
        if plot_choice == "Filter Characteristics":
            self.ana_fig.clear()
            ax1 = self.ana_fig.add_subplot(211)
            ax2 = self.ana_fig.add_subplot(212)
            
            fc = float(f_vals.get("Cutoff (Hz)", 3000)) if f_type in ["Lowpass", "Highpass"] else 3000
            numtaps = 101
            b = sp.firwin(numtaps, fc, window='blackman', fs=self.current_sr)
            a = np.array([1.0])
            w, H = sp.freqz(b, a, worN=2000, fs=self.current_sr)
            
            ax1.plot(b, 'b.-', label='b coefficients')
            ax1.plot(0, a[0], 'ro', label='a coefficients')
            ax1.set_title("Filter Coefficients")
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend()
            
            ax2.plot(w, np.abs(H), 'b-')
            ax2.set_title("Frequency Response")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_xlim(0, self.current_sr/2)
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            self.metrics_display.setHtml("<b>Filter Setup:</b> Blackman Window FIR, 101 Taps, Fc=" + str(fc) + "Hz")
            self.ana_fig.tight_layout()
            self.ana_canvas.draw()
            return
            
        clip_vals = self.clip_range.get_values()
        try:
            start_s = float(clip_vals.get("Start Time (s)", "0.5") or 0.5)
            end_s = float(clip_vals.get("End Time (s)", "1.5") or 1.5)
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Clip ranges must be valid numbers.")
            return
            
        start_idx = int(start_s * self.current_sr)
        end_idx = int(end_s * self.current_sr)
        working_audio = self.current_audio[start_idx:end_idx]
        
        if len(working_audio) == 0:
            QMessageBox.warning(self, "Warning", "Clip length exceeds audio duration or is empty.")
            return

        try:
            if f_type == "Lowpass":
                cut = float(f_vals.get("Cutoff (Hz)", 3000))
                working_audio = apply_lowpass_filter(working_audio, cut, self.current_sr)
            elif f_type == "Highpass":
                cut = float(f_vals.get("Cutoff (Hz)", 3000))
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

        plot_mode = self.plot_mode.combo.currentText()
        y_axis = self.y_axis_mode.combo.currentText()
        
        if plot_mode == "Replace":
            self.ana_fig.clear()
            ax = self.ana_fig.add_subplot(111)
            
        elif plot_mode == "Side-by-Side":
            num_axes = len(self.ana_fig.axes)
            if num_axes == 0:
                ax = self.ana_fig.add_subplot(111)
            else:
                gs = self.ana_fig.add_gridspec(1, num_axes + 1)
                for i, existing_ax in enumerate(self.ana_fig.axes):
                    existing_ax.set_subplotspec(gs[0, i])
                ax = self.ana_fig.add_subplot(gs[0, num_axes])
                
        elif plot_mode == "Overlap":
            if not self.ana_fig.axes:
                ax = self.ana_fig.add_subplot(111)
            else:
                ax = self.ana_fig.axes[-1]  

        metrics_text = ""
        plot_kwargs = {} 
        
        if plot_choice == "Waveform":
            label = f"Clip {start_s}-{end_s}s ({f_type})"
            plot_kwargs = {'audio_array': working_audio, 'sample_rate': self.current_sr, 'color': np.random.rand(3,), 'label': label}
            plot_waveform(**plot_kwargs, ax=ax)
            
        elif plot_choice == "Spectrogram":
            freqs, times, Zxx = compute_stft(working_audio, self.current_sr)
            plot_kwargs = {'frequencies': freqs, 'times': times, 'Zxx': Zxx}
            plot_spectrogram(**plot_kwargs, ax=ax)
            
        else:
            win_type = self.window_combo.combo.currentText()
            kwargs = {'beta': 14.0} if win_type == 'kaiser' else {}
            
            f_bins, mag_db, _ = compute_fft(working_audio, self.current_sr, window_type=win_type, **kwargs)
            linear_mags = 10**(mag_db/20.0) 
            
            if plot_choice == "Spectrum":
                harmonics_data = None
                if self.toggle_harmonics.isChecked():
                    f0 = find_fundamental_frequency(f_bins, linear_mags)
                    harmonics_data = find_harmonics(f_bins, linear_mags, f0)
                    centroid = calculate_spectral_centroid(f_bins, linear_mags)
                    thd = calculate_thd(f_bins, linear_mags, f0)
                    metrics_text = f"<b>Freq Domain:</b> F0: {f0:.1f} Hz | Centroid: {centroid:.1f} Hz | THD: {thd:.2f}%"

                if y_axis == "Linear Magnitude":
                    ax.plot(f_bins, linear_mags, label=f"Linear ({f_type})")
                    ax.set_xscale('log')
                    ax.set_xlim(20, 20000)
                    ax.set_title("Frequency Spectrum (Linear Magnitude)")
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylabel("Linear Amplitude")
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.legend()
                    plot_kwargs = {'frequencies': f_bins, 'magnitudes_db': linear_mags, 'label': f"Linear ({f_type})"} 
                else:
                    plot_kwargs = {'frequencies': f_bins, 'magnitudes_db': mag_db, 'color': np.random.rand(3,), 
                                   'harmonics_data': harmonics_data, 'label': f"dBFS ({f_type})"}
                    plot_spectrum(**plot_kwargs, ax=ax)
                
            elif plot_choice in ["IMD Stem", "Linearity Curve"]:
                adv_vals = self.advanced_params.get_values()
                f1 = float(str(adv_vals.get("F1 (Hz)", "1000")).strip() or 1000.0)
                f2 = float(str(adv_vals.get("F2 (Hz)", "1100")).strip() or 1100.0)
                f3_raw = str(adv_vals.get("F3 (Hz) (Optional)", "2000")).strip()
                f3 = float(f3_raw) if f3_raw else None
                
                metrics = measure_linearity_metrics(f_bins, linear_mags, f1, f2, f3)
                
                if plot_choice == "IMD Stem":
                    plot_kwargs = {'metrics': metrics}
                    plot_imd_stems(**plot_kwargs, ax=ax)
                elif plot_choice == "Linearity Curve":
                    x_int = (metrics['p_fund_avg'] - metrics['imd3_db']) / 2.0
                    inputs = np.linspace(-40, x_int + 5, 100)
                    fund_outputs = 1 * inputs + metrics['p_fund_avg']
                    imd3_outputs = 3 * inputs + metrics['imd3_db']
                    
                    ax.plot(inputs, fund_outputs, 'b-', linewidth=2, label="Fundamental (1:1 Slope)")
                    ax.plot(inputs, imd3_outputs, 'r--', linewidth=2, label="IMD3 (3:1 Slope)")
                    ax.plot(0, metrics['p_fund_avg'], 'bo', markersize=8, label=f"Measured Fund ({metrics['p_fund_avg']:.1f} dB)")
                    ax.plot(0, metrics['imd3_db'], 'ro', markersize=8, label=f"Measured IMD3 ({metrics['imd3_db']:.1f} dB)")
                    ax.plot(x_int, metrics['oip3_db'], 'k*', markersize=12, label=f"OIP3 ({metrics['oip3_db']:.1f} dB)")
                    
                    ax.set_title("Theoretical IP3 Intercept Projection")
                    ax.set_xlabel("Relative Input Power (dB)")
                    ax.set_ylabel("Output Power (dBFS)")
                    ax.set_ylim(-100, max(0, metrics['oip3_db'] + 10))
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.legend()
                    plot_kwargs = {'frequencies': f_bins, 'magnitudes_db': mag_db} 

        peak = calculate_peak(working_audio, in_dbfs=True)
        rms = calculate_rms(working_audio, in_dbfs=True)
        metrics_text = f"<b>Time Domain Clip:</b> Peak: {peak:.2f} dBFS | RMS: {rms:.2f} dBFS <br>" + metrics_text

        self.metrics_display.setHtml(metrics_text)
        self.ana_fig.tight_layout()
        self.ana_canvas.draw()
        
        title_tag = f"{plot_choice} [Linear]" if y_axis == "Linear Magnitude" else f"{plot_choice} [{f_type}]"
        self.add_graph_to_tree(title_tag, plot_choice, plot_kwargs)

    # ==========================================
    # TAB 5: REPORT GENERATION (MULTI-FILE SWEEP)
    # ==========================================
    def init_report_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        control_layout = QHBoxLayout()
        
        self.sweep_files_btn = QPushButton("Select Multi-Volume Files (.wav)")
        self.sweep_files_btn.clicked.connect(self.select_sweep_files)
        self.sweep_files_label = QLabel("No files selected.")
        
        self.vols_input = QLineEdit("10, 30, 50, 70, 90")
        self.vols_input.setPlaceholderText("Volume Labels (e.g. 10, 30, 50)")
        
        run_sweep_btn = QPushButton("Generate IIP3 Sweep & Delta Graphs")
        run_sweep_btn.clicked.connect(self.run_sweep_analysis)
        
        self.export_csv_btn = QPushButton("💾 Export Sweep Data to CSV")
        self.export_csv_btn.clicked.connect(self.export_report_csv)
        self.export_csv_btn.setEnabled(False)
        
        control_layout.addWidget(self.sweep_files_btn)
        control_layout.addWidget(self.sweep_files_label)
        control_layout.addWidget(QLabel("Volume Series (%):"))
        control_layout.addWidget(self.vols_input)
        control_layout.addWidget(run_sweep_btn)
        control_layout.addWidget(self.export_csv_btn)
        
        layout.addLayout(control_layout)
        
        self.report_fig = plt.figure(figsize=(10, 8))
        self.report_canvas = FigureCanvas(self.report_fig)
        self.report_toolbar = NavigationToolbar(self.report_canvas, self)
        
        layout.addWidget(self.report_toolbar)
        layout.addWidget(self.report_canvas)
        
        self.tabs.addTab(tab, "5. Report Generation")
        self.sweep_filepaths = []

    def select_sweep_files(self):
        filepaths, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files for Volume Sweep", "", "Audio Files (*.wav)")
        if filepaths:
            self.sweep_filepaths = sorted(filepaths)
            self.sweep_files_label.setText(f"{len(filepaths)} files loaded.")
            
    def export_report_csv(self):
        if not self.report_export_data:
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Export Data", "linearity_report_data.csv", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Volume (%)", "Fundamental Power (dBFS)", "IM2 Power (dBFS)", "IM3 Power (dBFS)", "Delta SNR (Fund - IM3)"])
                    for row in self.report_export_data:
                        writer.writerow(row)
                QMessageBox.information(self, "Success", "Data exported successfully! You can paste this directly into your project report.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def run_sweep_analysis(self):
        if not self.sweep_filepaths:
            QMessageBox.warning(self, "Warning", "Please select files first!")
            return
            
        try:
            vols = [float(v.strip()) for v in self.vols_input.text().split(',')]
        except ValueError:
            QMessageBox.warning(self, "Error", "Volume labels must be comma-separated numbers.")
            return
            
        if len(vols) != len(self.sweep_filepaths):
            QMessageBox.warning(self, "Error", "Number of volume labels must match number of files selected.")
            return
            
        adv_vals = self.advanced_params.get_values()
        f1 = float(str(adv_vals.get("F1 (Hz)", "1000")).strip() or 1000.0)
        f2 = float(str(adv_vals.get("F2 (Hz)", "1100")).strip() or 1100.0)
        f3_raw = str(adv_vals.get("F3 (Hz) (Optional)", "2000")).strip()
        f3 = float(f3_raw) if f3_raw else None
        
        clip_vals = self.clip_range.get_values()
        start_s = float(clip_vals.get("Start Time (s)", "0.5") or 0.5)
        end_s = float(clip_vals.get("End Time (s)", "1.5") or 1.5)

        fund_powers = []
        im2_powers = []
        im3_powers = []
        
        self.report_export_data = [] 
        
        for idx, fp in enumerate(self.sweep_filepaths):
            audio, sr = read_wav(fp, force_mono=True)
            start_idx = int(start_s * sr)
            end_idx = int(end_s * sr)
            working_audio = audio[start_idx:end_idx]
            
            f_bins, mag_db, _ = compute_fft(working_audio, sr, window_type='blackman')
            linear_mags = 10**(mag_db/20.0) 
            
            metrics = measure_linearity_metrics(f_bins, linear_mags, f1, f2, f3)
            
            f_pow = metrics['p_fund_avg']
            i2_pow = metrics['imd2_db']
            i3_pow = metrics['imd3_db']
            
            fund_powers.append(f_pow)
            im2_powers.append(i2_pow)
            im3_powers.append(i3_pow)
            
            self.report_export_data.append([vols[idx], f_pow, i2_pow, i3_pow, f_pow - i3_pow])
            
        self.report_fig.clear()
        ax1 = self.report_fig.add_subplot(211)
        ax2 = self.report_fig.add_subplot(212)
        
        ax1.plot(vols, fund_powers, 'o-', label=f"Fundamental", color='blue')
        ax1.plot(vols, im3_powers, '^-', label=f"IM3 (Odd)", color='red')
        ax1.plot(vols, im2_powers, 's-', label=f"IM2 (Even)", color='green')
        
        num_fit = min(3, len(vols))
        if num_fit >= 2:
            x_sub = np.array(vols[:num_fit])
            fit_fund = np.polyfit(x_sub, fund_powers[:num_fit], 1)
            fit_im3 = np.polyfit(x_sub, im3_powers[:num_fit], 1)
            fit_im2 = np.polyfit(x_sub, im2_powers[:num_fit], 1)
            
            m_f, b_f = fit_fund
            m_i3, b_i3 = fit_im3
            m_i2, b_i2 = fit_im2
            
            x_line = np.linspace(0, max(vols) * 1.5, 100)
            ax1.plot(x_line, np.polyval(fit_fund, x_line), 'b--', alpha=0.5)
            ax1.plot(x_line, np.polyval(fit_im3, x_line), 'r--', alpha=0.5)
            
            if m_f != m_i3:
                iip3 = (b_i3 - b_f) / (m_f - m_i3)
                if iip3 > 0:
                    ax1.axvline(x=iip3, color='red', linestyle=':', label=f"IIP3 ({iip3:.1f}%)")
                    
            if m_f != m_i2:
                iip2 = (b_i2 - b_f) / (m_f - m_i2)
                if iip2 > 0:
                    ax1.axvline(x=iip2, color='green', linestyle=':', label=f"IIP2 ({iip2:.1f}%)")
                    
        ax1.set_title("Input Intercept Point (IIP) Analysis")
        ax1.set_xlabel("Volume Setting (%)")
        ax1.set_ylabel("Power Magnitude (dB)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        
        delta = np.array(fund_powers) - np.array(im3_powers)
        ax2.plot(vols, delta, 'o-', linewidth=2, color='teal')
        ax2.set_title("Dynamic Range Degradation (P_out - P_im3)")
        ax2.set_xlabel("Volume Setting (%)")
        ax2.set_ylabel("Delta Power (dB)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        self.report_fig.tight_layout()
        self.report_canvas.draw()
        
        self.export_csv_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())