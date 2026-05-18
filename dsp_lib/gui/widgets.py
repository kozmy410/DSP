from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QComboBox, QLineEdit, QSlider, QPushButton, QFrame, QFormLayout, QSizePolicy)
from PyQt5.QtCore import Qt

class LabeledComboBox(QWidget):
    """A reusable dropdown with a label."""
    def __init__(self, label_text, items):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label_text)
        self.combo = QComboBox()
        self.combo.addItems(items)
        
        # Enforce a wider baseline and allow the box to expand horizontally
        self.combo.setMinimumWidth(350)
        self.combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        layout.addWidget(self.label)
        # Add stretch=1 to the combo box so it eats up the available layout space
        layout.addWidget(self.combo, stretch=1)

class DynamicParamForm(QWidget):
    """
    A form that dynamically generates input fields based on a dictionary of expected arguments.
    """
    def __init__(self):
        super().__init__()
        self.layout = QFormLayout(self)
        self.inputs = {}

    def update_form(self, param_schema):
        """
        Clears the current form and rebuilds it.
        param_schema format: {"frequency": {"type": "float", "default": "1000.0"}, ...}
        """
        # Clear existing layout
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.inputs.clear()

        # Build new inputs
        for param_name, config in param_schema.items():
            if config["type"] in ["list", "dropdown"]:
                widget = QComboBox()
                widget.addItems(config["options"])
            else:
                widget = QLineEdit()
                widget.setPlaceholderText(str(config.get("default", "")))
            
            self.inputs[param_name] = widget
            self.layout.addRow(QLabel(param_name), widget)

    def get_values(self):
        """Returns a dictionary of the current user inputs."""
        values = {}
        for name, widget in self.inputs.items():
            if isinstance(widget, QComboBox):
                values[name] = widget.currentText()
            else:
                values[name] = widget.text()
        return values

class FileItemWidget(QFrame):
    """
    A widget to display loaded files in the side panel, showing metadata
    and acting as a container for associated plots.
    """
    def __init__(self, filename, sample_rate, frames):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        
        # File Info
        duration = frames / sample_rate
        info_text = f"<b>{filename}</b><br>SR: {sample_rate} Hz | Dur: {duration:.2f}s"
        self.info_label = QLabel(info_text)
        self.info_label.setTextFormat(Qt.RichText)
        
        # Nested layout for generated plots (populated later)
        self.plots_layout = QVBoxLayout()
        
        layout.addWidget(self.info_label)
        layout.addLayout(self.plots_layout)