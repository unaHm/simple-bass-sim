#!/usr/bin/env python3
"""
main.py — modern startup for jackbasssim modular

Behavior:
- Start jack engine (real JACK if available, otherwise run a dummy audio loop)
- Autoload last preset (presets/last_preset.txt) if present
- Start PyQt GUI in a background thread
- Handle Ctrl+C for clean shutdown
"""

import sys
import threading
import time
from pathlib import Path
import jack_engine as je

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QTabWidget, QSlider,
                                 QProgressBar, QGridLayout, QGroupBox, QListWidget, QLineEdit, QCheckBox, QComboBox,
                                 QHBoxLayout)
    from PyQt6.QtCore import QTimer, Qt, QRect
    from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QBrush
except ImportError:
    print("PyQt6 not found, GUI will not be available. Please install it with: pip install PyQt6")

# Paths
ROOT = Path(__file__).resolve().parent
PRESET_DIR = ROOT / "presets"
LAST_PRESET_FILE = PRESET_DIR / "last_preset.txt"

def autoload_last_preset():
    """If last_preset.txt exists, try to load that preset file via jack_engine.apply_instrument_state."""
    try:
        if not LAST_PRESET_FILE.exists():
            print("No last_preset.txt found — skipping autoload")
            return False

        name = LAST_PRESET_FILE.read_text(encoding='utf-8').strip()
        if not name:
            print("last_preset.txt empty — skipping autoload")
            return False

        user_path = PRESET_DIR / f"{name}.json"
        factory_path = PRESET_DIR / "factory" / f"{name}.json"
        preset_path = user_path if user_path.exists() else (factory_path if factory_path.exists() else None)
        if preset_path is None:
            print(f"autoload: preset '{name}' not found in presets/ or presets/factory/ — skipping")
            return False

        print(f"autoload: loading preset '{name}'")
        ok = je.load_preset(name)
        print("autoload:", "success" if ok else "failed to apply instrument state")
        return ok
    except Exception as e:
        print("autoload failed:", e)
        return False


def start_jack_or_dummy():
    """Start JACK client if available, otherwise start dummy audio loop (background thread)."""
    # Prefer je.start_jack_client if present
    started = False
    try:
        if hasattr(je, 'start_jack_client'):
            ok = je.start_jack_client()
            # je.start_jack_client should return True on success; accept None as success
            if ok is None or ok is True:
                print("JACK client start attempted (start_jack_client returned OK).")
                started = True
        else:
            print("jack_engine has no start_jack_client() — attempting to activate client directly.")
            try:
                if getattr(je, 'client', None) is not None:
                    je.client.activate()
                    started = True
            except Exception as e:
                print("Direct client activation failed:", e)
    except Exception as e:
        print("start_jack_client raised:", e)

    if not started:
        # run dummy audio loop in a background thread (non-blocking)
        if hasattr(je, 'run_dummy_audio_loop'):
            print("Starting dummy audio loop thread (JACK unavailable).")
            t = threading.Thread(target=je.run_dummy_audio_loop, kwargs={'samplerate': getattr(je, 'SAMPLERATE', 48000), 'frames': 128}, daemon=True, name="dummy-audio")
            t.start()
            return t
        else:
            print("No run_dummy_audio_loop() available in jack_engine — audio won't be processed.")
            return None
    return None

class HorizontalAudioLevelMeter(QWidget):
    def __init__(self, parent=None, inverted=False):
        super().__init__(parent)
        self._value = 0
        self._inverted = inverted
        self.setMinimumWidth(400)
        self.setMinimumHeight(50)

    def setValue(self, value):
        # Ensure value is within 0-100 range for the meter
        if 0 <= value <= 100:
            self._value = value
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.contentsRect()

        # 1. Draw the dark background for the entire meter
        painter.fillRect(rect, QColor("#1A1A1A"))

        # 2. Calculate the fill rectangle based on the current value
        fill_ratio = self._value / 100.0
        fill_width = int(rect.width() * fill_ratio)
        if self._inverted:
            fill_rect = QRect(rect.right() - fill_width, rect.top(), fill_width, rect.height())
        else:
            fill_rect = QRect(rect.left(), rect.top(), fill_width, rect.height())

        # 3. Define the gradient that spans the entire width of the meter
        if self._inverted:
            gradient = QLinearGradient(float(rect.right()), float(rect.top()), float(rect.left()), float(rect.top()))
            gradient.setColorAt(0.0, QColor("green"))   # Green at the rightmost (0% of gradient)
            gradient.setColorAt(0.4, QColor("yellow"))  # Yellow
            gradient.setColorAt(0.8, QColor("orange"))  # Orange
            gradient.setColorAt(1.0, QColor("red"))     # Red at the leftmost (100% of gradient)
        else:
            gradient = QLinearGradient(float(rect.left()), float(rect.top()), float(rect.right()), float(rect.top()))
            gradient.setColorAt(0.0, QColor("green"))
            gradient.setColorAt(0.4, QColor("yellow"))
            gradient.setColorAt(0.8, QColor("orange"))
            gradient.setColorAt(1.0, QColor("red"))

        # 4. Fill the calculated rectangle with the gradient
        painter.fillRect(fill_rect, gradient)

        # 5. Draw the border on top
        painter.setPen(QColor(Qt.GlobalColor.black))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.end()

class MainWindow(QMainWindow):
    def __init__(self, polling_hz=20):
        super().__init__()
        self.setWindowTitle("jackbasssim Control")
        self.setGeometry(100, 100, 1024, 600)
        self._controls = {}
        
        # Main layout with tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self._create_main_tab()
        self._create_effects_tab()
        self._create_presets_tab()
        self._create_pickup_editor_tab()
        
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / polling_hz))
        self.timer.timeout.connect(self.update_status)
        self.timer.start()
        
        # Initial population of presets and state
        self.update_status(force_full_update=True)

    def _create_main_tab(self):
        """Create the main tab for meters and mix controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.tabs.addTab(tab, "Mix & Meters")

        # Meters Group
        meters_group = QGroupBox("Meters")
        meters_layout = QGridLayout()
        meters_group.setLayout(meters_layout)
        self._add_meter(meters_layout, 0, "CPU", "cpu_load",0, 100)
        self._add_meter(meters_layout, 1, "Peak", "peak_db", -60, 0)
        self._add_meter(meters_layout, 2, "Limiter GR", "limiter_gr_db", 0, 10, inverted=True)
        self._add_meter(meters_layout, 3, "Comp GR", "comp_gr_db", 0, 20, inverted=True)
        layout.addWidget(meters_group)

        # Sliders Group for Mix controls
        sliders_group = QGroupBox("Mixer")
        sliders_layout = QGridLayout()
        sliders_group.setLayout(sliders_layout)
        self._add_slider(sliders_layout, 0, "P1 Vol", "p1", 0, 100, 100.0)
        self._add_slider(sliders_layout, 1, "P2 Vol", "p2", 0, 100, 100.0)
        self._add_slider(sliders_layout, 2, "Master", "mg", 0, 200, 2.0)
        layout.addWidget(sliders_group)

        layout.addStretch()

    def _create_effects_tab(self):
        """Create the tab for all effect parameters."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.tabs.addTab(tab, "Effects")

        # Effects Bypass Group
        bypass_group = QGroupBox("Effect Activation")
        bypass_layout = QHBoxLayout()
        bypass_group.setLayout(bypass_layout)
        self._add_bypass_toggle(bypass_layout, "SVF", "env_filter")
        self._add_bypass_toggle(bypass_layout, "Octaver", "octaver")
        self._add_bypass_toggle(bypass_layout, "Compressor", "comp")
        layout.addWidget(bypass_group)

        # Effect Parameters
        svf_group = self._create_svf_group()
        oct_group = self._create_octaver_group()
        comp_group = self._create_compressor_group()

        layout.addWidget(svf_group)
        layout.addWidget(oct_group)
        layout.addWidget(comp_group)

        # Global Compressor Save
        comp_save_group = QGroupBox("Global Compressor Settings")
        comp_save_layout = QVBoxLayout()
        save_comp_btn = QPushButton("Save Compressor (global)")
        save_comp_btn.clicked.connect(self._save_global_compressor)
        comp_save_layout.addWidget(save_comp_btn)
        comp_save_group.setLayout(comp_save_layout)
        layout.addWidget(comp_save_group)

        layout.addStretch()

    def _create_svf_group(self):
        group = QGroupBox("State Variable Filter")
        layout = QGridLayout(group)
        self._add_slider(layout, 0, "SVF Cutoff", "svf_base_cutoff", 100, 5000)
        self._add_slider(layout, 1, "SVF Env", "svf_env_depth", 0, 5000)
        return group

    def _create_octaver_group(self):
        group = QGroupBox("Octaver")
        layout = QGridLayout(group)
        self._add_slider(layout, 0, "Octave Mix", "oct_mix", 0, 100, 100.0)
        return group

    def _create_compressor_group(self):
        group = QGroupBox("Compressor")
        layout = QGridLayout(group)
        self._add_slider(layout, 0, "Comp Thresh", "comp_threshold", -40, 0)
        self._add_slider(layout, 1, "Comp Ratio", "comp_ratio", 10, 120, 10.0)
        self._add_slider(layout, 2, "Comp Makeup (dB)", "comp_makeup", 0, 6, 1.0)
        return group

    def _create_presets_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.tabs.addTab(tab, "Presets")

        # Factory Presets
        factory_group = QGroupBox("Factory Presets")
        factory_layout = QVBoxLayout()
        self._controls['factory_list'] = QListWidget()
        load_factory_btn = QPushButton("Load Factory Preset")
        load_factory_btn.clicked.connect(self._load_factory_preset)
        factory_layout.addWidget(self._controls['factory_list'])
        factory_layout.addWidget(load_factory_btn)
        factory_group.setLayout(factory_layout)
        layout.addWidget(factory_group)

        # User Presets
        user_group = QGroupBox("User Presets")
        user_layout = QVBoxLayout()
        self._controls['user_list'] = QListWidget()
        load_user_btn = QPushButton("Load User Preset")
        load_user_btn.clicked.connect(self._load_user_preset)
        user_layout.addWidget(self._controls['user_list'])
        user_layout.addWidget(load_user_btn)
        user_group.setLayout(user_layout)
        layout.addWidget(user_group)

        # Save/Delete
        manage_group = QGroupBox("Manage")
        manage_layout = QGridLayout()
        self._controls['preset_name'] = QLineEdit()
        self._controls['preset_name'].setPlaceholderText("New preset name...")
        save_btn = QPushButton("Save")
        delete_btn = QPushButton("Delete")
        save_btn.clicked.connect(self._save_preset)
        delete_btn.clicked.connect(self._delete_preset)
        manage_layout.addWidget(QLabel("Name:"), 0, 0)
        manage_layout.addWidget(self._controls['preset_name'], 0, 1)
        manage_layout.addWidget(save_btn, 1, 0)
        manage_layout.addWidget(delete_btn, 1, 1)
        manage_group.setLayout(manage_layout)
        layout.addWidget(manage_group)

        self._controls['preset_status_label'] = QLabel("")
        self._controls['preset_status_label'].setObjectName("preset_status_label")
        layout.addWidget(self._controls['preset_status_label'])

        # Add the original reload button for convenience
        reload_button = QPushButton("Reload Last Preset")
        reload_button.clicked.connect(self._reload_last_preset)
        layout.addWidget(reload_button)

        layout.addStretch()

    def _create_pickup_editor_tab(self):
        """Create the tab for editing pickup configuration."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.tabs.addTab(tab, "Pickup Editor")

        # Enable Pickup 2 Checkbox
        self._controls['p2_enabled'] = QCheckBox("Enable Pickup 2")
        self._controls['p2_enabled'].toggled.connect(self._on_p2_enabled_change)
        layout.addWidget(self._controls['p2_enabled'])

        # Pickup 1 Group
        p1_group = QGroupBox("Pickup 1")
        p1_layout = QGridLayout()
        p1_group.setLayout(p1_layout)
        self._add_combo_box(p1_layout, 0, "P1 Type", "p1_type", ['single', 'splitP', 'humbucker', 'soapbar', 'none'])
        self._add_slider(p1_layout, 1, "P1 Closest (mm)", "p1_dist", 0, 2000, 10.0) # 0-200.0
        layout.addWidget(p1_group)

        # Pickup 2 Group
        self._controls['p2_group'] = QGroupBox("Pickup 2")
        p2_layout = QGridLayout()
        self._controls['p2_group'].setLayout(p2_layout)
        self._add_combo_box(p2_layout, 0, "P2 Type", "p2_type", ['single', 'splitP', 'humbucker', 'soapbar', 'none'])
        self._add_slider(p2_layout, 1, "P2 Closest (mm)", "p2_dist", 0, 2000, 10.0) # 0-200.0
        layout.addWidget(self._controls['p2_group'])

        layout.addStretch()

    def _add_combo_box(self, layout, row, label_text, key, items):
        label = QLabel(label_text)
        combo = QComboBox()
        combo.addItems(items)
        layout.addWidget(label, row, 0)
        layout.addWidget(combo, row, 1, 1, 2) # Span 2 columns
        self._controls[key] = combo
        combo.currentTextChanged.connect(lambda text, k=key: self._on_pickup_type_change(k, text))

    def _on_pickup_type_change(self, key, text):
        slot = 0 if 'p1' in key else 1
        if hasattr(je, 'set_pickup_type'):
            je.set_pickup_type({'pickup_slot': slot, 'type': text})

    def _on_p2_enabled_change(self, enabled):
        self._controls['p2_group'].setEnabled(enabled)
        # When enabling/disabling, we change the underlying type to 'none' or 'single'
        # This is more robust than just disabling the UI.
        new_type = 'single' if enabled else 'none'
        if hasattr(je, 'set_pickup_type'):
            je.set_pickup_type({'pickup_slot': 1, 'type': new_type})

        # Automatically set P2 volume to 1.0 when enabled
        if enabled:
            if hasattr(je, 'update_controls'):
                je.update_controls({'p2': 1.0})
            # Update UI for P2 volume immediately
            p2_ctrl = self._controls.get('p2')
            if p2_ctrl and 'slider' in p2_ctrl and 'label' in p2_ctrl:
                p2_ctrl['slider'].setValue(int(1.0 * p2_ctrl.get('multiplier', 1.0)))
                p2_ctrl['label'].setText(f"{1.0:.2f}")

    def _add_meter(self, layout, row, label_text, key, min_val, max_val, inverted=False):
        label = QLabel(f"{label_text}: N/A")
        bar = HorizontalAudioLevelMeter(inverted=inverted)
        layout.addWidget(label, row, 0)
        layout.addWidget(bar, row, 1)
        self._controls[key] = {'label': label, 'bar': bar}

    def _add_slider(self, layout, row, label_text, key, min_val, max_val, multiplier=1.0):
        label = QLabel(label_text)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        value_label = QLabel("N/A")
        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)
        self._controls[key] = {'slider': slider, 'label': value_label, 'multiplier': multiplier}
        slider.valueChanged.connect(lambda v, k=key: self._on_param_change(k, v))

    def _add_bypass_toggle(self, layout, label_text, key):
        button = QPushButton(label_text)
        button.setCheckable(True) # Make it a toggle button
        layout.addWidget(button)
        self._controls[key] = button
        # INVERTED LOGIC: checked means effect is ACTIVE (bypass=False)
        button.toggled.connect(lambda state, k=key: self._on_bypass_change(k, not state))

    def _on_param_change(self, key, value):
        if not self._controls[key]['slider'].isSliderDown(): return
        multiplier = self._controls[key].get('multiplier', 1.0)
        final_value = value / multiplier if key in ['p1', 'p2', 'oct_mix', 'comp_ratio'] or key.endswith('_dist') else (value / 100.0 if key != 'comp_makeup' and multiplier != 1.0 else value)

        if key in ['p1', 'p2', 'mg']:
             if hasattr(je, 'update_controls'): je.update_controls({key: final_value})
        elif key.endswith('_dist'):
            slot = 0 if 'p1' in key else 1
            if hasattr(je, 'set_pickup_distance'):
                je.set_pickup_distance({'pickup_slot': slot, 'distance_mm': final_value})
        elif key == 'oct_mix':
            if hasattr(je, 'update_effects_params'):
                # oct_mix slider controls two parameters
                je.update_effects_params({'param': 'oct_dry', 'value': 1.0 - final_value})
                je.update_effects_params({'param': 'oct_sub_gain', 'value': final_value})
        else:
            if hasattr(je, 'update_effects_params'): je.update_effects_params({'param': key, 'value': final_value})

    def _on_bypass_change(self, key, is_bypassed):
        if hasattr(je, 'set_bypass'):
            je.set_bypass({'name': key, 'state': is_bypassed})

    def _reload_last_preset(self):
        self._controls['preset_status_label'].setText("Reloading last preset...")
        QApplication.processEvents()
        ok = autoload_last_preset()
        self._controls['preset_status_label'].setText("Reload success!" if ok else "Reload failed.")
        self.update_status(force_full_update=True) # Refresh UI

    def _load_factory_preset(self):
        item = self._controls['factory_list'].currentItem()
        if item and hasattr(je, 'load_preset'):
            self._controls['preset_status_label'].setText(f"Loading {item.text()}...")
            je.load_preset(item.text())
            self.update_status(force_full_update=True)

    def _load_user_preset(self):
        item = self._controls['user_list'].currentItem()
        if item and hasattr(je, 'load_preset'):
            self._controls['preset_status_label'].setText(f"Loading {item.text()}...")
            je.load_preset(item.text())
            self.update_status(force_full_update=True)

    def _save_preset(self):
        name = self._controls['preset_name'].text().strip()
        if name and hasattr(je, 'save_preset'):
            self._controls['preset_status_label'].setText(f"Saving {name}...")
            je.save_preset(name)
            self.update_status(force_full_update=True)

    def _delete_preset(self):
        item = self._controls['user_list'].currentItem()
        if item and hasattr(je, 'delete_preset'):
            self._controls['preset_status_label'].setText(f"Deleting {item.text()}...")
            je.delete_preset(item.text())
            self.update_status(force_full_update=True)

    def _save_global_compressor(self):
        if hasattr(je, 'save_global_effects'):
            ok = je.save_global_effects()
            # We can use the preset status label to show the result
            self._controls['preset_status_label'].setText("Global compressor settings saved!" if ok else "Failed to save compressor settings.")

    def update_status(self, force_full_update=False):
        """Fetch state from jack_engine and update the UI."""
        if not hasattr(je, 'get_full_state'): return

        state = je.get_full_state(force_full_update)
        if not state: return

        # Update meters
        for key in ['cpu_load', 'peak_db', 'limiter_gr_db', 'comp_gr_db']:
            if key in state and key in self._controls:
                val = state[key]
                self._controls[key]['label'].setText(f"{self._controls[key]['label'].text().split(':')[0]}: {val:.2f}")
                # Map actual values to 0-100 for HorizontalAudioLevelMeter
                meter_val = 0
                if key == 'peak_db':
                    # Map -60dB to 0dB to 0-100
                    meter_val = int(((val + 60) / 60) * 100)
                elif key == 'limiter_gr_db':
                    # Map 0dB to 10dB (reduction) to 0-100
                    meter_val = int((abs(val) / 10) * 100)
                elif key == 'comp_gr_db':
                    # Map 0dB to 20dB (reduction) to 0-100
                    meter_val = int((abs(val) / 20) * 100)
                elif key == 'cpu_load':
                    # CPU load is already 0-100
                    meter_val = int(val)
                else:
                    meter_val = int(val) # Default for other meters if any

                self._controls[key]['bar'].setValue(meter_val)


        # Update sliders and bypasses (only if not interacting)
        for key, ctrl in self._controls.items():
            if isinstance(ctrl, dict) and 'slider' in ctrl:
                if not ctrl['slider'].isSliderDown():
                    # Special handling for dist sliders which have different keys in state
                    state_key = f"p{key[1]}_dist" if key.endswith('_dist') else key
                    if state_key in state:
                        val = state[state_key]
                        multiplier = ctrl.get('multiplier', 1.0)
                        ctrl['slider'].setValue(int(val * multiplier if key in ['p1', 'p2', 'oct_mix', 'comp_ratio'] or key.endswith('_dist') else (val * 100.0 if key != 'comp_makeup' and multiplier != 1.0 else val)))
                        ctrl['label'].setText(f"{val:.2f}")
            elif isinstance(ctrl, QPushButton) and ctrl.isCheckable(): # Now handling checkable PushButtons
                if key in state:  # state is bypass state
                    ctrl.setChecked(not state[key]) # checked = active = not bypassed
            elif isinstance(ctrl, QComboBox):
                if key in state:
                    # Block signals to prevent feedback loop
                    ctrl.blockSignals(True)
                    ctrl.setCurrentText(state[key])
                    ctrl.blockSignals(False)

        # Special handling for pickup editor state
        if 'p1_type' in state and 'p2_type' in state:
            p2_enabled = state['p2_type'] != 'none'
            self._controls['p2_enabled'].blockSignals(True)
            self._controls['p2_enabled'].setChecked(p2_enabled)
            self._controls['p2_enabled'].blockSignals(False)
            self._controls['p2_group'].setEnabled(p2_enabled)


        # Update preset lists if they exist in state
        if 'factory_presets' in state:
            self._controls['factory_list'].clear()
            self._controls['factory_list'].addItems(state['factory_presets'])
        if 'user_presets' in state:
            self._controls['user_list'].clear()
            self._controls['user_list'].addItems(state['user_presets'])


def run_application_logic():
    """This is the main logic for the audio engine, intended to run in a background thread."""
    global _dummy_audio_thread
    print("Application logic starting...")

    # 1) Start JACK or dummy audio loop
    _dummy_audio_thread = start_jack_or_dummy()

    # 2) Autoload last preset (best-effort)
    try:
        autoload_last_preset()
    except Exception as e:
        print("autoload_last_preset error:", e)

    # Keep the thread alive for background tasks if needed
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received in application logic thread.")
    except Exception as e:
        print("Application logic thread error:", e)
    finally:
        print("Application logic thread exiting.")


STYLESHEET = """
    QWidget {
        background-color: #222222;
        color: #FFB74D; /* Amber */
        font-family: "Lucida Console", Monaco, monospace;
        font-size: 14px;
        font-weight: bold;
    }
    QMainWindow {
        background-color: #1A1A1A;
    }
    QTabWidget::pane {
        border: 1px solid #444;
        background-color: #222222;
    }
    QTabBar::tab {
        background: #333333;
        color: #AAAAAA;
        padding: 10px;
        border: 1px solid #444;
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected, QTabBar::tab:hover {
        background: #222222;
        color: #FFB74D;
    }
    QGroupBox {
        background-color: #2A2A2A;
        border: 1px solid #444;
        border-radius: 5px;
        margin-top: 1ex;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 5px;
    }
    QPushButton {
        background-color: #444444;
        border: 1px solid #555;
        padding: 8px;
        border-radius: 3px;
    }
    QPushButton:hover {
        background-color: #555555;
    }
    QPushButton:pressed {
        background-color: #333333;
        border-style: inset;
    }
    QPushButton:checked {
        background-color: #004400; /* Darker green background when active */
        border: 1px solid #00FF00; /* Green border */
        color: #00FF00; /* Green text */
        border-style: inset; /* Pushed-in look */
    }
    QPushButton:checked:hover {
        background-color: #005500; /* Slightly lighter green on hover when checked */
        border-color: #33FF33;
    }
    QSlider::groove:horizontal {
        border: 1px solid #333;
        height: 4px;
        background: #1A1A1A;
        margin: 2px 0;
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #888888;
        border: 1px solid #AAAAAA;
        width: 18px;
        margin: -6px 0;
        border-radius: 3px;
    }
    QSlider::handle:horizontal:hover {
        background: #AAAAAA;
    }
    QListWidget, QLineEdit, QComboBox {
        background-color: #1A1A1A;
        border: 1px solid #444;
        padding: 5px;
        border-radius: 3px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox::down-arrow {
        image: url(./arrow.png); /* A proper implementation would use a resource file */
    }
    QLabel[objectName="preset_status_label"] {
        color: #FFFFFF;
    }
"""

def shutdown_hook():
    """Gracefully shut down the audio client."""
    print("Shutdown requested.")
    try:
        if hasattr(je, 'client') and je.client is not None and je.client.active:
            je.client.deactivate()
            print("JACK client deactivated.")
    except Exception as e:
        print(f"Error during JACK deactivation: {e}")
    print("Exiting.")


if __name__ == '__main__':
    if 'PyQt6' not in sys.modules:
        print("PyQt6 is required to run. Please install it.")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)  # Apply stylesheet here
    app.aboutToQuit.connect(shutdown_hook)

    window = MainWindow(polling_hz=20)
    window.show()

    # Start the application logic in a separate thread
    app_logic_thread = threading.Thread(target=run_application_logic, name="app-logic", daemon=True)
    app_logic_thread.start()

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        # The shutdown_hook will be called by aboutToQuit
        print("\nKeyboardInterrupt received, shutting down.")
