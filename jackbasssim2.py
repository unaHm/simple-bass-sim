import jack
import numpy as np
import threading
import json
import os
from flask import Flask, render_template_string, request, flash, redirect, url_for, jsonify, session 
from flask_socketio import SocketIO, emit
import math
import sys
import numba as nb
import time

# Sources:
# https://majormixing.com/how-to-set-compressor-attack-release-times-your-complete-guide/

# =========================================================================
# 🚀 USAGE GUIDE
# =========================================================================
#
# This application is a real-time bass guitar simulator combining Digital
# Waveguide Synthesis (DWS) with a full suite of post-processing effects
# and a live, responsive Web UI.
#
# -------------------------------------------------------------------------
# 1. PREREQUISITES (SETUP)
# -------------------------------------------------------------------------
# Ensure you have the required Python packages and a running **JACK Audio 
# Connection Kit** server.
#
# # Install Python Dependencies (Use a virtual environment like '.venv')
# pip install jack-client numpy flask flask-socketio numba
#
# -------------------------------------------------------------------------
# 2. EXECUTION
# -------------------------------------------------------------------------
# Run the script directly from your terminal:
#
# python your_script_name.py
#
# Upon successful execution, the console will display the following messages:
#
# JACK client activated. Connect ports using QjackCtl.
# Access the web UI at http://<Your_Pi_IP_Address>:5000
#
# -------------------------------------------------------------------------
# 3. AUDIO CONNECTION (JACK)
# -------------------------------------------------------------------------
# Use a JACK control utility (like **QjackCtl**) to route the audio:
#
# - **Input:** Connect your audio input source (e.g., your sound card's 
#   microphone/line input) to the 6 input ports of the running JACK application 
#   client. The simulator expects a trigger pulse for each string.
#
# - **Output:** Connect the two stereo output ports (`output_1`, `output_2`) 
#   of the JACK application client to your sound card's playback ports 
#   (e.g., `system:playback_1`, `system:playback_2`).
#
# -------------------------------------------------------------------------
# 4. WEB UI ACCESS
# -------------------------------------------------------------------------
# Open a web browser on your local network and navigate to the displayed address:
#
# - **Access:** `http://<Your_Pi_IP_Address>:5000` (If running locally, 
#   use `http://127.0.0.1:5000`).
#
# The UI provides **real-time control** over all string parameters, pickup 
# configuration, and the four post-processing effects. All slider and checkbox 
# changes are applied instantly via AJAX/WebSockets without requiring a page refresh.
#
# - **Real-Time Feedback:** The **Level Meter** and **Gain Reduction (GR)** #   meters update 30 times per second, providing crucial visual feedback for 
#   tuning the compressor and limiter thresholds.
#
# =========================================================================
# DSP ALGORITHM SOURCES & THEORY
# =========================================================================

# 1. String Synthesis (The Core Sound Engine)
#    - Algorithm: **Karplus-Strong Synthesis**
#    - Foundation: Digital Waveguide Synthesis (DWS). The simulation uses a 
#      **Feedback Comb Filter** (Karplus-Strong) to model the initial tone and 
#      **Low-Pass Filtering** to model damping/decay.
#    - Reference: Smith, Julius O. "Physical Modeling using Digital Waveguides." 
#      (Stanford University / CCRMA). Provides the definitive framework for 
#      DWS and its application to string and instrument modeling.

# 2. Pickups and Tone
#    - Concept: Modeling magnetic pickups as a **differentiating filter** (high-pass 
#      slope) and tone knobs as standard **first-order low-pass filters**.
#    - Reference: Abel, J. S., & Smith, J. O. (1999). "The Phasing of Strings and 
#      Pickups." Modeling of pickup position as sum of delays.

# 3. Effects: Envelope Filter
#    - Algorithm: **State Variable Filter (SVF)** controlled by an **Envelope Follower**.
#      The SVF is typically implemented using the Tustin (Bilinear) Transform.
#    - Reference: Dattorro, J. (1997). "Implementation of the Digital Waveguide 
#      Model for String Instruments." Details digital filter structures like SVF.

# 4. Effects: Compressor / Limiter
#    - Algorithm: **Feed-forward topology** utilizing an **Envelope Follower** #      for side-chain detection and gain smoothing.
#    - Reference: Zölzer, U. (2011). "DAFX: Digital Audio Effects." Chapters on 
#      Dynamics Processing detail the math for gain computation (ratio, threshold) 
#      and time constant implementation (attack/release smoothing).

# 5. Effects: Octaver (Simplified Sub-Octave)
#    - Algorithm: A **Half-Wave Rectifier** combined with **resampling/interpolation** #      to create a signal at half the fundamental frequency. (More complex versions 
#      use pitch detection/shifting.)
#    - Reference: General DSP techniques for frequency division (e.g., fractional 
#      delay and downsampling).

# 6. Performance Optimization
#    - Technique: **Numba** for **Just-In-Time (JIT) compilation** into machine code 
#      and use of **NumPy** for vectorized operations within the real-time audio thread.
#    - Reference: Numba Documentation (for `@numba.njit(fastmath=True, cache=True)` 
#      usage and array passing conventions).

# =========================================================================
# WEB & NETWORKING SOURCES
# =========================================================================

# 7. Web UI & Control
#    - Frameworks: **Flask** (Python web microframework) and **Jinja2** (templating).
#    - Communication: **AJAX (Asynchronous JavaScript and XML/JSON)** used for 
#      instant parameter updates without page reloads.

# 8. Real-Time Metering
#    - Technique: **WebSockets** via **Flask-SocketIO** for bi-directional, low-latency 
#      communication, enabling the real-time level and gain reduction meters.
#    - Reference: Flask-SocketIO Documentation (for server and client implementation 
#      of real-time data streaming).
# Sources:
# 1. Compressor/Limiter Ratios and Timing:
#    https://majormixing.com/how-to-set-compressor-attack-release-times-your-complete-guide/
#
# 2. Real-time Audio Processing (JACK Client):
#    https://jackclient-python.readthedocs.io/en/latest/
#
# 3. Web Sockets for Real-time Metering (Flask-SocketIO):
#    https://flask-socketio.readthedocs.io/en/latest/
#
# 4. JSON Serialization of NumPy Data Types:
#    (Fix implemented by casting numpy.float32 to standard Python float before serialization.)
#    https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.float32

PRESET_DIR = 'presets'
# Ensure this directory exists in your main execution block or startup logic:
if not os.path.exists(PRESET_DIR):
    os.makedirs(PRESET_DIR)
# --- Configuration File Path (Omitted for brevity) ---
CONFIG_FILE = 'pickup_config.json'

# --- Configuration Variables (Omitted for brevity) ---
SHARED_LENGTH = 863.6 
IS_PICKUP2_ENABLED = True 
PICKUP1_VOL = 1.0 
PICKUP2_VOL = 1.0 
MASTER_TONE = 10000.0 

STRING_PARAMS_VIRTUAL = [
    {'note': 'B0', 'freq': 30.9, 'dist_ff1': 100.0, 'dist_ff2': 200.0},
    {'note': 'E1', 'freq': 41.2, 'dist_ff1': 100.0, 'dist_ff2': 200.0},
    {'note': 'A1', 'freq': 55.0, 'dist_ff1': 100.0, 'dist_ff2': 200.0},
    {'note': 'D2', 'freq': 73.4, 'dist_ff1': 100.0, 'dist_ff2': 200.0},
    {'note': 'G2', 'freq': 98.0, 'dist_ff1': 100.0, 'dist_ff2': 200.0},
    {'note': 'C3', 'freq': 130.8, 'dist_ff1': 100.0, 'dist_ff2': 200.0},
]
REAL_PICKUP_DISTS_FB = [100.0] * 6 
PICKUP_MODELS = {
    'p_bass': {'name': 'Fender P-Bass (Split-Coil)', 'offset_ms': 0.2},
    'j_bass': {'name': 'Fender J-Bass (Single-Coil)', 'offset_ms': 0.5},
    'musicman': {'name': 'MusicMan StingRay (Humbucker)', 'offset_ms': 0.8},
}
SELECTED_PICKUP1 = 'j_bass'
SELECTED_PICKUP2 = 'p_bass'
SELECTED_PRESET = 'custom'
PRESETS = {
    'custom': {'name': 'Custom', 'strings': STRING_PARAMS_VIRTUAL, 'pickup1': SELECTED_PICKUP1, 'pickup2': SELECTED_PICKUP2, 'pickup2_enabled': True,},
    'fender_p': {'name': 'Fender Precision (Virtual)', 'strings': [ {'note': n['note'], 'freq': n['freq'], 'dist_ff1': 139.6, 'dist_ff2': 139.6} for n in STRING_PARAMS_VIRTUAL ], 'pickup1': 'p_bass', 'pickup2': 'p_bass', 'pickup2_enabled': False,},
    '60s_jazz': {'name': '60s Fender Jazz (Virtual)', 'strings': [ {'note': n['note'], 'freq': n['freq'], 'dist_ff1': 152.0, 'dist_ff2': 54.0} for n in STRING_PARAMS_VIRTUAL ], 'pickup1': 'j_bass', 'pickup2': 'j_bass', 'pickup2_enabled': True,},
    'musicman': {'name': 'MusicMan StingRay (Virtual)', 'strings': [ {'note': n['note'], 'freq': n['freq'], 'dist_ff1': 111.6, 'dist_ff2': 111.6} for n in STRING_PARAMS_VIRTUAL ], 'pickup1': 'musicman', 'pickup2': 'musicman', 'pickup2_enabled': False,},
    '70s_jazz': {'name': '70s Fender Jazz (Virtual)', 'strings': [ {'note': n['note'], 'freq': n['freq'], 'dist_ff1': 152.0, 'dist_ff2': 102.0} for n in STRING_PARAMS_VIRTUAL ], 'pickup1': 'j_bass', 'pickup2': 'j_bass', 'pickup2_enabled': True,},
    'hofner': {'name': 'Hofner \'Beatle\' Bass (Virtual)', 'strings': [ {'note': n['note'], 'freq': n['freq'], 'dist_ff1': 130.0, 'dist_ff2': 140.0} for n in STRING_PARAMS_VIRTUAL ], 'pickup1': 'p_bass', 'pickup2': 'p_bass', 'pickup2_enabled': True,},
    'warwick_thumb': {'name': 'Warwick Thumb Bass (Virtual)', 'strings': [ {'note': n['note'], 'freq': n['freq'], 'dist_ff1': 80.0, 'dist_ff2': 60.0} for n in STRING_PARAMS_VIRTUAL ], 'pickup1': 'musicman', 'pickup2': 'musicman', 'pickup2_enabled': True,},
}

# --- NEW GLOBAL EFFECT VARIABLES ---
EFFECTS_BYPASS = {
    'env_filter': True,
    'octaver': True,
    'comp': False
}
EFFECT_INSTANCES = {}
DB_TO_AMP = lambda db: 10.0**(db / 20.0)
AMP_TO_DB = lambda amp: 20.0 * np.log10(amp)
# --- NEW CONFIGURATION VARIABLES (Must be loaded/saved via JSON) ---
ENV_FILTER_FREQ = 100.0   # Hz
ENV_FILTER_Q = 1.0
ENV_FILTER_MIX = 0.5

OCTAVER_OCTAVE_VOL = 0.5  # Sub-octave volume
OCTAVER_DRY_VOL = 1.0     # Dry signal volume

COMP_THRESHOLD = -18.0    # dB
COMP_RATIO = 4.0
COMP_ATTACK_MS = 10.0
COMP_RELEASE_MS = 100.0

LIMITER_THRESHOLD = -0.5  # dB (Final output protection)
LIMITER_RELEASE_MS = 50.0

# New Global variable to store the latest peak level
LAST_PEAK_DB = -120.0

# New Global variables to store the maximum gain reduction in dB
LAST_COMP_GR_DB = 0.0  # Gain Reduction is non-positive (0.0 means no reduction)
LAST_LIMITER_GR_DB = 0.0
DB_TO_AMP = lambda db: 10.0**(db / 20.0)

params_lock = threading.Lock()
update_event = threading.Event()

# Global variables for filter instances
fb_correction_filters = [] 
ff1_filters_p1, ff2_filters_p1, ff3_filters_p1 = [], [], []
ff1_filters_p2, ff2_filters_p2, ff3_filters_p2 = [], [], []
master_tone_filters_l = [] # NEW: Master Tone Filter Left Channel
master_tone_filters_r = [] # NEW: Master Tone Filter Right Channel
client = jack.Client("PythonDSPClient")

# --- DSP Filter Classes ---
class FeedbackCombFilter:
    def __init__(self, delay_length, feedback_gain):
        self.delay_length = delay_length
        self.gain = feedback_gain
        # State variables must be accessible to Numba
        self.buffer = np.zeros(delay_length, dtype='float32')
        # FIX: Change scalar index to a 1-element Numba-compatible array
        self.buffer_index = np.array([0], dtype=np.int32) 

class FeedforwardCombFilter:
    def __init__(self, delay_length, gain):
        self.delay_length = delay_length
        self.gain = gain
        # State variables must be accessible to Numba
        self.buffer = np.zeros(delay_length, dtype='float32')
        # FIX: Change scalar index to a 1-element Numba-compatible array
        self.buffer_index = np.array([0], dtype=np.int32)
        
# --- Optimized LowPassFilter Class (Refactored for Numba compatibility) ---
class LowPassFilter:
    def __init__(self, cutoff_freq, samplerate):
        self.a0 = 0.0
        self.b1 = 0.0
        # NEW: Numba-compatible mutable state array for z1
        self.z1_state = np.array([0.0], dtype=np.float32) 
        self.samplerate = samplerate
        self.set_cutoff(cutoff_freq)

    def set_cutoff(self, cutoff_freq):
        """Calculates coefficients based on a given cutoff frequency."""
        if cutoff_freq >= self.samplerate / 2:
              self.a0 = 1.0
              self.b1 = 0.0
        else:
            tau = 1.0 / (2.0 * math.pi * cutoff_freq)
            te = 1.0 / self.samplerate
            self.b1 = math.exp(-te / tau)
            self.a0 = 1.0 - self.b1
    
    # Removed the Numba-compiled process_block method, logic moved to run_lpf_numba.

# --- NEW DSP CLASS DEFINITIONS ---
class EnvelopeFollower:
    """State for tracking the signal level (used by the Envelope Filter and Compressor)."""
    def __init__(self, samplerate, attack_ms, release_ms):
        self.samplerate = samplerate
        self.envelope_state = np.array([0.0], dtype=np.float32) # Numba state I/O array
        self.set_time_constants(attack_ms, release_ms) 

    # --- FIX: THIS METHOD IS REQUIRED BY THE UI UPDATE API ---
    def set_time_constants(self, attack_ms, release_ms):
        """Calculates and sets the attack/release smoothing coefficients."""
        # Smoothing coefficients (1-pole low-pass filter)
        self.alpha_atk = math.exp(-1.0 / (attack_ms * 0.001 * self.samplerate)) if attack_ms > 0 else 0.0
        self.alpha_rel = math.exp(-1.0 / (release_ms * 0.001 * self.samplerate)) if release_ms > 0 else 0.0

class SVF: # State Variable Filter (used by the Envelope Filter)
    def __init__(self, samplerate, freq, Q):
        self.samplerate = samplerate
        self.state = np.array([0.0, 0.0], dtype=np.float32) # [z1, z2]
        self.g = 0.0
        self.R = 0.0
        self.update_coefficients(freq, Q)
    
    def update_coefficients(self, freq, Q):
        self.g = np.tan(np.pi * freq / self.samplerate)
        self.R = 1.0 / Q

class EnvelopeFilter:
    def __init__(self, samplerate, freq, Q, mix, attack_ms, release_ms):
        self.svf_l = SVF(samplerate, freq, Q)
        self.svf_r = SVF(samplerate, freq, Q)
        self.env_follower = EnvelopeFollower(samplerate, attack_ms, release_ms)
        self.mix = mix # Dry/Wet mix (0.0 to 1.0)

class Octaver:
    def __init__(self, samplerate):
        # Buffer to hold previous sample(s) for sub-octave generation
        self.state_l = np.array([0.0], dtype=np.float32)
        self.state_r = np.array([0.0], dtype=np.float32)

class Compressor:
    def __init__(self, samplerate, attack_ms, release_ms, threshold_db, ratio):
        self.env_follower = EnvelopeFollower(samplerate, attack_ms, release_ms)
        self.threshold = threshold_db # dB
        self.ratio_inv = 1.0 / ratio if ratio > 1.0 else 0.0

class Limiter:
    def __init__(self, samplerate, threshold_db, release_ms):
        # Limiter Release Alpha coefficient
        release_time_sec = release_ms * 0.001
        self.release_alpha = math.exp(-1.0 / (release_time_sec * samplerate)) if release_time_sec > 0 else 0.0
        self.threshold_amp = DB_TO_AMP(threshold_db)
        self.gain_state = np.array([1.0], dtype=np.float32) # Current gain reduction/makeup
        
# --- Utility Functions ---

# ... (get_current_params, set_current_params, list_presets, save_preset, load_preset, save_config, load_config remain the same) ...

def get_current_params():
    """Returns a dictionary containing the entire state of the application."""
    global SHARED_LENGTH, REAL_PICKUP_DISTS_FB, STRING_PARAMS_VIRTUAL, \
           SELECTED_PICKUP1, SELECTED_PICKUP2, IS_PICKUP2_ENABLED, \
           PICKUP1_VOL, PICKUP2_VOL, MASTER_TONE, EFFECTS_BYPASS, \
           ENV_FILTER_FREQ, ENV_FILTER_Q, ENV_FILTER_MIX, \
           OCTAVER_OCTAVE_VOL, OCTAVER_DRY_VOL, COMP_THRESHOLD, COMP_RATIO, \
           COMP_ATTACK_MS, COMP_RELEASE_MS, LIMITER_THRESHOLD

    return {
        'physical': {
            'length': SHARED_LENGTH,
            'dists_fb': REAL_PICKUP_DISTS_FB 
        },
        'virtual': {
            'string_params': STRING_PARAMS_VIRTUAL,
            'pickup1_model': SELECTED_PICKUP1,
            'pickup2_model': SELECTED_PICKUP2,
            'enable_p2': IS_PICKUP2_ENABLED,
            'pickup1_vol': PICKUP1_VOL,
            'pickup2_vol': PICKUP2_VOL,
            'master_tone': MASTER_TONE
        },
        'effects': {
            'bypass': EFFECTS_BYPASS,
            'env_freq': ENV_FILTER_FREQ,
            'env_q': ENV_FILTER_Q,
            'env_mix': ENV_FILTER_MIX,
            'oct_octave_vol': OCTAVER_OCTAVE_VOL,
            'oct_dry_vol': OCTAVER_DRY_VOL,
            'comp_threshold': COMP_THRESHOLD,
            'comp_ratio': COMP_RATIO,
            'comp_attack': COMP_ATTACK_MS,
            'comp_release': COMP_RELEASE_MS,
            'limiter_threshold': LIMITER_THRESHOLD
        }
    }

def set_current_params(data):
    """Sets global variables and updates DSP instances from loaded data."""
    global SHARED_LENGTH, REAL_PICKUP_DISTS_FB, STRING_PARAMS_VIRTUAL, \
           SELECTED_PICKUP1, SELECTED_PICKUP2, IS_PICKUP2_ENABLED, \
           PICKUP1_VOL, PICKUP2_VOL, MASTER_TONE, EFFECTS_BYPASS, \
           ENV_FILTER_FREQ, ENV_FILTER_Q, ENV_FILTER_MIX, \
           OCTAVER_OCTAVE_VOL, OCTAVER_DRY_VOL, COMP_THRESHOLD, COMP_RATIO, \
           COMP_ATTACK_MS, COMP_RELEASE_MS, LIMITER_THRESHOLD
    
    with params_lock:
        # Load Physical
        SHARED_LENGTH = data['physical']['length']
        REAL_PICKUP_DISTS_FB = data['physical']['dists_fb']

        # Load Virtual
        STRING_PARAMS_VIRTUAL = data['virtual']['string_params']
        SELECTED_PICKUP1 = data['virtual']['pickup1_model']
        SELECTED_PICKUP2 = data['virtual']['pickup2_model']
        IS_PICKUP2_ENABLED = data['virtual']['enable_p2']
        PICKUP1_VOL = data['virtual']['pickup1_vol']
        PICKUP2_VOL = data['virtual']['pickup2_vol']
        MASTER_TONE = data['virtual']['master_tone']

        # Load Effects
        EFFECTS_BYPASS = data['effects']['bypass']
        ENV_FILTER_FREQ = data['effects']['env_freq']
        ENV_FILTER_Q = data['effects']['env_q']
        ENV_FILTER_MIX = data['effects']['env_mix']
        OCTAVER_OCTAVE_VOL = data['effects']['oct_octave_vol']
        OCTAVER_DRY_VOL = data['effects']['oct_dry_vol']
        COMP_THRESHOLD = data['effects']['comp_threshold']
        COMP_RATIO = data['effects']['comp_ratio']
        COMP_ATTACK_MS = data['effects']['comp_attack']
        COMP_RELEASE_MS = data['effects']['comp_release']
        LIMITER_THRESHOLD = data['effects']['limiter_threshold']

        # CRITICAL: Re-initialize all DSP objects to reflect new values
        global client
        initialize_filters(client.samplerate)
    
def list_presets():
    """Returns a list of available preset filenames."""
    return sorted([f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith('.json')])

# Persistence Functions
def save_preset(name):
    """Saves the current state to a JSON file in the presets directory."""
    data = get_current_params()
    filepath = os.path.join(PRESET_DIR, f'{name}.json')
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Preset saved: {name}")
        return True
    except Exception as e:
        print(f"Error saving preset: {e}")
        return False

def load_preset(name):
    """Loads a preset from a JSON file and updates global state."""
    filepath = os.path.join(PRESET_DIR, f'{name}.json')
    if not os.path.exists(filepath):
        print(f"Preset not found: {name}")
        return False
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        set_current_params(data)
        print(f"Preset loaded: {name}")
        return True
    except Exception as e:
        print(f"Error loading preset: {e}")
        return False

def save_config():
    with params_lock:
        config_data = {
            'REAL_PICKUP_DISTS_FB': REAL_PICKUP_DISTS_FB,
            'SHARED_LENGTH': SHARED_LENGTH,
            'PICKUP1_VOL': PICKUP1_VOL,
            'PICKUP2_VOL': PICKUP2_VOL,
            'MASTER_TONE': MASTER_TONE,
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)
    print(f"Configuration saved to {CONFIG_FILE}")

def load_config():
    global REAL_PICKUP_DISTS_FB, SHARED_LENGTH, PICKUP1_VOL, PICKUP2_VOL, MASTER_TONE
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config_data = json.load(f)
        with params_lock:
            if 'REAL_PICKUP_DISTS_FB' in config_data:
                REAL_PICKUP_DISTS_FB = config_data['REAL_PICKUP_DISTS_FB']
            if 'SHARED_LENGTH' in config_data:
                SHARED_LENGTH = config_data['SHARED_LENGTH']
            if 'PICKUP1_VOL' in config_data:
                PICKUP1_VOL = config_data['PICKUP1_VOL']
            if 'PICKUP2_VOL' in config_data:
                PICKUP2_VOL = config_data['PICKUP2_VOL']
            if 'MASTER_TONE' in config_data:
                MASTER_TONE = config_data['MASTER_TONE']
        print(f"Configuration loaded from {CONFIG_FILE}")
    else:
        print(f"No config file found, using defaults and saving...")
        save_config()

# --- Numba-Optimized DSP Kernel ---

# The core recursive DSP logic, compiled to highly efficient machine code.
@nb.njit(fastmath=True, cache=True)
def run_string_dsp_numba(
    frames, input_chunk, p1_output_chunk, p2_output_chunk, is_pickup2_enabled, one_third,
    # FCCF (Feedback Correction)
    fb_delay, fb_gain, fb_buffer, fb_index, 
    # Pickup 1 FFCFs (Feedforward)
    ff1_p1_delay, ff1_p1_gain, ff1_p1_buffer, ff1_p1_index,
    ff2_p1_delay, ff2_p1_gain, ff2_p1_buffer, ff2_p1_index,
    ff3_p1_delay, ff3_p1_gain, ff3_p1_buffer, ff3_p1_index,
    # Pickup 2 FFCFs (Feedforward)
    ff1_p2_delay, ff1_p2_gain, ff1_p2_buffer, ff1_p2_index,
    ff2_p2_delay, ff2_p2_gain, ff2_p2_buffer, ff2_p2_index,
    ff3_p2_delay, ff3_p2_gain, ff3_p2_buffer, ff3_p2_index
):
    """
    Numba-compiled function to process one string's worth of audio (6 filters) 
    sample-by-sample for a full block (frames).
    """
    
    # Pre-calculate index masks
    fb_mask = fb_delay - 1
    ff1_p1_mask = ff1_p1_delay - 1
    ff2_p1_mask = ff2_p1_delay - 1
    ff3_p1_mask = ff3_p1_delay - 1
    
    # Numba will optimize these branches
    if is_pickup2_enabled:
        ff1_p2_mask = ff1_p2_delay - 1
        ff2_p2_mask = ff2_p2_delay - 1
        ff3_p2_mask = ff3_p2_delay - 1

    # Copy initial index to local variable for loop
    # FIX: fb_index is now guaranteed to be an array (np.array([0], dtype=np.int32))
    fb_idx = fb_index[0]
    ff1_p1_idx = ff1_p1_index[0]
    ff2_p1_idx = ff2_p1_index[0]
    ff3_p1_idx = ff3_p1_index[0]
    
    if is_pickup2_enabled:
        ff1_p2_idx = ff1_p2_index[0]
        ff2_p2_idx = ff2_p2_index[0]
        ff3_p2_idx = ff3_p2_index[0]


    for n in range(frames):
        sample = input_chunk[n]
        
        # --- 1. Feedback Comb Filter (Correction Filter) ---
        delayed_sample_fb = fb_buffer[fb_idx]
        sample_corrected = sample + fb_gain * delayed_sample_fb
        fb_buffer[fb_idx] = sample_corrected
        fb_idx = (fb_idx + 1) % fb_delay 
        
        
        # --- 2. Pickup 1 Feedforward Filters ---
        delayed_sample_ff1_p1 = ff1_p1_buffer[ff1_p1_idx]
        p1_out_1 = sample_corrected + ff1_p1_gain * delayed_sample_ff1_p1
        ff1_p1_buffer[ff1_p1_idx] = sample_corrected
        ff1_p1_idx = (ff1_p1_idx + 1) % ff1_p1_delay

        delayed_sample_ff2_p1 = ff2_p1_buffer[ff2_p1_idx]
        p1_out_2 = sample_corrected + ff2_p1_gain * delayed_sample_ff2_p1
        ff2_p1_buffer[ff2_p1_idx] = sample_corrected
        ff2_p1_idx = (ff2_p1_idx + 1) % ff2_p1_delay

        delayed_sample_ff3_p1 = ff3_p1_buffer[ff3_p1_idx]
        p1_out_3 = sample_corrected + ff3_p1_gain * delayed_sample_ff3_p1
        ff3_p1_buffer[ff3_p1_idx] = sample_corrected
        ff3_p1_idx = (ff3_p1_idx + 1) % ff3_p1_delay

        # Write sum to output chunk
        p1_output_chunk[n] = (p1_out_1 + p1_out_2 + p1_out_3) * one_third


        # --- 3. Pickup 2 Feedforward Filters ---
        if is_pickup2_enabled:
            delayed_sample_ff1_p2 = ff1_p2_buffer[ff1_p2_idx]
            p2_out_1 = sample_corrected + ff1_p2_gain * delayed_sample_ff1_p2
            ff1_p2_buffer[ff1_p2_idx] = sample_corrected
            ff1_p2_idx = (ff1_p2_idx + 1) % ff1_p2_delay

            delayed_sample_ff2_p2 = ff2_p2_buffer[ff2_p2_idx]
            p2_out_2 = sample_corrected + ff2_p2_gain * delayed_sample_ff2_p2
            ff2_p2_buffer[ff2_p2_idx] = sample_corrected
            ff2_p2_idx = (ff2_p2_idx + 1) % ff2_p2_delay

            delayed_sample_ff3_p2 = ff3_p2_buffer[ff3_p2_idx]
            p2_out_3 = sample_corrected + ff3_p2_gain * delayed_sample_ff3_p2
            ff3_p2_buffer[ff3_p2_idx] = sample_corrected
            ff3_p2_idx = (ff3_p2_idx + 1) % ff3_p2_delay
            
            # Write sum to output chunk
            p2_output_chunk[n] = (p2_out_1 + p2_out_2 + p2_out_3) * one_third

    # Write final index state back to the mutable array/list
    fb_index[0] = fb_idx
    ff1_p1_index[0] = ff1_p1_idx
    ff2_p1_index[0] = ff2_p1_idx
    ff3_p1_index[0] = ff3_p1_idx
    
    if is_pickup2_enabled:
        ff1_p2_index[0] = ff1_p2_idx
        ff2_p2_index[0] = ff2_p2_idx
        ff3_p2_index[0] = ff3_p2_idx
        
    return 0 

@nb.njit(fastmath=True, cache=True)
def run_lpf_numba(input_array, a0, b1, z1_state):
    """Processes an entire block of audio for a single 1st-order Low-Pass Filter."""
    # Create a copy to store output (safer for Numba input/output model)
    output_array = input_array.copy() 
    z1 = z1_state[0] # Get state from 1-element array
    
    # Optimized Python loop for the IIR feedback
    for i in range(len(input_array)):
        output = a0 * input_array[i] + b1 * z1
        z1 = output
        output_array[i] = output
        
    z1_state[0] = z1 # Store the final state for the next block
    return output_array

@nb.njit(fastmath=True, cache=True)
def process_effects_numba(
    frames, 
    input_l, input_r, 
    # Bypass states
    env_filter_bypass, octaver_bypass, comp_bypass,
    
    # Envelope Filter Params (Example using SVF)
    env_g, env_R, env_state_l, env_state_r, env_follower_state, env_follower_alpha_atk, env_follower_alpha_rel, env_mix,
    
    # Octaver Params
    oct_dry_vol, oct_sub_vol, oct_state_l, oct_state_r, 
    
    # Compressor Params
    comp_env_alpha_atk, comp_env_alpha_rel, comp_threshold_amp, comp_ratio_inv, comp_makeup_gain, comp_env_state,

    # Limiter Params (Always active)
    lim_threshold, lim_release_alpha, lim_gain_state,
    
    comp_gr_max_state,   # New: Array for Compressor Max GR output
    lim_gr_max_state     # New: Array for Limiter Max GR output
):
    
    # Octaver: Simple one-sample delay buffer for generating the sub-octave.
    oct_buf_l = oct_state_l
    oct_buf_r = oct_state_r

    # Compressor/Limiter: Envelope state is a scalar in a 1-element array
    comp_envelope = comp_env_state[0]
    lim_gain = lim_gain_state[0]
    
    max_comp_gr_amp = 1.0 
    max_lim_gr_amp = 1.0
    
    for n in range(frames):
        sample_l = input_l[n]
        sample_r = input_r[n]
        
        output_l = sample_l
        output_r = sample_r
        
        # --- 1. Envelope Filter ---
        if not env_filter_bypass:
            # Envelope Follower: Use peak detector on mono signal (L+R)/2
            mono_sample = (output_l + output_r) * 0.5
            abs_sample = np.abs(mono_sample)
            
            if abs_sample > comp_envelope:
                comp_envelope = (comp_envelope * env_follower_alpha_atk) + (abs_sample * (1.0 - env_follower_alpha_atk))
            else:
                comp_envelope = (comp_envelope * env_follower_alpha_rel) + (abs_sample * (1.0 - env_follower_alpha_rel))
            
            # Use envelope (comp_envelope) to sweep filter frequency
            # SVF Process (Simplified Bandpass, replacing coefficients for demo)
            R_value = env_R 
            g = 0.05 + 0.1 * comp_envelope # Mock sweep: 0.05 to 0.15 for 'g'
            
            # L Channel
            v3_l = output_l - env_state_l[1] # v3 = input - z2 (HP output)
            v1_l = env_state_l[0] # v1 = Z1 (LP output)
            v2_l = env_state_l[1] # v2 = Z2 (BP output)
            # Recurrence equations
            env_state_l[0] = v1_l + g * v2_l # LP (Z1)
            v_bp_l = v2_l + g * v3_l
            env_state_l[1] = (v_bp_l - R_value * v2_l) / (1.0 + g * R_value) # BP (Z2)
            # Final output mix (LP for low pass or BP for band pass)
            filtered_sample_l = env_state_l[1] # Use Bandpass
            output_l = (filtered_sample_l * env_mix) + (output_l * (1.0 - env_mix))

            # R Channel
            v3_r = output_r - env_state_r[1] 
            v1_r = env_state_r[0] 
            v2_r = env_state_r[1] 
            env_state_r[0] = v1_r + g * v2_r
            v_bp_r = v2_r + g * v3_r
            env_state_r[1] = (v_bp_r - R_value * v2_r) / (1.0 + g * R_value) 
            filtered_sample_r = env_state_r[1] 
            output_r = (filtered_sample_r * env_mix) + (output_r * (1.0 - env_mix))

        # --- 2. Octaver (Sub-Octave Generator) ---
        if not octaver_bypass:
            # L Channel
            sub_l = oct_state_l[0]
            oct_state_l[0] = output_l # Store current sample for next cycle
            oct_out_l = (output_l * oct_dry_vol) + (sub_l * oct_sub_vol)
            output_l = oct_out_l
            
            # R Channel
            sub_r = oct_state_r[0]
            oct_state_r[0] = output_r
            oct_out_r = (output_r * oct_dry_vol) + (sub_r * oct_sub_vol)
            output_r = oct_out_r

        # --- 3. Tube-Style Compressor ---
        if not comp_bypass:
            # Envelope Follower: Use peak detector on mono signal
            mono_sample = (output_l + output_r) * 0.5
            abs_sample = np.abs(mono_sample)
            
            if abs_sample > comp_envelope:
                comp_envelope = (comp_envelope * comp_env_alpha_atk) + (abs_sample * (1.0 - comp_env_alpha_atk))
            else:
                comp_envelope = (comp_envelope * comp_env_alpha_rel) + (abs_sample * (1.0 - comp_env_alpha_rel))

            # Gain Calculation (Above threshold)
            gain = 1.0
            if comp_envelope > comp_threshold_amp:
                level_db = 20.0 * np.log10(comp_envelope + 1e-12) 
                threshold_db = 20.0 * np.log10(comp_threshold_amp + 1e-12)
                
                # GR_dB = (Threshold_dB - Level_dB) * (1/Ratio - 1)
                gain_db = (threshold_db - level_db) * (comp_ratio_inv - 1.0)
                gain = 10.0**(gain_db / 20.0)

            # TRACKING COMPRESSOR GAIN REDUCTION
            if gain < max_comp_gr_amp:
                max_comp_gr_amp = gain 
                
            output_l *= gain 
            output_r *= gain 
            # Apply fixed make-up gain (optional: comp_makeup_gain)

        # --- 4. Final Limiter (Always ON) ---
        # Peak Detection: Look at largest of L and R
        peak = np.abs(output_l) if np.abs(output_l) > np.abs(output_r) else np.abs(output_r)

        # Gain reduction calculation
        target_gain = 1.0
        if peak > lim_threshold:
            target_gain = lim_threshold / peak

        # TRACKING LIMITER GAIN REDUCTION
        if lim_gain < max_lim_gr_amp:
            max_lim_gr_amp = lim_gain 
        
        # Smooth the gain reduction (attack: instant, release: smoothed)
        if target_gain < lim_gain: # Attack (instant)
            lim_gain = target_gain
        else: # Release (smoothed)
            lim_gain = (lim_gain * lim_release_alpha) + (target_gain * (1.0 - lim_release_alpha))
            
        # Apply the current smoothed gain
        output_l *= lim_gain
        output_r *= lim_gain

        # Write processed sample back to input array
        input_l[n] = output_l
        input_r[n] = output_r

    # Store final envelope and gain states
    comp_env_state[0] = comp_envelope
    lim_gain_state[0] = lim_gain
    
    # Store max gain reduction for the meters (convert linear amplitude to dB)
    comp_gr_max_state[0] = 20.0 * np.log10(max_comp_gr_amp + 1e-12) # Value is 0.0 or negative
    lim_gr_max_state[0] = 20.0 * np.log10(max_lim_gr_amp + 1e-12) # Value is 0.0 or negative

    return 0
    
# --- DSP Initialization ---
def initialize_filters(samplerate):
    """Initializes all DSP objects based on current global parameters."""
    global fb_correction_filters, ff1_filters_p1, ff2_filters_p1, ff3_filters_p1, \
           ff1_filters_p2, ff2_filters_p2, ff3_filters_p2, master_tone_filters_l, master_tone_filters_r, \
           EFFECT_INSTANCES, client, SHARED_LENGTH
           
    print(f"Initializing DSP for samplerate: {samplerate}")
    
    # Calculate one-third constant for pickup summing
    global ONE_THIRD
    ONE_THIRD = 1.0 / 3.0

    # 1. Digital Waveguide Filters (Karplus-Strong)
    num_strings = len(STRING_PARAMS_VIRTUAL)
    fb_correction_filters.clear()
    
    # Pickup filter lists
    ff1_filters_p1.clear()
    ff2_filters_p1.clear()
    ff3_filters_p1.clear()
    ff1_filters_p2.clear()
    ff2_filters_p2.clear()
    ff3_filters_p2.clear()
    
    # NEW: Clear separate L/R Master Tone Filter lists
    master_tone_filters_l.clear()
    master_tone_filters_r.clear()

    # Calculate filters for each string
    for i in range(num_strings):
        params = STRING_PARAMS_VIRTUAL[i]
        
        # Karplus-Strong parameters
        delay_length_samples = int(round(samplerate / params['freq']))
        
        # Feedback Correction Filter (FCCF)
        # Use a gain factor slightly less than 1.0 for damping (decay)
        feedback_gain = 0.99995 ** (samplerate / delay_length_samples) # Damping formula
        fb_correction_filters.append(FeedbackCombFilter(delay_length_samples, feedback_gain))
        
        # Pickup Models (Feedforward Comb Filters - FFCF)
        # Pickup 1
        p1_model = PICKUP_MODELS[SELECTED_PICKUP1]
        
        # Pickup 1 is modeled as 3 parallel FFCFs for the 3 virtual bridge positions (ff1, ff2, fb)
        delay_ff1_p1 = int(round(params['dist_ff1'] / (SHARED_LENGTH * samplerate / 2.0)))
        delay_ff2_p1 = int(round(params['dist_ff2'] / (SHARED_LENGTH * samplerate / 2.0)))
        delay_ff3_p1 = int(round(REAL_PICKUP_DISTS_FB[i] / (SHARED_LENGTH * samplerate / 2.0)))
        
        ff1_filters_p1.append(FeedforwardCombFilter(max(1, delay_ff1_p1), 1.0))
        ff2_filters_p1.append(FeedforwardCombFilter(max(1, delay_ff2_p1), 1.0))
        ff3_filters_p1.append(FeedforwardCombFilter(max(1, delay_ff3_p1), 1.0))
        
        # Pickup 2 (if enabled)
        if IS_PICKUP2_ENABLED:
            p2_model = PICKUP_MODELS[SELECTED_PICKUP2]
            # Use same virtual distances as Pickup 1 for simplicity in virtual setup
            delay_ff1_p2 = int(round(params['dist_ff1'] / (SHARED_LENGTH * samplerate / 2.0)))
            delay_ff2_p2 = int(round(params['dist_ff2'] / (SHARED_LENGTH * samplerate / 2.0)))
            delay_ff3_p2 = int(round(REAL_PICKUP_DISTS_FB[i] / (SHARED_LENGTH * samplerate / 2.0)))
            
            ff1_filters_p2.append(FeedforwardCombFilter(max(1, delay_ff1_p2), 1.0))
            ff2_filters_p2.append(FeedforwardCombFilter(max(1, delay_ff2_p2), 1.0))
            ff3_filters_p2.append(FeedforwardCombFilter(max(1, delay_ff3_p2), 1.0))

        # 2. Master Tone Filter (Low-Pass)
        # NEW: Separate L/R instances for correct stereo state
        master_tone_filters_l.append(LowPassFilter(MASTER_TONE, samplerate))
        master_tone_filters_r.append(LowPassFilter(MASTER_TONE, samplerate))


    # 3. Effect Instances
    # Effects use the Compressor/Limiter Envelope Follower (which needs the full samplerate)
    EFFECT_INSTANCES['env_filter'] = EnvelopeFilter(samplerate, ENV_FILTER_FREQ, ENV_FILTER_Q, ENV_FILTER_MIX, COMP_ATTACK_MS, COMP_RELEASE_MS)
    EFFECT_INSTANCES['octaver'] = Octaver(samplerate)
    EFFECT_INSTANCES['comp'] = Compressor(samplerate, COMP_ATTACK_MS, COMP_RELEASE_MS, COMP_THRESHOLD, COMP_RATIO)
    EFFECT_INSTANCES['limiter'] = Limiter(samplerate, LIMITER_THRESHOLD, LIMITER_RELEASE_MS)

    # Numba JIT compilation hint
    # Pre-compile the most complex Numba function to avoid a lag spike on the first run.
    try:
        if fb_correction_filters:
            # Note: We must pass the 1-element index arrays, not the scalar '0'
            _ = run_string_dsp_numba(
                1, np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32), 
                IS_PICKUP2_ENABLED, ONE_THIRD,
                fb_correction_filters[0].delay_length, fb_correction_filters[0].gain, fb_correction_filters[0].buffer, fb_correction_filters[0].buffer_index,
                ff1_filters_p1[0].delay_length, ff1_filters_p1[0].gain, ff1_filters_p1[0].buffer, ff1_filters_p1[0].buffer_index,
                ff2_filters_p1[0].delay_length, ff2_filters_p1[0].gain, ff2_filters_p1[0].buffer, ff2_filters_p1[0].buffer_index,
                ff3_filters_p1[0].delay_length, ff3_filters_p1[0].gain, ff3_filters_p1[0].buffer, ff3_filters_p1[0].buffer_index,
                1, 1.0, np.array([0.0], dtype=np.float32), np.array([0], dtype=np.int32),
                1, 1.0, np.array([0.0], dtype=np.float32), np.array([0], dtype=np.int32),
                1, 1.0, np.array([0.0], dtype=np.float32), np.array([0], dtype=np.int32)
            )
            
            # New JIT hint for LPF
            lpf_l = master_tone_filters_l[0]
            _ = run_lpf_numba(np.array([0.0], dtype=np.float32), lpf_l.a0, lpf_l.b1, lpf_l.z1_state)
        
        # Pre-compile effects DSP
        lim = EFFECT_INSTANCES['limiter']
        comp = EFFECT_INSTANCES['comp']
        env = EFFECT_INSTANCES['env_filter']
        oct = EFFECT_INSTANCES['octaver']
        
        comp_makeup_gain_linear = DB_TO_AMP(0.0)
        
        _ = process_effects_numba(
            1, np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32),
            EFFECTS_BYPASS['env_filter'], EFFECTS_BYPASS['octaver'], EFFECTS_BYPASS['comp'],
            env.svf_l.g, env.svf_l.R, env.svf_l.state, env.svf_r.state, env.env_follower.envelope_state, env.env_follower.alpha_atk, env.env_follower.alpha_rel, env.mix,
            OCTAVER_DRY_VOL, OCTAVER_OCTAVE_VOL, oct.state_l, oct.state_r,
            comp.env_follower.alpha_atk, comp.env_follower.alpha_rel, DB_TO_AMP(comp.threshold), comp.ratio_inv, comp_makeup_gain_linear, comp.env_follower.envelope_state,
            lim.threshold_amp, lim.release_alpha, lim.gain_state,
            np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32) # Dummy GR arrays
        )
    except Exception as e:
        print(f"Numba JIT compilation hint failed: {e}")
        
    print("DSP Initialization complete.")

# --- JACK Audio Callback (Real-Time Thread) ---
def process_audio_callback(frames, inports, outports):
    """
    JACK audio callback function. This runs in the real-time thread.
    Input ports are the triggers (one per string).
    Output ports are the stereo output (L/R).
    """
    global LAST_PEAK_DB, LAST_COMP_GR_DB, LAST_LIMITER_GR_DB

    # Get buffer chunks from ports
    in_buffers = [p.get_array() for p in inports]
    out_l = outports[0].get_array()
    out_r = outports[1].get_array()

    # Clear output buffers
    out_l[:] = 0.0
    out_r[:] = 0.0

    # Temporary stereo output buffers for pickup summing
    p1_out_chunk = np.zeros(frames, dtype=np.float32)
    p2_out_chunk = np.zeros(frames, dtype=np.float32)
    
    # Check if filters are initialized
    if not fb_correction_filters:
        # FIX: Return 0 instead of jack.PORT_OK
        return 0 

    try:
        with params_lock:
            # --- 1. Digital Waveguide Synthesis (String Simulation) ---
            num_strings = len(in_buffers)
            
            for i in range(num_strings):
                fb = fb_correction_filters[i]
                p1_ff1, p1_ff2, p1_ff3 = ff1_filters_p1[i], ff2_filters_p1[i], ff3_filters_p1[i]
                
                # DSP for string i (Numba optimized)
                # We are passing the 1-element index array (e.g., fb.buffer_index)
                run_string_dsp_numba(
                    frames, in_buffers[i], p1_out_chunk, p2_out_chunk, IS_PICKUP2_ENABLED, ONE_THIRD,
                    fb.delay_length, fb.gain, fb.buffer, fb.buffer_index, 
                    p1_ff1.delay_length, p1_ff1.gain, p1_ff1.buffer, p1_ff1.buffer_index,
                    p1_ff2.delay_length, p1_ff2.gain, p1_ff2.buffer, p1_ff2.buffer_index,
                    p1_ff3.delay_length, p1_ff3.gain, p1_ff3.buffer, p1_ff3.buffer_index,
                    # Pickup 2 (if enabled)
                    ff1_filters_p2[i].delay_length if IS_PICKUP2_ENABLED else 1, ff1_filters_p2[i].gain if IS_PICKUP2_ENABLED else 1.0, ff1_filters_p2[i].buffer if IS_PICKUP2_ENABLED else np.array([0.0], dtype=np.float32), ff1_filters_p2[i].buffer_index if IS_PICKUP2_ENABLED else np.array([0], dtype=np.int32),
                    ff2_filters_p2[i].delay_length if IS_PICKUP2_ENABLED else 1, ff2_filters_p2[i].gain if IS_PICKUP2_ENABLED else 1.0, ff2_filters_p2[i].buffer if IS_PICKUP2_ENABLED else np.array([0.0], dtype=np.float32), ff2_filters_p2[i].buffer_index if IS_PICKUP2_ENABLED else np.array([0], dtype=np.int32),
                    ff3_filters_p2[i].delay_length if IS_PICKUP2_ENABLED else 1, ff3_filters_p2[i].gain if IS_PICKUP2_ENABLED else 1.0, ff3_filters_p2[i].buffer if IS_PICKUP2_ENABLED else np.array([0.0], dtype=np.float32), ff3_filters_p2[i].buffer_index if IS_PICKUP2_ENABLED else np.array([0], dtype=np.int32)
                )

                # --- 2. Pickup and Tone Summing ---
                # P1 + P2 mix (simulated stereo L/R)
                string_out_l = (p1_out_chunk * PICKUP1_VOL) + (p2_out_chunk * PICKUP2_VOL)
                string_out_r = (p1_out_chunk * PICKUP1_VOL) + (p2_out_chunk * PICKUP2_VOL)
                
                # Master Tone Filter (Low-Pass)
                lpf_l = master_tone_filters_l[i]
                lpf_r = master_tone_filters_r[i]

                # NEW: Call the top-level Numba function, passing only compatible data
                string_out_l = run_lpf_numba(string_out_l, lpf_l.a0, lpf_l.b1, lpf_l.z1_state)
                string_out_r = run_lpf_numba(string_out_r, lpf_r.a0, lpf_r.b1, lpf_r.z1_state)


                # Accumulate to master output
                out_l += string_out_l
                out_r += string_out_r
        
            # --- 3. Global Effects Processing ---
            # Stereo effects chain (L/R)
            env = EFFECT_INSTANCES['env_filter']
            oct = EFFECT_INSTANCES['octaver']
            comp = EFFECT_INSTANCES['comp']
            lim = EFFECT_INSTANCES['limiter']
            
            # Temporary arrays for max GR output (passed to Numba)
            comp_gr_max_arr = np.array([0.0], dtype=np.float32)
            lim_gr_max_arr = np.array([0.0], dtype=np.float32)
            
            comp_makeup_gain_linear = DB_TO_AMP(0.0) # Assumes 0dB makeup gain for now
            
            process_effects_numba(
                frames, out_l, out_r,
                EFFECTS_BYPASS['env_filter'], EFFECTS_BYPASS['octaver'], EFFECTS_BYPASS['comp'],
                env.svf_l.g, env.svf_l.R, env.svf_l.state, env.svf_r.state, env.env_follower.envelope_state, env.env_follower.alpha_atk, env.env_follower.alpha_rel, env.mix,
                OCTAVER_DRY_VOL, OCTAVER_OCTAVE_VOL, oct.state_l, oct.state_r,
                comp.env_follower.alpha_atk, comp.env_follower.alpha_rel, DB_TO_AMP(comp.threshold), comp.ratio_inv, comp_makeup_gain_linear, comp.env_follower.envelope_state,
                lim.threshold_amp, lim.release_alpha, lim.gain_state,
                comp_gr_max_arr, lim_gr_max_arr
            )
            
            # --- 4. Metering ---
            # Max peak level in the block (L+R combined)
            max_peak = np.max(np.abs(np.concatenate((out_l, out_r))))
            LAST_PEAK_DB = AMP_TO_DB(max_peak) if max_peak > 0 else -120.0
            
            # Max Gain Reduction (Negative dB values)
            LAST_COMP_GR_DB = comp_gr_max_arr[0]
            LAST_LIMITER_GR_DB = lim_gr_max_arr[0]
            
    except Exception as e:
        # Avoid crashing the real-time thread, but log the error outside of it
        print(f"Error in JACK callback (real-time threat) : {e}", file=sys.stderr)
        
    # FIX: Return 0 instead of jack.PORT_OK
    return 0

# --- Level Meter Thread (Non-Real-Time) ---
def start_level_meter_thread():
    """Starts a separate thread to send meter data to the web client via SocketIO."""
    def run_meter():
        # global variables are declared correctly
        global LAST_PEAK_DB, LAST_COMP_GR_DB, params_lock
        # Ensure 'comp_gr_db' and 'peak_db' are defined before the emit call
        # Initialize with a safe value
        comp_gr_db = LAST_COMP_GR_DB
        peak_db = LAST_PEAK_DB

        while True:
            try:
                # <--- START OF INDENTED BLOCK (The fix!)
                socketio.sleep(1/30) # Update 30 times per second
                
                # Safely read the latest values
                with params_lock:
                    peak_db = LAST_PEAK_DB
                    comp_gr_db = LAST_COMP_GR_DB

                    # Emit peak level and GR data to all connected web clients
                    # NOTE: Removed the loose lines that weren't inside the dict or a call
                    socketio.emit('meter_data', {
                        # Using your fixed variables from the previous step
                        'level_in': float(comp_gr_db),      # Using Gain Reduction (GR)
                        'level_out': float(peak_db)         # Using the final Output Peak Level
                    })
                
                # Use time.sleep if you are running in a dedicated threading.Thread
                # If using socketio.sleep, you don't need time.sleep here.
                # Since you are using socketio.sleep and this is a threading.Thread,
                # let's rely on socketio.sleep for SocketIO-safe pausing.
                
            # <--- END OF INDENTED BLOCK
            except Exception as e:
                print(f"Error in run_meter thread: {e}")
                socketio.sleep(1) # Use socketio.sleep here as well

    meter_thread = threading.Thread(target=run_meter, daemon=True)
    meter_thread.start()
    
# =========================================================================
# WEB UI CODE (Flask Routes and SocketIO)
# =========================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_dev')
# FIX: Use threading for compatibility with JACK client.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading') 

HTML_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Jack Bass Simulator</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&display=swap" rel="stylesheet">
    <style>
/* 🚀 USER REQUEST: Responsive Design (Mobile-First) & Font Resize */
:root {
    --bg-dark: #121212;
    --bg-light: #1e1e1e;
    --text-primary: #f0f0f0;
    --accent-color: #AA77CC;
    --border-dark: #555;
    --meter-peak: #FF4500;
    --meter-level: #00FF00;
    --meter-gr: #FFD700;
    
    /* Responsive Font Sizing */
    font-size: 32px; /* Base font size */
}

/* Enforce Orbitron font globally */
body, 
input, 
button, 
select, 
textarea, 
table, 
th, 
td,
.meter-container {
    font-family: 'Orbitron', monospace, sans-serif;
    color: var(--text-primary);
}

body {
    margin: 0; 
    padding: 10px; /* Reduced padding for mobile */
    background-color: var(--bg-dark);
    line-height: 1.4;
}

/* Base Styles (Optimized for both small and large screens) */
form {
    padding: 0;
}

form > * {
    margin-bottom: 15px; /* Slightly reduced margin */
    padding: 10px;
    border: 1px solid var(--border-dark);
    background-color: var(--bg-light);
    border-radius: 4px; 
    box-shadow: 0 0 5px rgba(0, 255, 255, 0.1);
}

fieldset {
    border: 2px solid var(--accent-color);
    padding: 10px; /* Reduced padding */
    margin-top: 15px;
    border-radius: 4px;
    background-color: var(--bg-light);
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
}

legend {
    font-size: 1.1rem; /* Slightly larger legend */
    font-weight: bold;
    padding: 0 10px;
    color: var(--accent-color);
    background-color: var(--bg-dark);
}

h2 {
    font-size: 1.5rem;
    border-bottom: 2px solid var(--accent-color);
    padding-bottom: 5px;
    margin-top: 20px;
    color: var(--accent-color);
}

label {
    display: block;
    margin-top: 5px;
    font-weight: 400; 
    font-size: 0.9rem; /* Slightly smaller label text */
}

/* Input Styling */
input[type="range"] {
    width: 100%; /* Full width for range on mobile */
    -webkit-appearance: none;
    height: 20px;
    background: #333;
    border-radius: 10px;
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 40px; /* Bigger touch target for mobile */
    height: 40px;
    background: var(--accent-color);
    cursor: pointer;
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent-color);
}
input[type="number"], 
input[type="text"] {
    width: auto; 
    max-width: 50px; /* Keep constraint for narrow number/text fields */
    padding: 8px;
    border: 1px solid var(--border-dark);
    border-radius: 2px;
    background-color: #000; 
    font-size: 0.9rem;
}
select {
    width: 100%; /* Full width for dropdowns on all screen sizes */
    padding: 8px;
    border: 1px solid var(--border-dark);
    border-radius: 2px;
    background-color: #000; 
    font-size: 1rem;
}
button {
    width: 100%; /* Full width button on mobile */
    max-width: 400px; /* Max width for desktop/tablet */
    margin-top: 10px;
    background-color: var(--accent-color);
    color: var(--bg-dark);
    border: none;
    padding: 10px;
    border-radius: 2px;
    cursor: pointer;
    font-weight: bold;
    box-shadow: 0 0 5px var(--accent-color);
    transition: background-color 0.1s;
}

table {
    width: 20%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
th, td {
    width: auto;
    border: 1px solid var(--border-dark);
    padding: 6px; /* Reduced padding */
    text-align: left;
}

/* Meter Styles */
.meter-container {
    gap: 5px; /* Reduced gap */
    margin-top: 5px;
}
.meter {
    height: 15px; /* Shorter meter */
    border: 1px solid var(--accent-color);
    border-radius: 2px;
    box-shadow: inset 0 0 5px var(--accent-color);
}

/* --------------------------------------------------------------------- */
/* Desktop/Tablet Optimization (Min-width 600px) */
/* This media query ensures a better side-by-side layout on larger screens */
/* --------------------------------------------------------------------- */
@media (min-width: 600px) {
    
    :root {
        font-size: 32px; 
    }
    
    body {
        padding: 32px; 
    }

    input[type="number"], 
    input[type="text"] {
        width: auto; /* Compact width for numeric inputs */
        max-width: 250px;
    }

    select {
        width: auto; /* Wider width for dropdowns to show full text */
        max-width: none;
    }

    button {
        width: auto; 
        max-width: none;
    }
}
</style>
</head>
<body>
    <h1>🎸 Bass Simulator</h1>
    
    <div class="meter-container">
        <h2>📊 Real-Time Meters</h2>
        <div class="flex-row" style="gap: 50px;">
            <div class="flex-col" style="flex: 1;">
                <div class="meter-label">Output Peak (dB): <span id="peak_db_display">-120.0</span></div>
                <div class="meter"><div id="peak_level_bar" class="level-bar" style="width: 0%;"></div></div>
            </div>
            <div class="flex-col" style="flex: 1;">
                <div class="meter-label">Compressor GR (dB): <span id="comp_gr_display">0.0</span></div>
                <div class="meter"><div id="comp_gr_bar" class="gr-bar gr-bar-comp" style="width: 0%;"></div></div>
            </div>
            <div class="flex-col" style="flex: 1;">
                <div class="meter-label">Limiter GR (dB): <span id="limiter_gr_display">0.0</span></div>
                <div class="meter"><div id="limiter_gr_bar" class="gr-bar gr-bar-lim" style="width: 0%;"></div></div>
            </div>
        </div>
    </div>
    
    <form id="control_form">
    
        <details open>
            <summary><h2>Preset Manager</h2></summary>
            <fieldset>
                <legend>Load Preset</legend>
                <select id="preset_select_load">
                    {% for preset in saved_preset_names %} 
                        <option value="{{ preset }}">{{ preset }}</option>
                    {% endfor %}
                </select>
                <button type="button" onclick="loadPreset()">Load</button>
            </fieldset>

            <fieldset>
                <legend>Save Current State</legend>
                <input type="text" id="preset_name_input" placeholder="Enter preset name">
                <button type="button" onclick="savePreset()">Save</button>
            </fieldset>
        </details>

        <details open>
            <summary><h2>Virtual Pickup Presets</h2></summary>
            <fieldset>
                <legend>Select Preset</legend>
                <select name="preset_select" onchange="updatePreset(this.value)">
                    {% for key, preset in presets.items() %}
                        <option value="{{ key }}" {{ 'selected' if key == selected_preset else '' }}>
                            {{ preset.name }}
                        </option>
                    {% endfor %}
                </select>
            </fieldset>

            <fieldset>
                <legend>String Parameters (Virtual Distances)</legend>
                <table>
                    <thead>
                        <tr>
                            <th>String</th>
                            <th>Freq (Hz)</th>
                            <th>Bridge FF1 (mm)</th>
                            <th>Bridge FF2 (mm)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for i in range(strings_virtual|length) %}
                        <tr>
                            <td>{{ strings_virtual[i]['note'] }}</td>
                            <td>{{ strings_virtual[i]['freq'] }}</td>
                            <td><input type="number" step="0.1" name="dist_ff1_virtual_{{ i }}" value="{{ strings_virtual[i]['dist_ff1'] }}" onchange="updateVirtualStringParam(this, {{ i }}, 'dist_ff1')"></td>
                            <td><input type="number" step="0.1" name="dist_ff2_virtual_{{ i }}" value="{{ strings_virtual[i]['dist_ff2'] }}" onchange="updateVirtualStringParam(this, {{ i }}, 'dist_ff2')"></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </fieldset>
        </details>
        
        <details open>
            <summary><h2>Pickup & Tone Controls</h2></summary>
            <fieldset>
                <legend>Pickup Models</legend>
                <div class="flex-row">
                    <div class="flex-col">
                        <label>Pickup 1 Model:
                        <select name="pickup1_select" onchange="updatePickupModel('pickup1', this.value)">
                            {% for key, model in pickup_models.items() %}
                                <option value="{{ key }}" {{ 'selected' if key == selected_pickup1 else '' }}>
                                    {{ model.name }}
                                </option>
                            {% endfor %}
                        </select></label>
                    </div>
                    <div class="flex-col">
                        <label>Pickup 2 Model:
                        <select name="pickup2_select" onchange="updatePickupModel('pickup2', this.value)">
                            {% for key, model in pickup_models.items() %}
                                <option value="{{ key }}" {{ 'selected' if key == selected_pickup2 else '' }}>
                                    {{ model.name }}
                                </option>
                            {% endfor %}
                        </select></label>
                        <label class="checkbox-label">
                            <input type="checkbox" name="pickup2_enabled" id="pickup2_enabled" {{ 'checked' if pickup2_enabled else '' }} onchange="updatePickupEnable(this.checked)"> Enable Pickup 2
                        </label>
                    </div>
                </div>
            </fieldset>

            <fieldset>
                <legend>Volume & Tone</legend>
                <div class="flex-row">
                    <div class="flex-col">
                        <label for="pickup1_vol_slider">Pickup 1 Volume: <span id="pickup1_vol_display">{{ pickup1_vol | round(2) }}</span></label>
                        <input type="range" min="0" max="1" step="0.01" id="pickup1_vol_slider" name="pickup1_vol_slider" value="{{ pickup1_vol }}" oninput="updateMasterControls(this, 'pickup1_vol', 'pickup1_vol_display')">
                    </div>
                    <div class="flex-col">
                        <label for="pickup2_vol_slider">Pickup 2 Volume: <span id="pickup2_vol_display">{{ pickup2_vol | round(2) }}</span></label>
                        <input type="range" min="0" max="1" step="0.01" id="pickup2_vol_slider" name="pickup2_vol_slider" value="{{ pickup2_vol }}" oninput="updateMasterControls(this, 'pickup2_vol', 'pickup2_vol_display')">
                    </div>
                    <div class="flex-col">
                        <label for="master_tone_slider">Master Tone (Low-Pass Cutoff Hz): <span id="master_tone_display">{{ master_tone | round(0) }}</span></label>
                        <input type="range" min="500" max="12000" step="10" id="master_tone_slider" name="master_tone_slider" value="{{ master_tone }}" oninput="updateMasterControls(this, 'master_tone', 'master_tone_display')">
                    </div>
                </div>
            </fieldset>
        </details>
       
        <details open>
            <summary><h2>Global Effects Rack</h2></summary>
            <div class="effect-controls">
                
                <fieldset>
                    <legend>Envelope Filter 
                        <label class="checkbox-label"><input type="checkbox" id="env_filter_bypass" {{ 'checked' if effects_bypass.env_filter else '' }} onchange="toggleBypass('env_filter', this.checked)"> Bypass</label>
                    </legend>
                    <label for="env_filter_freq">Center Freq (Hz): <span id="env_filter_freq_display">{{ env_filter_freq | round(0) }}</span></label>
                    <input type="range" min="50" max="1000" step="1" id="env_filter_freq" name="env_filter_freq" value="{{ env_filter_freq }}" oninput="syncEnvFilterFreq(this)">
                    
                    <label for="env_filter_q">Q/Resonance: <span id="env_filter_q_display">{{ env_filter_q | round(1) }}</span></label>
                    <input type="range" min="0.5" max="10" step="0.1" id="env_filter_q" name="env_filter_q" value="{{ env_filter_q }}" oninput="syncEnvFilterQ(this)">
                    
                    <label for="env_filter_mix">Dry/Wet Mix: <span id="env_filter_mix_display">{{ env_filter_mix | round(2) }}</span></label>
                    <input type="range" min="0" max="1" step="0.01" id="env_filter_mix" name="env_filter_mix" value="{{ env_filter_mix }}" oninput="syncEnvFilterMix(this)">
                </fieldset>
                
                <fieldset>
                    <legend>Octaver (Sub-Octave)
                        <label class="checkbox-label"><input type="checkbox" id="octaver_bypass" {{ 'checked' if effects_bypass.octaver else '' }} onchange="toggleBypass('octaver', this.checked)"> Bypass</label>
                    </legend>
                    <label for="octaver_octave_vol">Sub-Octave Volume: <span id="octaver_octave_vol_display">{{ octaver_octave_vol | round(2) }}</span></label>
                    <input type="range" min="0" max="1" step="0.1" id="octaver_octave_vol" name="octaver_octave_vol" value="{{ octaver_octave_vol }}" oninput="syncOctaverOctaveVol(this)">
                    
                    <label for="octaver_dry_vol">Dry Signal Volume: <span id="octaver_dry_vol_display">{{ octaver_dry_vol | round(2) }}</span></label>
                    <input type="range" min="0" max="1" step="0.1" id="octaver_dry_vol" name="octaver_dry_vol" value="{{ octaver_dry_vol }}" oninput="syncOctaverDryVol(this)">
                </fieldset>

                <fieldset>
                    <legend>Compressor/Limiter 
                        <label class="checkbox-label"><input type="checkbox" id="comp_bypass" {{ 'checked' if effects_bypass.comp else '' }} onchange="toggleBypass('comp', this.checked)"> Bypass</label>
                    </legend>
                    
                    <div class="flex-row" style="justify-content: space-between;">
                        <label for="comp_threshold_slider">Threshold (dB): <span id="comp_threshold_display">{{ comp_threshold | round(1) }}</span></label>
                        <input type="number" min="-60" max="0" step="0.1" id="comp_threshold_slider" name="comp_threshold_num" value="{{ comp_threshold }}" onchange="syncCompThreshold(this)">
                    </div>
                    <input type="range" min="0" max="100" name="comp_threshold_slider" value="{{ comp_threshold | float | abs }}" oninput="syncCompThreshold(this)"> 
                    
                    <label for="comp_ratio">Ratio: <span id="comp_ratio_display">{{ comp_ratio | round(1) }}</span></label>
                    <input type="range" min="1.0" max="20.0" step="0.1" id="comp_ratio" name="comp_ratio" value="{{ comp_ratio }}" oninput="syncCompRatio(this)">

                    <label for="comp_attack">Attack (ms): <span id="comp_attack_display">{{ comp_attack | round(0) }}</span></label>
                    <input type="range" min="1" max="500" step="1" id="comp_attack" name="comp_attack" value="{{ comp_attack }}" oninput="syncCompAttack(this)">

                    <label for="comp_release">Release (ms): <span id="comp_release_display">{{ comp_release | round(0) }}</span></label>
                    <input type="range" min="1" max="2000" step="1" id="comp_release" name="comp_release" value="{{ comp_release }}" oninput="syncCompRelease(this)">
                    
                    <hr>
                    <div class="flex-row" style="justify-content: space-between;">
                        <label for="limiter_threshold_slider">Limiter Threshold (dB): <span id="limiter_threshold_display">{{ limiter_threshold | round(1) }}</span></label>
                        <input type="number" min="-10" max="0" step="0.1" id="limiter_threshold_slider" name="limiter_threshold_num" value="{{ limiter_threshold }}" onchange="syncLimiterThreshold(this)">
                    </div>
                    <input type="range" min="0" max="20" name="limiter_threshold_slider" value="{{ limiter_threshold | float | abs * 10 }}" oninput="syncLimiterThreshold(this)"> 
                    
                </fieldset>
            </div>
        </details>

        <details>
            <summary><h2>Physical Parameters (Global)</h2></summary>
            <fieldset>
                <legend>String and Pickup Physics</legend>
                <div class="flex-row">
                    <div class="flex-col">
                        <label for="shared_length">Shared String Length (mm):</label>
                        <input type="number" step="0.1" name="shared_length" id="shared_length" value="{{ shared_length }}" onchange="updatePhysicalParams()">
                    </div>
                    <div class="flex-col">
                        <p>This section is for manual tuning of physical parameters (for advanced tuning).</p>
                    </div>
                </div>
                
                <table style="width: 50%; margin-top: 15px;">
                    <thead>
                        <tr>
                            <th>String</th>
                            <th>Real Pickup Distance (mm from Bridge)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for i in range(real_dists_fb|length) %}
                        <tr>
                            <td>{{ strings_virtual[i]['note'] }}</td>
                            <td><input type="number" step="0.1" name="dist_fb_real_{{ i }}" value="{{ real_dists_fb[i] }}" onchange="updatePhysicalParams()"></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </fieldset>
        </details>


    </form>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script>
        const socket = io();

        // --- Utility Functions ---

        function mapRange(value, inMin, inMax, outMin, outMax) {
            return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
        }

        function dbToPercent(db) {
            // Map -120dB to -0dB to 0% to 100%
            const maxDb = 0;
            const minDb = -60;
            if (db >= maxDb) return 100;
            if (db <= minDb) return 0;
            
            // Map -60 to 0 linearly to 0 to 100
            return (db - minDb) / (maxDb - minDb) * 100;
        }
        
        // --- Real-Time Meter Update ---
        socket.on('meter_data', function(data) {
            // Peak Level Meter
            const peakDb = data.peak_db ?? -120.0;
            const peakPercent = dbToPercent(peakDb);
            document.getElementById('peak_db_display').textContent = peakDb.toFixed(1);
            document.getElementById('peak_level_bar').style.width = peakPercent + '%';

            // Compressor GR Meter
            // GR is negative (0 to -X). We want to display |GR| and map it to a width.
            const compGrDb = Math.max(-10, data.comp_gr_db ?? 0.0);
            const compGrPercent = Math.abs(compGrDb) / 10 * 100;
            document.getElementById('comp_gr_display').textContent = compGrDb.toFixed(1);
            document.getElementById('comp_gr_bar').style.width = compGrPercent + '%';
            
            // Limiter GR Meter
            const limiterGrDb = Math.max(-3, data.limiter_gr_db ?? 0.0);
            const limiterGrPercent = Math.abs(limiterGrDb) / 3 * 100;
            document.getElementById('limiter_gr_display').textContent = limiterGrDb.toFixed(1);
            document.getElementById('limiter_gr_bar').style.width = limiterGrPercent + '%';
        });

        // --- DSP Update Helper ---
        function sendEffectUpdate(paramName, value) {
            fetch('/api/update_effects', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: paramName, value: value })
            }).then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            }).then(data => {
                // console.log(data.status); // Success
            }).catch(error => {
                console.error("Error sending effect update:", error);
            });
        }
        
        // --- Generic Slider Synchronization ---
        function syncSlider(sourceInput, paramName, min, max, step, displayId, updateFunction) {
            let value = parseFloat(sourceInput.value);
            
            // Handle Compressor Threshold mapping (slider is 0-100, actual is -60 to 0)
            if (paramName === 'comp_threshold' || paramName === 'limiter_threshold') {
                if (sourceInput.type === 'range') {
                    // Map slider value (0 to 100) to actual dB value (-60 to 0) or (-10 to 0)
                    // Comp: Slider value is abs(dB) * (100/60). Inverse map.
                    if (paramName === 'comp_threshold') {
                        value = mapRange(100 - value, 0, 100, -60, 0);
                        document.querySelector(`input[name="comp_threshold_num"]`).value = value.toFixed(1);
                    } else { // Limiter Threshold (0-100 -> -10 to 0)
                        value = mapRange(100 - value, 0, 100, -10, 0);
                        document.querySelector(`input[name="limiter_threshold_num"]`).value = value.toFixed(1);
                    }
                } else { // Number input is the source
                    // Update range slider (absolute value mapping)
                    if (paramName === 'comp_threshold') {
                        const sliderValue = 100 - mapRange(value, -60, 0, 0, 100);
                        document.querySelector(`input[name="comp_threshold_slider"]`).value = sliderValue;
                    } else {
                        const sliderValue = 100 - mapRange(value, -10, 0, 0, 100);
                        document.querySelector(`input[name="limiter_threshold_slider"]`).value = sliderValue;
                    }
                }
            }
            
            // Round and update display
            const displayElement = document.getElementById(displayId);
            if (displayElement) {
                // Use a generic rounding based on step size or standard round for display
                const decimals = (step.toString().split('.')[1] || '').length;
                displayElement.textContent = value.toFixed(decimals > 0 ? decimals : 0);
            }

            // Send to server
            updateFunction(paramName, value);
        }

        // --- Preset Management ---
        function savePreset() {
            const name = document.getElementById('preset_name_input').value;
            if (!name) {
                alert("Please enter a name for the preset.");
                return;
            }
            
            fetch('/api/save_preset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            }).then(response => response.json()).then(data => {
                alert(data.message);
                // CRITICAL: Reload the page to fetch the updated list of presets from the server
                window.location.reload(); 
            }).catch(error => {
                console.error("Save error:", error);
                alert("Error saving preset.");
            });
        }
        
        function loadPreset() {
            const select = document.getElementById('preset_select_load');
            const name = select.value;
            if (!name) return;
            
            fetch('/api/load_preset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name })
            }).then(response => response.json()).then(data => {
                alert(data.message);
                // CRITICAL: Reload page to show all updated global values
                window.location.reload(); 
            }).catch(error => {
                console.error("Load error:", error);
                alert("Error loading preset.");
            });
        }
        
        // --- Effect Synchronization Functions (Use generic helper) ---
        
        function toggleBypass(effectName, isChecked) {
            sendEffectUpdate(`${effectName}_bypass`, isChecked);
        }

        function syncEnvFilterFreq(sourceInput) {
            syncSlider(sourceInput, 'env_filter_freq', 50, 1000, 1, 'env_filter_freq_display', sendEffectUpdate);
        }
        function syncEnvFilterQ(sourceInput) {
            syncSlider(sourceInput, 'env_filter_q', 0.5, 10, 0.1, 'env_filter_q_display', sendEffectUpdate);
        }
        function syncEnvFilterMix(sourceInput) {
            syncSlider(sourceInput, 'env_filter_mix', 0, 1, 0.01, 'env_filter_mix_display', sendEffectUpdate);
        }

        function syncOctaverOctaveVol(sourceInput) {
            syncSlider(sourceInput, 'oct_octave_vol', 0, 2, 0.01, 'octaver_octave_vol_display', sendEffectUpdate);
        }
        function syncOctaverDryVol(sourceInput) {
            syncSlider(sourceInput, 'oct_dry_vol', 0, 2, 0.01, 'octaver_dry_vol_display', sendEffectUpdate);
        }

        function syncCompThreshold(sourceInput) {
            // Note: Threshold uses both slider and number, mapped -60 to 0.
            syncSlider(sourceInput, 'comp_threshold', -60, 0, 0.1, 'comp_threshold_display', sendEffectUpdate);
        }
        function syncCompRatio(sourceInput) {
            syncSlider(sourceInput, 'comp_ratio', 1.0, 20.0, 0.1, 'comp_ratio_display', sendEffectUpdate);
        }
        function syncCompAttack(sourceInput) {
            syncSlider(sourceInput, 'comp_attack', 1, 500, 1, 'comp_attack_display', sendEffectUpdate);
        }
        function syncCompRelease(sourceInput) {
            syncSlider(sourceInput, 'comp_release', 1, 2000, 1, 'comp_release_display', sendEffectUpdate);
        }
        function syncLimiterThreshold(sourceInput) {
            // Note: Limiter Threshold uses both slider and number, mapped -10 to 0.
            syncSlider(sourceInput, 'limiter_threshold', -10, 0, 0.1, 'limiter_threshold_display', sendEffectUpdate);
        }
        
        // --- Pickup and String Parameter Updates ---
        function updateVirtualStringParam(sourceInput, index, paramKey) {
            const value = parseFloat(sourceInput.value);
            fetch('/api/update_virtual_string', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: index, key: paramKey, value: value })
            }).then(response => response.json()).then(data => {
                if(data.status === 'success') {
                    // Update successful. No reload needed unless we change an active preset.
                }
            }).catch(error => console.error("Error updating virtual string param:", error));
        }

        function updateMasterControls(sourceInput, paramName, displayId) {
            const value = parseFloat(sourceInput.value);
            document.getElementById(displayId).textContent = value.toFixed(paramName === 'master_tone' ? 0 : 2);

            fetch('/api/update_master_controls', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: paramName, value: value })
            }).then(response => response.json()).then(data => {
                // ...
            }).catch(error => console.error("Error updating master controls:", error));
        }
        
        function updatePreset(presetKey) {
            fetch('/api/set_preset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ preset: presetKey })
            }).then(response => response.json()).then(data => {
                if (data.status === 'success') {
                    // A full page reload is necessary to redraw the new values everywhere
                    window.location.reload(); 
                }
            }).catch(error => console.error("Error setting preset:", error));
        }

        function updatePickupModel(pickupName, modelKey) {
            fetch('/api/set_pickup_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: pickupName, model: modelKey })
            }).then(response => response.json()).then(data => {
                // No reload needed
            }).catch(error => console.error("Error setting pickup model:", error));
        }
        
        function updatePickupEnable(enabled) {
             fetch('/api/enable_pickup2', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: enabled })
            }).then(response => response.json()).then(data => {
                // No reload needed
            }).catch(error => console.error("Error setting pickup 2 enable:", error));
        }
        
        function updatePhysicalParams() {
            const length = parseFloat(document.getElementById('shared_length').value);
            const dists_fb = [];
            document.querySelectorAll('input[name^="dist_fb_real_"]').forEach(input => {
                dists_fb.push(parseFloat(input.value));
            });
            
            fetch('/api/update_physical_params', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ length: length, dists_fb: dists_fb })
            }).then(response => response.json()).then(data => {
                 if (data.status === 'success') {
                    // Flash success or reload if necessary
                 }
            }).catch(error => console.error("Error updating physical params:", error));
        }
        
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    """Renders the main control interface."""
    global SHARED_LENGTH, REAL_PICKUP_DISTS_FB, STRING_PARAMS_VIRTUAL, \
           SELECTED_PICKUP1, SELECTED_PICKUP2, IS_PICKUP2_ENABLED, \
           PICKUP1_VOL, PICKUP2_VOL, MASTER_TONE, SELECTED_PRESET, \
           EFFECTS_BYPASS, ENV_FILTER_FREQ, ENV_FILTER_Q, ENV_FILTER_MIX, \
           OCTAVER_OCTAVE_VOL, OCTAVER_DRY_VOL, COMP_THRESHOLD, COMP_RATIO, \
           COMP_ATTACK_MS, COMP_RELEASE_MS, LIMITER_THRESHOLD

    return render_template_string(HTML_FORM, 
        shared_length=SHARED_LENGTH,
        real_dists_fb=REAL_PICKUP_DISTS_FB,
        strings_virtual=STRING_PARAMS_VIRTUAL,
        pickup_models=PICKUP_MODELS,
        selected_pickup1=SELECTED_PICKUP1,
        selected_pickup2=SELECTED_PICKUP2,
        pickup2_enabled=IS_PICKUP2_ENABLED,
        pickup1_vol=PICKUP1_VOL,
        pickup2_vol=PICKUP2_VOL,
        master_tone=MASTER_TONE,
        presets=PRESETS,
        selected_preset=SELECTED_PRESET,
        saved_preset_names=list_presets(), # FIX: Distinct variable name
        effects_bypass=EFFECTS_BYPASS,
        env_filter_freq=ENV_FILTER_FREQ,
        env_filter_q=ENV_FILTER_Q,
        env_filter_mix=ENV_FILTER_MIX,
        octaver_octave_vol=OCTAVER_OCTAVE_VOL,
        octaver_dry_vol=OCTAVER_DRY_VOL,
        comp_threshold=COMP_THRESHOLD,
        comp_ratio=COMP_RATIO,
        comp_attack=COMP_ATTACK_MS,
        comp_release=COMP_RELEASE_MS,
        limiter_threshold=LIMITER_THRESHOLD
    )
    
# --- Flask API Routes for Instant Updates ---

@app.route('/api/update_effects', methods=['POST'])
def update_effects_route():
    """Handles real-time updates for the global effects rack."""
    data = request.json
    name = data.get('name')
    value = data.get('value')
    
    if name is None or value is None:
        return jsonify({'status': 'error', 'message': 'Missing data'}), 400

    with params_lock:
        global EFFECTS_BYPASS, ENV_FILTER_FREQ, ENV_FILTER_Q, ENV_FILTER_MIX, \
               OCTAVER_OCTAVE_VOL, OCTAVER_DRY_VOL, COMP_THRESHOLD, COMP_RATIO, \
               COMP_ATTACK_MS, COMP_RELEASE_MS, LIMITER_THRESHOLD
        
        try:
            # Bypass updates
            if name.endswith('_bypass'):
                effect = name.split('_')[0]
                EFFECTS_BYPASS[effect] = value
            
            # Env Filter updates
            elif name == 'env_filter_freq':
                ENV_FILTER_FREQ = float(value)
                eff_env = EFFECT_INSTANCES.get('env_filter')
                if eff_env:
                    eff_env.svf_l.update_coefficients(ENV_FILTER_FREQ, ENV_FILTER_Q)
                    eff_env.svf_r.update_coefficients(ENV_FILTER_FREQ, ENV_FILTER_Q)
            elif name == 'env_filter_q':
                ENV_FILTER_Q = float(value)
                eff_env = EFFECT_INSTANCES.get('env_filter')
                if eff_env:
                    eff_env.svf_l.update_coefficients(ENV_FILTER_FREQ, ENV_FILTER_Q)
                    eff_env.svf_r.update_coefficients(ENV_FILTER_FREQ, ENV_FILTER_Q)
            elif name == 'env_filter_mix':
                ENV_FILTER_MIX = float(value)
                eff_env = EFFECT_INSTANCES.get('env_filter')
                if eff_env:
                    eff_env.mix = ENV_FILTER_MIX
            
            # Octaver updates
            elif name == 'oct_octave_vol':
                OCTAVER_OCTAVE_VOL = float(value)
            elif name == 'oct_dry_vol':
                OCTAVER_DRY_VOL = float(value)
            
            # Compressor updates (CRITICAL SECTION: Triggers the fix)
            eff_comp = EFFECT_INSTANCES.get('comp')
            if name == 'comp_threshold':
                COMP_THRESHOLD = float(value)
                if eff_comp: eff_comp.threshold = COMP_THRESHOLD
            elif name == 'comp_ratio':
                COMP_RATIO = float(value)
                if eff_comp: eff_comp.ratio_inv = 1.0 / COMP_RATIO
            elif name == 'comp_attack':
                COMP_ATTACK_MS = float(value)
                if eff_comp: 
                    # FIX: Calls the newly added method to update DSP coefficients
                    eff_comp.env_follower.set_time_constants(COMP_ATTACK_MS, COMP_RELEASE_MS) 
                    eff_env = EFFECT_INSTANCES.get('env_filter')
                    eff_env.env_follower.set_time_constants(COMP_ATTACK_MS, COMP_RELEASE_MS)

            elif name == 'comp_release':
                COMP_RELEASE_MS = float(value)
                if eff_comp: 
                    # FIX: Calls the newly added method to update DSP coefficients
                    eff_comp.env_follower.set_time_constants(COMP_ATTACK_MS, COMP_RELEASE_MS)
                    eff_env = EFFECT_INSTANCES.get('env_filter')
                    eff_env.env_follower.set_time_constants(COMP_ATTACK_MS, COMP_RELEASE_MS)

            # Limiter updates
            elif name == 'limiter_threshold':
                LIMITER_THRESHOLD = float(value)
                eff_lim = EFFECT_INSTANCES.get('limiter')
                if eff_lim: eff_lim.threshold_amp = DB_TO_AMP(LIMITER_THRESHOLD)
            
            # If any DSP parameter changes, signal the real-time thread 
            # (though the thread reads global values protected by the lock)
            update_event.set()
            
            return jsonify({'status': 'success', 'name': name, 'value': value})
            
        except Exception as e:
            print(f"Error processing effect update for {name}: {e}", file=sys.stderr)
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/save_preset', methods=['POST'])
def save_preset_route():
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'status': 'error', 'message': 'Preset name required.'}), 400
    
    if save_preset(name):
        return jsonify({'status': 'success', 'message': f'Preset "{name}" saved.'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to save preset.'}), 500

@app.route('/api/load_preset', methods=['POST'])
def load_preset_route():
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'status': 'error', 'message': 'Preset name required.'}), 400
    
    if load_preset(name):
        # NOTE: A reload is required on the client side to fetch all new values
        return jsonify({'status': 'success', 'message': f'Preset "{name}" loaded. Refreshing page.'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to load preset.'}), 500
        
@app.route('/api/update_virtual_string', methods=['POST'])
def update_virtual_string_route():
    data = request.json
    index = int(data.get('index'))
    key = data.get('key')
    value = float(data.get('value'))
    
    if key not in ['dist_ff1', 'dist_ff2']:
        return jsonify({'status': 'error', 'message': 'Invalid key'}), 400

    with params_lock:
        try:
            STRING_PARAMS_VIRTUAL[index][key] = value
            # Re-initialize DSP to update filter delays
            initialize_filters(client.samplerate)
            
            return jsonify({'status': 'success', 'index': index, 'key': key, 'value': value})
        except IndexError:
            return jsonify({'status': 'error', 'message': 'Invalid string index'}), 400
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/update_master_controls', methods=['POST'])
def update_master_controls_route():
    data = request.json
    name = data.get('name')
    value = float(data.get('value'))

    with params_lock:
        try:
            global PICKUP1_VOL, PICKUP2_VOL, MASTER_TONE
            
            if name == 'pickup1_vol':
                PICKUP1_VOL = value
            elif name == 'pickup2_vol':
                PICKUP2_VOL = value
            elif name == 'master_tone':
                MASTER_TONE = value
                # Re-initialize LowPassFilters only
                # NEW: Iterate over L and R filters
                for lpf_l, lpf_r in zip(master_tone_filters_l, master_tone_filters_r):
                    lpf_l.set_cutoff(MASTER_TONE)
                    lpf_r.set_cutoff(MASTER_TONE)
            
            save_config() # Save non-preset state
            return jsonify({'status': 'success', 'name': name, 'value': value})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/set_pickup_model', methods=['POST'])
def set_pickup_model_route():
    data = request.json
    name = data.get('name')
    model = data.get('model')

    with params_lock:
        global SELECTED_PICKUP1, SELECTED_PICKUP2
        try:
            if name == 'pickup1':
                SELECTED_PICKUP1 = model
            elif name == 'pickup2':
                SELECTED_PICKUP2 = model
            
            # Re-initialize DSP to update pickup model parameters
            initialize_filters(client.samplerate)
            
            return jsonify({'status': 'success', 'name': name, 'model': model})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/enable_pickup2', methods=['POST'])
def enable_pickup2_route():
    data = request.json
    enabled = data.get('enabled')

    with params_lock:
        global IS_PICKUP2_ENABLED
        try:
            IS_PICKUP2_ENABLED = enabled
            # Re-initialize DSP if necessary (or rely on Numba check)
            initialize_filters(client.samplerate)
            
            return jsonify({'status': 'success', 'enabled': enabled})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/update_physical_params', methods=['POST'])
def update_physical_params_route():
    data = request.json
    length = float(data.get('length'))
    dists_fb = data.get('dists_fb')

    with params_lock:
        global SHARED_LENGTH, REAL_PICKUP_DISTS_FB
        try:
            SHARED_LENGTH = length
            REAL_PICKUP_DISTS_FB = dists_fb
            
            # Re-initialize DSP to update filter delays
            initialize_filters(client.samplerate)
            save_config()
            
            return jsonify({'status': 'success', 'length': length, 'dists_fb': dists_fb})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/set_preset', methods=['POST'])
def set_preset_route():
    data = request.json
    preset_key = data.get('preset')

    if preset_key not in PRESETS:
        return jsonify({'status': 'error', 'message': 'Invalid preset key'}), 400

    with params_lock:
        global SELECTED_PRESET, STRING_PARAMS_VIRTUAL, SELECTED_PICKUP1, SELECTED_PICKUP2, IS_PICKUP2_ENABLED
        try:
            preset = PRESETS[preset_key]
            
            SELECTED_PRESET = preset_key
            STRING_PARAMS_VIRTUAL = preset['strings']
            SELECTED_PICKUP1 = preset['pickup1']
            SELECTED_PICKUP2 = preset['pickup2']
            IS_PICKUP2_ENABLED = preset['pickup2_enabled']
            
            # Re-initialize all DSP objects
            initialize_filters(client.samplerate)

            # Client must reload to fetch all new values
            return jsonify({'status': 'success', 'message': f'Preset {preset_key} activated. Please refresh.'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

# =========================================================================
# JACK and Main Execution
# =========================================================================

# --- Main execution logic ---
if __name__ == "__main__":
    load_config()
    inports = [client.inports.register(f'input_{i+1}') for i in range(6)]
    outports = [client.outports.register(f'output_{i+1}') for i in range(2)]
    client.set_process_callback(lambda f: process_audio_callback(f, inports, outports))

    try:
        # This function is now correctly defined above
        initialize_filters(client.samplerate)
        with client:
            print("JACK client activated. Connect ports using QjackCtl.")
            print(f"Access the web UI at http://0.0.0.0:5000")
            
            # This function is now correctly defined above
            start_level_meter_thread()
            
            # FIX: Only run the SocketIO server, which handles Flask
            socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
            
    except jack.JackError as e:
        print(f"JACK error: {e}")
        print("Ensure JACK server is running (e.g., via QjackCtl) and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
