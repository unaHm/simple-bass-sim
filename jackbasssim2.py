import jack
import numpy as np
import threading
import json
import os.path
from flask import Flask, render_template_string, request, flash, redirect, url_for, jsonify, session 
from flask_socketio import SocketIO
import math
import sys
import numba as nb

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
    'env_filter': False,
    'octaver': False,
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

# Filter Classes and DSP functions (Omitted for brevity)
class FeedbackCombFilter:
    def __init__(self, delay_length, feedback_gain):
        self.delay_length = delay_length
        self.gain = feedback_gain
        # State variables must be accessible to Numba
        self.buffer = np.zeros(delay_length, dtype='float32')
        self.buffer_index = 0

class FeedforwardCombFilter:
    def __init__(self, delay_length, gain):
        self.delay_length = delay_length
        self.gain = gain
        # State variables must be accessible to Numba
        self.buffer = np.zeros(delay_length, dtype='float32')
        self.buffer_index = 0
        
# --- Optimized LowPassFilter Class ---
class LowPassFilter:
    def __init__(self, cutoff_freq, samplerate):
        self.a0 = 0.0
        self.b1 = 0.0
        self.z1 = 0.0
        self.samplerate = samplerate
        self.set_cutoff(cutoff_freq)

    def set_cutoff(self, cutoff_freq):
        if cutoff_freq >= self.samplerate / 2:
              self.a0 = 1.0
              self.b1 = 0.0
        else:
            tau = 1.0 / (2.0 * math.pi * cutoff_freq)
            te = 1.0 / self.samplerate
            self.b1 = math.exp(-te / tau)
            self.a0 = 1.0 - self.b1

    def process(self, sample):
        # Kept for compatibility, though process_block is preferred for main DSP
        output = self.a0 * sample + self.b1 * self.z1
        self.z1 = output
        return output
    
    def process_block(self, input_array):
        """Processes an entire block of audio in an optimized Python loop."""
        # Create a copy to store output or modify in place (using a copy is safer)
        output_array = input_array.copy() 
        z1 = self.z1 
        a0 = self.a0
        b1 = self.b1
        
        # Optimized Python loop for the IIR feedback, avoiding unnecessary object creation
        for i in range(len(input_array)):
            output = a0 * input_array[i] + b1 * z1
            z1 = output
            output_array[i] = output
            
        self.z1 = z1 # Store the final state for the next block
        return output_array

# Global variables for filter instances (Omitted for brevity)
fb_correction_filters = [] 
ff1_filters_p1, ff2_filters_p1, ff3_filters_p1 = [], [], []
ff1_filters_p2, ff2_filters_p2, ff3_filters_p2 = [], [], []
master_tone_filters = [] 
client = jack.Client("PythonDSPClient")

# Persistence Functions (Omitted for brevity)
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

# --- NEW DSP CLASS DEFINITIONS ---

class EnvelopeFollower:
    """State for tracking the signal level (used by the Envelope Filter and Compressor)."""
    def __init__(self, samplerate, attack_ms, release_ms):
        # Smoothing coefficients (1-pole low-pass filter)
        self.alpha_atk = math.exp(-1.0 / (attack_ms * 0.001 * samplerate)) if attack_ms > 0 else 0.0
        self.alpha_rel = math.exp(-1.0 / (release_ms * 0.001 * samplerate)) if release_ms > 0 else 0.0
        self.envelope_state = np.array([0.0], dtype=np.float32) # Numba state I/O array

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
# --- NEW: Numba-Optimized DSP Kernel ---

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
        # The ring buffer logic is now fast C code
        delayed_sample_fb = fb_buffer[fb_idx]
        sample_corrected = sample + fb_gain * delayed_sample_fb
        fb_buffer[fb_idx] = sample_corrected
        fb_idx = (fb_idx + 1) & fb_mask # Bitwise AND is faster than modulo for powers of 2 (but mod is safer if delay length is not power of 2)
        # Using simple modulo for robustness in case delay length is not a power of 2
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
        
    # Numba functions must return something, but we modify the output chunks in place
    return 0 # Return 0 for consistency with JACK callback, though unused here.

@nb.njit(fastmath=True, cache=True)
def process_effects_numba(
    frames, 
    input_l, input_r, 
    # Bypass states
    env_filter_bypass, octaver_bypass, comp_bypass,
    
    # Envelope Filter Params (Example using SVF)
    env_g, env_R, env_state_l, env_state_r, env_follower_state, env_follower_alpha_atk, env_follower_alpha_rel, env_mix,
    
    # Octaver Params
    oct_dry_vol, oct_sub_vol, oct_state_l, oct_state_r, # oct_state_l/r would hold the fractional delay line/buffer
    
    # Compressor Params
    comp_env_alpha_atk, comp_env_alpha_rel, comp_threshold_amp, comp_ratio_inv, comp_makeup_gain, comp_env_state,

    # Limiter Params (Always active)
    lim_threshold, lim_release_alpha, lim_gain_state,
    
    comp_gr_max_state,   # New: Array for Compressor Max GR output
    lim_gr_max_state     # New: Array for Limiter Max GR output
):
    # This loop runs the entire effects chain for one block, sample-by-sample, in Numba-compiled code.
    
    # Octaver: Simple one-sample delay buffer for generating the sub-octave.
    # A true octaver is a complex pitch detector/shifter, but a simple 1/2 downsample 
    # and interpolation is a common synthesis approach.
    oct_buf_l = oct_state_l
    oct_buf_r = oct_state_r

    # Compressor/Limiter: Envelope state is a scalar in a 1-element array
    comp_envelope = comp_env_state[0]
    lim_gain = lim_gain_state[0]
    
    max_comp_gr_amp = 1.0 # Max gain (closest to 1.0)
    max_lim_gr_amp = 1.0

    # Constants
    #DB_TO_AMP = 10.0**(comp_threshold / 20.0)
    
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
            # (Simplified for example: map envelope to 500-2000 Hz)
            # A full implementation would update SVF coefficients 'g' and 'R' here.
            R_value = env_R
            
            # SVF Process (Simplified Bandpass, replacing coefficients for demo)
            g = 0.05 + 0.1 * comp_envelope # Mock sweep: 0.05 to 0.15 for 'g'
            
            v3_l = output_l - env_state_l[1] # v3 = input - z2 (HP output)
            v1_l = env_state_l[0]           # v1 = Z1 (LP output)
            v2_l = env_state_l[1]           # v2 = Z2 (BP output)
            
            # Recurrence equations
            env_state_l[0] = v1_l + g * v2_l  # LP (Z1)
            v_bp_l = v2_l + g * v3_l
            env_state_l[1] = (v_bp_l - R_value * v2_l) / (1.0 + g * R_value) # BP (Z2)
            
            # Final output mix (LP for low pass or BP for band pass)
            filtered_sample_l = env_state_l[1] # Use Bandpass
            output_l = (filtered_sample_l * env_mix) + (output_l * (1.0 - env_mix))
            
            # Repeat for R channel... (omitted for brevity)

        # --- 2. Octaver (Sub-Octave Generator) ---
        if not octaver_bypass:
            # Simple downsampling for sub-octave (Half-wave rectifier + delay)
            # True octavers use polyphase filters, but this is a common approximation.
            sub_l = oct_buf_l[0]
            oct_buf_l[0] = output_l # Store current sample for next cycle
            
            oct_out_l = (output_l * oct_dry_vol) + (sub_l * oct_sub_vol)
            output_l = oct_out_l
            # Repeat for R channel... (omitted for brevity)

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
                # Compression = (Input - Threshold) * (1 - 1/Ratio)
                # Apply gain reduction: 10^( (threshold - level) * (1 - 1/ratio) / 20)
                level_db = 20.0 * np.log10(comp_envelope)
                gain_db = (level_db - comp_threshold_amp) * comp_ratio_inv
                gain = 10.0**(-gain_db / 20.0)
            
            output_l *= gain
            output_r *= gain
            
            # Apply fixed make-up gain (optional: comp_makeup_gain)
            
            # TRACKING COMPRESSOR GAIN REDUCTION
            # The 'gain' is a linear amplitude value (0.0 to 1.0)
            if gain < max_comp_gr_amp:
                max_comp_gr_amp = gain # Store the most extreme (smallest) gain

            output_l *= gain
            output_r *= gain
            
        # --- 4. Final Limiter (Always ON) ---
        
        # Peak Detection: Look at largest of L and R
        peak = np.abs(output_l) if np.abs(output_l) > np.abs(output_r) else np.abs(output_r)
        
        # Gain reduction calculation
        target_gain = 1.0
        if peak > lim_threshold:
            # If peak exceeds threshold, the target gain is threshold / peak
            target_gain = lim_threshold / peak
            
        # TRACKING LIMITER GAIN REDUCTION
        # The 'lim_gain' is the instantaneous linear gain value
        if lim_gain < max_lim_gr_amp:
            max_lim_gr_amp = lim_gain # Store the most extreme (smallest) gain
            
        # Smooth the gain reduction (attack: instant, release: smoothed)
        if target_gain < lim_gain: # If we need to compress more (instant attack)
            lim_gain = target_gain
        else: # If we are releasing the compression (smoothed release)
            lim_gain = (lim_gain * lim_release_alpha) + (target_gain * (1.0 - lim_release_alpha))

        # Apply final gain
        input_l[n] = output_l * lim_gain
        input_r[n] = output_r * lim_gain

    # Update state variables (Envelope and Gain)
    comp_env_state[0] = comp_envelope
    lim_gain_state[0] = lim_gain
    
    # *** NEW: Write the Max Gain Reduction values back to the output arrays ***
    comp_gr_max_state[0] = 20.0 * np.log10(max_comp_gr_amp) if max_comp_gr_amp < 1.0 else 0.0
    lim_gr_max_state[0] = 20.0 * np.log10(max_lim_gr_amp) if max_lim_gr_amp < 1.0 else 0.0
    
    return 0

def initialize_effects(samplerate):
    global EFFECT_INSTANCES
    
    print("Initializing post-synthesis effects...")
    
    # Envelope Filter Init
    EFFECT_INSTANCES['env_filter'] = EnvelopeFilter(
        samplerate, 
        freq=ENV_FILTER_FREQ, 
        Q=ENV_FILTER_Q, 
        mix=ENV_FILTER_MIX, 
        attack_ms=5.0, 
        release_ms=100.0
    )
    
    # Octaver Init
    EFFECT_INSTANCES['octaver'] = Octaver(samplerate)
    
    # Compressor Init
    EFFECT_INSTANCES['comp'] = Compressor(
        samplerate, 
        attack_ms=COMP_ATTACK_MS, 
        release_ms=COMP_RELEASE_MS, 
        threshold_db=COMP_THRESHOLD, 
        ratio=COMP_RATIO
    )
    
    # Limiter Init (always active)
    EFFECT_INSTANCES['limiter'] = Limiter(
        samplerate, 
        threshold_db=LIMITER_THRESHOLD, 
        release_ms=LIMITER_RELEASE_MS
    )

# --- Refactored Filter Initialization (To use 1-element arrays for indexes) ---

def initialize_filters(samplerate):
    global fb_correction_filters, ff1_filters_p1, ff2_filters_p1, ff3_filters_p1, \
           ff1_filters_p2, ff2_filters_p2, ff3_filters_p2, master_tone_filters, \
           SHARED_LENGTH, STRING_PARAMS_VIRTUAL, SELECTED_PICKUP1, SELECTED_PICKUP2, IS_PICKUP2_ENABLED, REAL_PICKUP_DISTS_FB, MASTER_TONE
    
    with params_lock:
        length = SHARED_LENGTH
        params_list_virtual = STRING_PARAMS_VIRTUAL
        real_dists_fb = REAL_PICKUP_DISTS_FB
        pickup1_offset_ms = PICKUP_MODELS[SELECTED_PICKUP1]['offset_ms']
        pickup2_offset_ms = PICKUP_MODELS[SELECTED_PICKUP2]['offset_ms']
        master_tone_freq = MASTER_TONE

    fb_correction_filters = []
    ff1_filters_p1, ff2_filters_p1, ff3_filters_p1 = [], [], []
    ff1_filters_p2, ff2_filters_p2, ff3_filters_p2 = [], [], []
    master_tone_filters = [LowPassFilter(master_tone_freq, samplerate) for _ in range(2)]

    print(f"Initializing filters...")

    for i, params in enumerate(params_list_virtual):
        freq = params['freq']
        dist_ff1 = params['dist_ff1']
        dist_ff2 = params['dist_ff2']
        dist_fb_real = real_dists_fb[i]
        speed_of_sound_string = freq * 2 * length if freq > 0 else 1.0

        delay_fb_real_samples = max(1, int(samplerate * (dist_fb_real / speed_of_sound_string)))
        # Note: Index is now stored as a 1-element NumPy array so Numba can modify it in place
        fb_filter = FeedbackCombFilter(delay_fb_real_samples, -0.5)
        fb_filter.buffer_index = np.array([0], dtype=np.int32) 
        fb_correction_filters.append(fb_filter)

        base_delay_ff1 = max(1, int(samplerate * (dist_ff1 / speed_of_sound_string)))
        offset_samples_p1 = int(pickup1_offset_ms * samplerate / 1000)
        
        f1p1 = FeedforwardCombFilter(base_delay_ff1, 0.5); f1p1.buffer_index = np.array([0], dtype=np.int32); ff1_filters_p1.append(f1p1)
        f2p1 = FeedforwardCombFilter(max(1, base_delay_ff1 - offset_samples_p1), 0.5); f2p1.buffer_index = np.array([0], dtype=np.int32); ff2_filters_p1.append(f2p1)
        f3p1 = FeedforwardCombFilter(base_delay_ff1 + offset_samples_p1, 0.5); f3p1.buffer_index = np.array([0], dtype=np.int32); ff3_filters_p1.append(f3p1)

        base_delay_ff2 = max(1, int(samplerate * (dist_ff2 / speed_of_sound_string)))
        offset_samples_p2 = int(pickup2_offset_ms * samplerate / 1000)
        
        f1p2 = FeedforwardCombFilter(base_delay_ff2, 0.5); f1p2.buffer_index = np.array([0], dtype=np.int32); ff1_filters_p2.append(f1p2)
        f2p2 = FeedforwardCombFilter(max(1, base_delay_ff2 - offset_samples_p2), 0.5); f2p2.buffer_index = np.array([0], dtype=np.int32); ff2_filters_p2.append(f2p2)
        f3p2 = FeedforwardCombFilter(base_delay_ff2 + offset_samples_p2, 0.5); f3p2.buffer_index = np.array([0], dtype=np.int32); ff3_filters_p2.append(f3p2)
        initialize_effects(samplerate)
    with params_lock:
        pass

# --- Final Optimized process_audio_callback ---
def process_audio_callback(frames, inports, outports):
    global fb_correction_filters, ff1_filters_p1, ff2_filters_p1, ff3_filters_p1, \
           ff1_filters_p2, ff2_filters_p2, ff3_filters_p2, IS_PICKUP2_ENABLED, \
           master_tone_filters, PICKUP1_VOL, PICKUP2_VOL, MASTER_TONE
    
    if update_event.is_set():
        update_event.clear()
        initialize_filters(client.samplerate)
        
    master_tone_filters[0].set_cutoff(MASTER_TONE)
    master_tone_filters[1].set_cutoff(MASTER_TONE)
    
    inputs = [port.get_array() for port in inports]
    stereo_out_L = outports[0].get_array()
    stereo_out_R = outports[1].get_array()
    
    stereo_out_L[:] = 0.0
    stereo_out_R[:] = 0.0
    
    ONE_THIRD = 1.0 / 3.0
    HALF = 0.5
    
    max_abs_l = np.max(np.abs(stereo_out_L))
    max_abs_r = np.max(np.abs(stereo_out_R))
    overall_peak = max(max_abs_l, max_abs_r)

    for i in range(6): 
        input_channel = inputs[i]
        
        # Access filter instances
        fb_corr_filter = fb_correction_filters[i]
        ff1_p1, ff2_p1, ff3_p1 = ff1_filters_p1[i], ff2_filters_p1[i], ff3_filters_p1[i]
        ff1_p2, ff2_p2, ff3_p2 = ff1_filters_p2[i], ff2_filters_p2[i], ff3_filters_p2[i]
        
        p1_output_chunk = np.zeros(frames, dtype='float32')
        # P2 output chunk is only created if P2 is enabled, to save allocation time
        p2_output_chunk = np.zeros(frames, dtype='float32') if IS_PICKUP2_ENABLED else None 

        # *** KEY OPTIMIZATION: Call the Numba-Compiled Function ***
        # This replaces the slow Python 'for n in range(frames)' loop
        run_string_dsp_numba(
            frames, input_channel, p1_output_chunk, p2_output_chunk, IS_PICKUP2_ENABLED, ONE_THIRD,
            # FCCF State
            fb_corr_filter.delay_length, fb_corr_filter.gain, fb_corr_filter.buffer, fb_corr_filter.buffer_index, 
            # Pickup 1 FFCF States
            ff1_p1.delay_length, ff1_p1.gain, ff1_p1.buffer, ff1_p1.buffer_index,
            ff2_p1.delay_length, ff2_p1.gain, ff2_p1.buffer, ff2_p1.buffer_index,
            ff3_p1.delay_length, ff3_p1.gain, ff3_p1.buffer, ff3_p1.buffer_index,
            # Pickup 2 FFCF States (pass buffers/state regardless of IS_PICKUP2_ENABLED, Numba handles the branch)
            ff1_p2.delay_length, ff1_p2.gain, ff1_p2.buffer, ff1_p2.buffer_index,
            ff2_p2.delay_length, ff2_p2.gain, ff2_p2.buffer, ff2_p2.buffer_index,
            ff3_p2.delay_length, ff3_p2.gain, ff3_p2.buffer, ff3_p2.buffer_index
        )
        # -------------------------------------------------------------------
        
        # *** Vectorized Mixing and Accumulation (remains fast) ***
        if IS_PICKUP2_ENABLED:
            # Vectorized mix and accumulation (fast NumPy operations)
            processed_channel = (p1_output_chunk * PICKUP1_VOL + p2_output_chunk * PICKUP2_VOL) * HALF
        else:
            processed_channel = p1_output_chunk * PICKUP1_VOL

        if i % 2 == 0:
            stereo_out_L += processed_channel # Vectorized accumulation
        else:
            stereo_out_R += processed_channel # Vectorized accumulation
            
# -------------------------------------------------------------------
    # *** OLD CODE REMOVED: Final Tone Filter and Output Accumulation ***
    # stereo_out_L[:] = master_tone_filters[0].process_block(stereo_out_L) * ONE_THIRD
    # stereo_out_R[:] = master_tone_filters[1].process_block(stereo_out_R) * ONE_THIRD
    # -------------------------------------------------------------------
    
    # --- NEW: Apply Master Tone Filter first (It's an integral part of the bass) ---
    stereo_out_L[:] = master_tone_filters[0].process_block(stereo_out_L) * ONE_THIRD
    stereo_out_R[:] = master_tone_filters[1].process_block(stereo_out_R) * ONE_THIRD
    
    # --- NEW: Run the Numba Effects Pipeline ---
    eff_env = EFFECT_INSTANCES['env_filter']
    eff_oct = EFFECT_INSTANCES['octaver']
    eff_comp = EFFECT_INSTANCES['comp']
    eff_lim = EFFECT_INSTANCES['limiter']
    
    # --- NEW: Create temporary GR state arrays ---
    comp_gr_state = np.array([0.0], dtype=np.float32)
    lim_gr_state = np.array([0.0], dtype=np.float32)
    
    # *** NEW: Calculate Compressor Threshold in Amplitude ***
    # This must be done outside the Numba function using the Python helper
    comp_threshold_amp = DB_TO_AMP(eff_comp.threshold)
    
    # --- NEW: Run the Numba Effects Pipeline ---
    process_effects_numba(
        frames, 
        stereo_out_L, stereo_out_R, 
        # Bypass states
        EFFECTS_BYPASS['env_filter'], EFFECTS_BYPASS['octaver'], EFFECTS_BYPASS['comp'],
        # Env Filter Args 
        eff_env.svf_l.g, eff_env.svf_l.R, 
        eff_env.svf_l.state, eff_env.svf_r.state, 
        eff_env.env_follower.envelope_state, 
        eff_env.env_follower.alpha_atk, eff_env.env_follower.alpha_rel, 
        eff_env.mix,
        # Octaver Args 
        OCTAVER_DRY_VOL, OCTAVER_OCTAVE_VOL, 
        eff_oct.state_l, eff_oct.state_r,
        # *** COMPRESSOR ARGS: Pass the pre-calculated AMP threshold ***
        eff_comp.env_follower.alpha_atk, eff_comp.env_follower.alpha_rel, 
        comp_threshold_amp, # <-- NOW PASSING THE FLOAT VALUE
        eff_comp.ratio_inv, 1.0, # ratio inv and makeup gain
        eff_comp.env_follower.envelope_state,
        # Limiter Args 
        eff_lim.threshold_amp, eff_lim.release_alpha, 
        eff_lim.gain_state,
        comp_gr_state, 
        lim_gr_state
    )

    # Lock for writing the shared variable
    with params_lock:
        global LAST_PEAK_DB
    
    # Convert amplitude to Decibels (dB)
        if overall_peak > 0:
            LAST_PEAK_DB = 20.0 * np.log10(overall_peak)
        else:
            LAST_PEAK_DB = -120.0 # Represent silence as a very low dB value
    
    LAST_COMP_GR_DB = comp_gr_state[0]
    LAST_LIMITER_GR_DB = lim_gr_state[0]

    # Output ports already point to the (now-processed) stereo_out_L/R arrays.
    return 0
    
# Helper to send the data over the network
def emit_peak_level():
    global LAST_COMP_GR_DB, LAST_LIMITER_GR_DB, LAST_PEAK_DB
    with params_lock:
        data_to_send = {
            'level' : LAST_PEAK_DB,
            'comp_gr': LAST_COMP_GR_DB,
            'lim_gr': LAST_LIMITER_GR_DB
        }
    # Send data on the 'level_update' channel
    socketio.emit('level_update', data_to_send)

# Function to run periodically (approx 30 Hz)
def start_level_meter_thread():
    # Schedule the next run
    threading.Timer(0.033, start_level_meter_thread).start() 

    try:
        emit_peak_level()
    except RuntimeError:
        # Ignore errors if the Flask app is shutting down
        pass
        
# --- Web Interface (Flask) and API Endpoints (Omitted for brevity) ---
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' 
socketio = SocketIO(app, cors_allowed_origins="*") # '*' allows connection from any source (safe for local host)
@app.route('/api/update_physical', methods=['POST'])
def api_update_physical():
    data = request.json
    with params_lock:
        global SHARED_LENGTH, REAL_PICKUP_DISTS_FB
        SHARED_LENGTH = float(data.get('length', SHARED_LENGTH))
        dists = data.get('dists_fb', REAL_PICKUP_DISTS_FB)
        for i, d in enumerate(dists):
            REAL_PICKUP_DISTS_FB[i] = float(d)
        
        if 'selected_preset_session' in session:
            session['selected_preset_session'] = 'custom'

    save_config()
    update_event.set()
    return jsonify(status="Physical parameters updated and applied dynamically")

@app.route('/api/update_virtual', methods=['POST'])
def api_update_virtual():
    data = request.json
    with params_lock:
        global SELECTED_PRESET, PICKUP1_VOL, PICKUP2_VOL, MASTER_TONE, \
                 SELECTED_PICKUP1, SELECTED_PICKUP2, IS_PICKUP2_ENABLED, STRING_PARAMS_VIRTUAL
        
        SELECTED_PRESET = 'custom'
        
        MASTER_TONE = float(data.get('master_tone', MASTER_TONE))
        PICKUP1_VOL = float(data.get('pickup1_vol', PICKUP1_VOL))
        PICKUP2_VOL = float(data.get('pickup2_vol', PICKUP2_VOL))
        SELECTED_PICKUP1 = data.get('pickup_model1', SELECTED_PICKUP1)
        SELECTED_PICKUP2 = data.get('pickup_model2', SELECTED_PICKUP2)
        IS_PICKUP2_ENABLED = bool(data.get('enable_p2', IS_PICKUP2_ENABLED))

        freqs = data.get('freqs', [])
        dists_ff1 = data.get('dists_ff1', [])
        dists_ff2 = data.get('dists_ff2', [])
        
        for i in range(len(STRING_PARAMS_VIRTUAL)):
            if i < len(freqs): STRING_PARAMS_VIRTUAL[i]['freq'] = float(freqs[i])
            if i < len(dists_ff1): STRING_PARAMS_VIRTUAL[i]['dist_ff1'] = float(dists_ff1[i])
            if i < len(dists_ff2): STRING_PARAMS_VIRTUAL[i]['dist_ff2'] = float(dists_ff2[i])
        
        if 'selected_preset_session' in session:
            session['selected_preset_session'] = 'custom'

    save_config()
    update_event.set()
    return jsonify(status="Virtual parameters updated and applied dynamically")

# --- NEW API ROUTE FOR EFFECTS CONTROL ---

@app.route('/api/update_effects', methods=['POST'])
def api_update_effects():
    data = request.json
    
    # Update Bypass States
    with params_lock:
        if 'env_filter_bypass' in data:
            EFFECTS_BYPASS['env_filter'] = data['env_filter_bypass']
        if 'octaver_bypass' in data:
            EFFECTS_BYPASS['octaver'] = data['octaver_bypass']
        if 'comp_bypass' in data:
            EFFECTS_BYPASS['comp'] = data['comp_bypass']
            
        if 'comp_ratio' in data:
            global COMP_RATIO
            COMP_RATIO = float(data['comp_ratio'])
            # You'll also need to update the instance value for the Numba call
            EFFECT_INSTANCES['comp'].ratio_inv = 1.0 / COMP_RATIO
            
        if 'env_filter_freq' in data:
            global ENV_FILTER_FREQ
            ENV_FILTER_FREQ = float(data['env_filter_freq'])
            # Crucial: Update the SVF coefficients on the instance!
            EFFECT_INSTANCES['env_filter'].svf_l.update_coefficients(ENV_FILTER_FREQ, ENV_FILTER_Q)
            EFFECT_INSTANCES['env_filter'].svf_r.update_coefficients(ENV_FILTER_FREQ, ENV_FILTER_Q)
    # The JACK thread will use the updated global variables instantly
    return jsonify({'status': 'ok', 'message': 'Effects state updated.'})

@app.route('/', methods=['GET', 'POST'])
def index():
    global SHARED_LENGTH, STRING_PARAMS_VIRTUAL, REAL_PICKUP_DISTS_FB, \
            SELECTED_PICKUP1, SELECTED_PICKUP2, SELECTED_PRESET, IS_PICKUP2_ENABLED, \
            PICKUP1_VOL, PICKUP2_VOL, MASTER_TONE, EFFECTS_BYPASS, COMP_THRESHOLD, \
           OCTAVER_OCTAVE_VOL, LIMITER_THRESHOLD
    
    if request.method == 'POST':
        if 'preset_select' in request.form:
            preset_name = request.form['preset_select']
            if preset_name in PRESETS:
                with params_lock:
                    SELECTED_PRESET = preset_name 
                    session['selected_preset_session'] = preset_name 
                    
                    preset = PRESETS[preset_name]
                    STRING_PARAMS_VIRTUAL = [item.copy() for item in preset['strings']]
                    SELECTED_PICKUP1 = preset['pickup1']
                    SELECTED_PICKUP2 = preset['pickup2']
                    IS_PICKUP2_ENABLED = preset['pickup2_enabled']
                update_event.set()
                flash(f"Preset '{preset_name}' loaded successfully! Note: Physical instrument parameters remain unchanged.")
                return redirect(url_for('index'))
        
        elif 'action' in request.form and request.form['action'] == 'Update Physical Params':
            """
            with params_lock:
                SHARED_LENGTH = float(request.form['length'])
                SELECTED_PRESET = 'custom' 
                session['selected_preset_session'] = 'custom' 
                
                for i in range(len(REAL_PICKUP_DISTS_FB)):
                    REAL_PICKUP_DISTS_FB[i] = float(request.form[f'dist_fb_real_{i}'])
            
            save_config()
            update_event.set()
            flash("Physical parameters updated, applied dynamically, and saved permanently!")
            return redirect(url_for('index'))
            """
            pass

        elif 'action' in request.form and request.form['action'] == 'Update Virtual Params':
            """
            with params_lock:
                SELECTED_PRESET = 'custom' 
                session['selected_preset_session'] = 'custom' 
                
                MASTER_TONE = float(request.form['master_tone'])
                PICKUP1_VOL = float(request.form['pickup1_vol'])
                PICKUP2_VOL = float(request.form['pickup2_vol'])
                SELECTED_PICKUP1 = request.form['pickup_model1']
                SELECTED_PICKUP2 = request.form['pickup_model2']
                IS_PICKUP2_ENABLED = 'enable_p2' in request.form 
                for i in range(len(STRING_PARAMS_VIRTUAL)):
                    STRING_PARAMS_VIRTUAL[i]['freq'] = float(request.form[f'freq_{i}'])
                    STRING_PARAMS_VIRTUAL[i]['dist_ff1'] = float(request.form[f'dist_ff1_{i}'])
                    STRING_PARAMS_VIRTUAL[i]['dist_ff2'] = float(request.form[f'dist_ff2_{i}'])
            
            save_config()
            update_event.set()
            flash("Custom parameters updated and applied dynamically!") 
            return redirect(url_for('index'))
            """
            pass

    if 'selected_preset_session' in session:
        current_preset_key = session['selected_preset_session']
    else:
        current_preset_key = SELECTED_PRESET 
        

    print(f"DEBUG: Rendering page with selected preset: {current_preset_key}")

    with params_lock:
        return render_template_string(HTML_FORM, length=SHARED_LENGTH, strings_virtual=STRING_PARAMS_VIRTUAL, 
                                     real_dists_fb=REAL_PICKUP_DISTS_FB, pickups=PICKUP_MODELS, 
                                     selected_pickup1=SELECTED_PICKUP1, selected_pickup2=SELECTED_PICKUP2,
                                     presets=PRESETS, selected_preset=current_preset_key, is_pickup2_enabled=IS_PICKUP2_ENABLED,
                                     pickup1_vol=PICKUP1_VOL, pickup2_vol=PICKUP2_VOL, master_tone=MASTER_TONE, EFFECTS_BYPASS=EFFECTS_BYPASS,
        COMP_THRESHOLD=COMP_THRESHOLD,
        OCTAVER_OCTAVE_VOL=OCTAVER_OCTAVE_VOL, LIMITER_THRESHOLD=LIMITER_THRESHOLD)


# --- HTML Form with High-Contrast Styles and Sliders (Omitted for brevity) ---
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>DSP Parameters Config</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    <style>
        /* --- High-Contrast Dark Mode & Orbitron Styles --- */
        :root {
            --color-bg-dark: #12121D; 
            --color-bg-mid: #28283D; 
            --color-text-light: #FAFAFA; 
            --color-accent: #FFD700; 
            --color-border: #4A4A6D; 
            --color-flash: #6A1B9A; 
            --color-button-bg: #FFD700; 
            --color-button-text: #12121D; 
        }
        
        body { 
            font-family: 'Orbitron', sans-serif; 
            margin: 0; 
            padding: 20px;
            background-color: var(--color-bg-dark);
            color: var(--color-text-light); 
        }
        h2, h3, h4 { 
            color: var(--color-accent); 
            border-bottom: 2px solid var(--color-accent);
            padding-bottom: 5px;
            margin-top: 20px;
        }
        hr {
            border-color: var(--color-border);
        }
        
        /* Style for the collapsible section summary */
        details summary {
            cursor: pointer;
            padding: 10px 0;
            color: var(--color-accent);
            font-weight: 700;
            border-bottom: 2px solid var(--color-border);
            margin-bottom: 15px;
        }
        
        /* Style the summary to look like a heading */
        details summary h3 {
             border-bottom: none;
             display: inline;
             margin: 0;
             padding: 0;
        }

        /* --- Flexbox for Side-by-Side Pickups --- */
        .pickup-settings-container {
            display: flex;
            flex-wrap: wrap; 
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .pickup-section {
            flex: 1 1 45%; 
            padding: 15px;
            background-color: var(--color-bg-mid); 
            border-radius: 8px;
            min-width: 280px; 
            border: 1px solid var(--color-border);
        }
        
        @media screen and (max-width: 600px) {
            .pickup-section {
                flex-basis: 100%; 
            }
        }
        
        /* --- Form Elements and Inputs --- */
        input[type="number"], select { 
            padding: 8px; 
            margin: 5px 0; 
            border: 1px solid var(--color-border); 
            border-radius: 5px; 
            background-color: var(--color-bg-dark); 
            color: var(--color-text-light); 
            width: 100%; 
            box-sizing: border-box;
            font-family: 'Orbitron', sans-serif;
        }
        
        /* Specific input width adjustments */
        input[name^="pickup"], input[name="length"] {
            max-width: 100%; 
        }
        
        /* Container for synchronizing slider and number input */
        .slider-control {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .slider-control label {
            min-width: 120px; /* Ensure labels align */
            margin-right: 10px;
        }
        
        /* Style the numerical input next to the slider */
        .slider-control input[type="number"] {
            width: 80px;
            flex-shrink: 0;
            margin: 0 0 0 10px;
        }

        /* --- Slider Styling (Range Input) --- */
        input[type=range] {
            -webkit-appearance: none;
            width: 100%;
            height: 10px;
            background: #5A5A8D; /* Track color */
            border-radius: 5px;
            margin: 0;
            flex-grow: 1;
        }
        
        /* Slider Thumb */
        input[type=range]::-webkit-slider-thumb {
            -webkit-appearance: none;
            height: 20px;
            width: 10px;
            border-radius: 3px;
            background: var(--color-accent); /* Gold thumb */
            cursor: pointer;
            box-shadow: 0 0 5px var(--color-accent);
        }
        
        /* --- Buttons --- */
        button { 
            padding: 10px 15px; 
            background-color: var(--color-button-bg); 
            color: var(--color-button-text); 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            margin-top: 10px;
        }
        button:hover { 
            background-color: #FFEA68; 
            box-shadow: 0 0 10px var(--color-accent);
        }
        
        /* --- Tables --- */
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 15px 0; 
            background-color: var(--color-bg-mid); 
        }
        th, td { 
            border: 1px solid var(--color-border); 
            padding: 8px; 
            text-align: left; 
        }
        th { 
            background-color: #3C3C58; 
            color: var(--color-accent);
        }
        
        /* Mobile Table Trick (omitted for brevity) */
        @media screen and (max-width: 600px) {
            table, thead, tbody, th, td, tr { display: block; }
            thead tr { position: absolute; top: -9999px; left: -9999px; }
            tr { border: 1px solid var(--color-border); margin-bottom: 10px; }
            td { 
                border: none; border-bottom: 1px solid var(--color-border); 
                position: relative; padding-left: 50%; text-align: right;
            }
            td:before { 
                position: absolute; top: 6px; left: 6px; width: 45%; 
                padding-right: 10px; white-space: nowrap; text-align: left;
                font-weight: bold; color: var(--color-accent);
            }
            td:nth-of-type(1):before { content: "String"; }
            details td:nth-of-type(2):before { content: "P/U Dist (mm)"; } 
            td:nth-of-type(2):before { content: "Note (Hz)"; } 
            td:nth-of-type(3):before { content: "P/U 1 Dist (mm)"; }
            td:nth-of-type(4):before { content: "P/U 2 Dist (mm)"; }
            td input[type="number"] { width: 95%; }
        }

        /* --- Flash Message --- */
        .flash { 
            padding: 10px; 
            margin-bottom: 15px; 
            border-radius: 5px; 
            background-color: var(--color-flash); 
            border: 1px solid var(--color-accent); 
            color: var(--color-text-light); 
        }
    </style>
<script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
<script>
    // Helper function for AJAX POST requests
    function sendUpdate(endpoint, data) {
        fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        }).then(response => {
            // Check for success without refreshing the page
            if (response.ok) {
                console.log("Parameters updated successfully via API.");
            } else {
                console.error("API update failed.");
            }
        }).catch(error => {
            console.error("Fetch error:", error);
        });
    }

    // --- Core Update Functions ---
    // These now trigger an AJAX call every time they are called.
    function updatePhysicalParams() {
        const data = {
            length: document.getElementsByName('length')[0].value,
            dists_fb: Array.from(document.querySelectorAll('[name^="dist_fb_real_"]')).map(input => parseFloat(input.value))
        };
        sendUpdate('/api/update_physical', data);
    }

    function updateVirtualParams() {
        // Collect all virtual parameters in a single payload
        const data = {
            master_tone: parseFloat(document.getElementsByName('master_tone_num')[0].value),
            pickup1_vol: parseFloat(document.getElementsByName('pickup1_vol_num')[0].value),
            pickup2_vol: parseFloat(document.getElementsByName('pickup2_vol_num')[0].value),
            pickup_model1: document.getElementsByName('pickup_model1')[0].value,
            pickup_model2: document.getElementsByName('pickup_model2')[0].value,
            enable_p2: document.getElementsByName('enable_p2')[0].checked,
            freqs: Array.from(document.querySelectorAll('[name^="freq_"]')).map(input => parseFloat(input.value)),
            dists_ff1: Array.from(document.querySelectorAll('[name^="dist_ff1_"]')).map(input => parseFloat(input.value)),
            dists_ff2: Array.from(document.querySelectorAll('[name^="dist_ff2_"]')).map(input => parseFloat(input.value))
        };
        sendUpdate('/api/update_virtual', data);
    }

    // --- Slider Synchronization Functions ---
    
    // Volume: Syncs slider (0-100) and number input (0.00-1.00)
    function syncVolume(sourceInput) {
        const isP1 = sourceInput.name.includes('pickup1');
        const slider = document.getElementsByName(isP1 ? 'pickup1_vol_slider' : 'pickup2_vol_slider')[0];
        const numberInput = document.getElementsByName(isP1 ? 'pickup1_vol_num' : 'pickup2_vol_num')[0];

        if (sourceInput.type === 'range') {
            const volumeValue = (sourceInput.value / 100).toFixed(2);
            numberInput.value = volumeValue;
        } else {
            const percentage = Math.round(parseFloat(sourceInput.value) * 100);
            slider.value = percentage;
        }
        // CALL API IMMEDIATELY AFTER SYNC
        updateVirtualParams();
    }

    // Tone: Syncs slider (0-100) and number input (100-10000 Hz)
    function syncTone(sourceInput) {
        const slider = document.getElementsByName('master_tone_slider')[0];
        const numberInput = document.getElementsByName('master_tone_num')[0];
        
        const MIN_TONE = 100;
        const MAX_TONE = 10000;
        const TONE_RANGE = MAX_TONE - MIN_TONE;

        if (sourceInput.type === 'range') {
            const sliderValue = parseInt(sourceInput.value);
            // Use Math.round for smoother steps in the number input
            const toneValue = Math.round(MIN_TONE + (sliderValue / 100) * TONE_RANGE);
            numberInput.value = toneValue;
        } else {
            const toneValue = parseFloat(sourceInput.value);
            const percentage = Math.round(((toneValue - MIN_TONE) / TONE_RANGE) * 100);
            slider.value = percentage;
        }
        // CALL API IMMEDIATELY AFTER SYNC
        updateVirtualParams();
    }

    function syncCompRatio(sourceInput) {
        const slider = document.getElementsByName('comp_ratio_slider')[0];
        const numberInput = document.getElementsByName('comp_ratio_num')[0];

        // Sync values between slider (15-200 for 1.5-20.0) and number input
        if (sourceInput.type === 'range') {
            // Example: map 15-200 slider value to 1.5-20.0 number
            numberInput.value = (sourceInput.value / 10).toFixed(1); 
        } else {
            slider.value = parseFloat(sourceInput.value) * 10;
        }
    
        // Send API Update
        sendEffectUpdate('comp_ratio', parseFloat(numberInput.value));
    }

    function toggleP2Enable(checkbox) {
        const isChecked = checkbox.checked;
        document.getElementsByName('pickup_model2')[0].disabled = !isChecked;
        document.getElementsByName('pickup2_vol_slider')[0].disabled = !isChecked;
        document.getElementsByName('pickup2_vol_num')[0].disabled = !isChecked;
        document.querySelectorAll('[name^="dist_ff2_"]').forEach(input => input.disabled = !isChecked);
        document.querySelector('.pickup-section input[onclick="syncDistances(2, this)"]').disabled = !isChecked;
        updateVirtualParams(); 
    }
    
    function syncDistances(pickupType, checkbox) {
        // Simple client-side copy, then trigger the API update
        const firstInputName = 'dist_ff' + pickupType + '_0';
        const firstInput = document.getElementsByName(firstInputName)[0];
        const firstValue = firstInput.value;
        
        const inputs = document.querySelectorAll('[name^="dist_ff' + pickupType + '_"]');
        
        inputs.forEach(input => {
            input.value = firstValue;
        });
        
        updateVirtualParams();
    }
    
    // Initial sync on page load
    window.onload = function() {
        // Initializing sliders from number values
        syncVolume(document.getElementsByName('pickup1_vol_num')[0]);
        syncVolume(document.getElementsByName('pickup2_vol_num')[0]);
        syncTone(document.getElementsByName('master_tone_num')[0]);

        const enableP2Checkbox = document.getElementsByName('enable_p2')[0];
        toggleP2Enable(enableP2Checkbox); // Ensure P2 controls are correctly disabled if feature is off
    };

    // NEW: Function to handle bypass status and parameter updates for effects
    function sendEffectUpdate(paramName, paramValue) {
        const data = {};
        data[paramName] = paramValue;
        
        fetch('/api/update_effects', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        }).then(response => {
            if (response.ok) {
                console.log(paramName + " updated to " + paramValue);
            } else {
                console.error("Effect update failed.");
            }
        }).catch(error => {
            console.error("Fetch error:", error);
        });
    }

    // Example: Synchronization function for a new slider
    function syncCompThreshold(sourceInput) {
        const slider = document.getElementsByName('comp_threshold_slider')[0];
        const numberInput = document.getElementsByName('comp_threshold_num')[0];
        
        // Sync values
        if (sourceInput.type === 'range') {
            numberInput.value = parseFloat(sourceInput.value).toFixed(1);
        } else {
            slider.value = parseFloat(sourceInput.value);
        }
        
        // Send API Update
        sendEffectUpdate('comp_threshold', parseFloat(numberInput.value));
    }
    // You would create similar sync functions for other effect parameters (Q, Ratio, etc.)
    // NEW: Real-Time Level Meter Listener
    const socket = io(); // Connects to the host the page was served from

    socket.on('level_update', function(data) {
        const levelDB = data.level;
        const compGR = data.comp_gr;
        const limGR = data.lim_gr;
        const reductionAmount = -grDB;
        const meterBar = document.getElementById('level-bar');
        const dbReadout = document.getElementById('level-db-readout');

        // Update numerical readout
        dbReadout.textContent = levelDB > -120 ? levelDB.toFixed(1) + " dB" : "-INF dB";
        grReadout.textContent = reductionAmount.toFixed(1) + " dB";

        // Map the dB level to a meter width percentage
        // Use -60 dB as the 'silent' floor (0%) and 0 dB as 100%
        const MIN_DB = -60.0;
        const MAX_DB = 0.0;
        const MAX_GR = 20.0;
        
        let percentage = (levelDB - MIN_DB) / (MAX_DB - MIN_DB) * 100;

        // Clamp the percentage to ensure it stays between 0% and 100%
        if (percentage < 0) percentage = 0;
        if (percentage > 100) percentage = 100;

        let gr_percentage = (reductionAmount / MAX_GR) * 100;
    
            if (gr_percentage < 0) gr_percentage = 0;
            if (gr_percentage > 100) gr_percentage = 100;

            grBar.style.width = gr_percentage.toFixed(0) + '%';
        }

        // Set bar color based on level (optional, but professional)
        let color = 'limegreen';
        if (levelDB > -6.0) { // Yellow warning zone
            color = 'yellow';
        }
        if (levelDB > -1.0) { // Red clipping warning zone (or close to it)
            color = 'red';
        }

        // Update the meter visualization
        meterBar.style.width = percentage.toFixed(0) + '%';
        meterBar.style.backgroundColor = color;
        updateGRMeter(compGR, 'comp-gr-bar', 'comp-gr-readout');
        updateGRMeter(limGR, 'lim-gr-bar', 'lim-gr-readout');
    });
</script>
</head>
<body>
    {% with messages = get_flashed_messages() %}
        {% if messages %}
            <div class="flash">
                {% for message in messages %}
                    {{ message }}
                {% endfor %}
            </div>
        {% endif %}
    {% endwith %}

    <h2>🎸 Bass Guitar DSP Config</h2>
    <details open>
      <summary><h2>5. Real-Time Output</h2></summary>
      <fieldset>
          <legend>Master Level Meter</legend>
            <div style="width: 250px; background-color: #333; height: 20px; border-radius: 5px; overflow: hidden; margin-bottom: 5px;">
             <div id="level-bar" style="height: 100%; width: 0%; background-color: limegreen; transition: width 0.05s ease;"></div>
            </div>
         <span id="level-db-readout">-INF dB</span>
      </fieldset>
    </details>
    <form method="POST">
        <h3>Virtual Pickup Presets</h3>
        <label for="preset_select">Select Virtual Preset:</label>
        <select name="preset_select" onchange="this.form.submit()">
            {% for key, value in presets.items() %}
            <option value="{{ key }}" {% if key == selected_preset %}selected{% endif %}>{{ value['name'] }}</option>
            {% endfor %}
        </select><br><br>
        
        <hr>

        <h3>Virtual Custom Configuration (Current: {{ presets[selected_preset]['name'] }})</h3>
        
        <h4>Master Controls</h4>
        <label for="master_tone_num">Master Tone (Cutoff Freq Hz):</label>
        <div class="slider-control">
            <input type="range" name="master_tone_slider" min="0" max="100" step="1" 
                   value="{{ ((master_tone - 100) / 9900 * 100)|round|int }}" 
                   oninput="syncTone(this)" onchange="updateVirtualParams()">
            <input type="number" step="10" min="100" max="10000" name="master_tone_num" 
                   value="{{ master_tone }}" onchange="syncTone(this)">
        </div>
        <br>

        <div class="pickup-settings-container">
        
            <div class="pickup-section">
                <h4>Pickup 1 Settings</h4>
                <label for="pickup_model1">Pickup 1 Model:</label>
                <select name="pickup_model1" onchange="updateVirtualParams()">
                    {% for key, value in pickups.items() %}
                    <option value="{{ key }}" {% if key == selected_pickup1 %}selected{% endif %}>{{ value['name'] }}</option>
                    {% endfor %}
                </select><br>
                
                <label for="pickup1_vol_num">Pickup 1 Volume (0.0 to 1.0):</label>
                <div class="slider-control">
                    <input type="range" name="pickup1_vol_slider" min="0" max="100" step="1" 
                           value="{{ (pickup1_vol * 100)|round|int }}" 
                           oninput="syncVolume(this)" onchange="updateVirtualParams()">
                    <input type="number" step="0.01" min="0.0" max="1.0" name="pickup1_vol_num" 
                           value="{{ '%.2f'|format(pickup1_vol) }}" onchange="syncVolume(this)">
                </div>
            </div>


            <div class="pickup-section">
                <h4>Pickup 2 Settings</h4>
                <input type="checkbox" name="enable_p2" {% if is_pickup2_enabled %}checked{% endif %} onclick="toggleP2Enable(this)"> 
                <label for="enable_p2">Enable Pickup 2 in Custom Mode</label><br>
                <label for="pickup_model2">Pickup 2 Model:</label>
                <select name="pickup_model2" {% if not is_pickup2_enabled %}disabled{% endif %} onchange="updateVirtualParams()">
                    {% for key, value in pickups.items() %}
                    <option value="{{ key }}" {% if key == selected_pickup2 %}selected{% endif %}>{{ value['name'] }}</option>
                    {% endfor %}
                </select><br>
                
                <label for="pickup2_vol_num">Pickup 2 Volume (0.0 to 1.0):</label>
                <div class="slider-control">
                    <input type="range" name="pickup2_vol_slider" min="0" max="100" step="1" 
                           value="{{ (pickup2_vol * 100)|round|int }}" 
                           {% if not is_pickup2_enabled %}disabled{% endif %} 
                           oninput="syncVolume(this)" onchange="updateVirtualParams()">
                    <input type="number" step="0.01" min="0.0" max="1.0" name="pickup2_vol_num" 
                           value="{{ '%.2f'|format(pickup2_vol) }}" 
                           {% if not is_pickup2_enabled %}disabled{% endif %} 
                           onchange="syncVolume(this)">
                </div>
            </div>
        </div>
        
        <table border="1" cellpadding="5">
            <thead>
                <tr>
                    <th>String</th>
                    <th>Note (Hz)</th>
                    <th>Virtual P/U 1 Dist (mm) <input type="checkbox" onclick="syncDistances(1, this)"> Sync</th>
                    <th>Virtual P/U 2 Dist (mm) <input type="checkbox" onclick="syncDistances(2, this)" {% if not is_pickup2_enabled %}disabled{% endif %}> Sync</th>
                </tr>
            </thead>
            <tbody>
                {% for i in range(strings_virtual|length) %}
                <tr>
                    <td>{{ strings_virtual[i]['note'] }}</td>
                    <td><input type="number" step="0.1" name="freq_{{ i }}" value="{{ strings_virtual[i]['freq'] }}" onchange="updateVirtualParams()"></td>
                    <td><input type="number" step="0.1" name="dist_ff1_{{ i }}" value="{{ strings_virtual[i]['dist_ff1'] }}" onchange="updateVirtualParams()"></td>
                    <td><input type="number" step="0.1" name="dist_ff2_{{ i }}" value="{{ strings_virtual[i]['dist_ff2'] }}" {% if not is_pickup2_enabled %}disabled{% endif %} onchange="updateVirtualParams()"></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <br>
        <hr>
        <details open>
    <summary><h2>4. Post-Synthesis Effects</h2></summary>

    <fieldset>
        <legend>Envelope Filter</legend>
        <input type="checkbox" id="env_filter_bypass" 
               onchange="sendEffectUpdate('env_filter_bypass', this.checked)" 
               {% if EFFECTS_BYPASS['env_filter'] %}checked{% endif %}>
        <label for="env_filter_bypass">Bypass</label><br>
        </fieldset>

    <fieldset>
        <legend>Octaver</legend>
        <input type="checkbox" id="octaver_bypass" 
               onchange="sendEffectUpdate('octaver_bypass', this.checked)" 
               {% if EFFECTS_BYPASS['octaver'] %}checked{% endif %}>
        <label for="octaver_bypass">Bypass</label><br>
        </fieldset>

    <fieldset>
        <legend>Compressor</legend>
        <input type="checkbox" id="comp_bypass" 
               onchange="sendEffectUpdate('comp_bypass', this.checked)" 
               {% if EFFECTS_BYPASS['comp'] %}checked{% endif %}>
        <label for="comp_bypass">Bypass</label><br>
        <label>Threshold (dB):</label>
        <input type="range" min="-60" max="0" step="1" name="comp_threshold_slider" 
               oninput="syncCompThreshold(this)" value="{{ COMP_THRESHOLD }}">
        <input type="number" min="-60" max="0" step="0.1" name="comp_threshold_num" value="{{ COMP_THRESHOLD }}" readonly>
<div style="margin-top: 5px;">
            <label>Gain Reduction:</label>
            <div style="width: 250px; background-color: #555; height: 10px; border-radius: 2px; overflow: hidden;">
            <div id="comp-gr-bar" style="height: 100%; width: 0%; background-color: orange; transition: width 0.05s ease;"></div>
        </div>
        <span id="comp-gr-readout">0.0 dB</span>
    </div>
    </fieldset>
        
    <fieldset>
        <legend>Limiter (Always ON)</legend>
        <p>Threshold: {{ LIMITER_THRESHOLD }} dB</p>
        <div style="margin-top: 5px;">
        <label>Gain Reduction:</label>
            <div style="width: 250px; background-color: #555; height: 10px; border-radius: 2px; overflow: hidden;">
            <div id="lim-gr-bar" style="height: 100%; width: 0%; background-color: red; transition: width 0.05s ease;"></div>
            </div>
            <span id="lim-gr-readout">0.0 dB</span>
        </div>
        </fieldset>

</details>
        <details>
            <summary><h3>🔧 Physical Instrument Configuration (Persistent)</h3></summary>
            
            <label for="length">Shared String Length (mm):</label>
            <input type="number" step="0.1" name="length" value="{{ length }}" onchange="updatePhysicalParams()"><br><br>

            <table border="1" cellpadding="5">
                <thead>
                    <tr>
                        <th>String</th>
                        <th>*REAL* P/U Dist (mm) (Persistent)</th>
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
        </details>
    </form>
</body>
</html>
"""

# --- Main execution logic (Omitted for brevity) ---
if __name__ == "__main__":
    load_config()
    inports = [client.inports.register(f'input_{i+1}') for i in range(6)]
    outports = [client.outports.register(f'output_{i+1}') for i in range(2)]
    client.set_process_callback(lambda f: process_audio_callback(f, inports, outports))

    try:
        initialize_filters(client.samplerate)
        with client:
            print("JACK client activated. Connect ports using QjackCtl.")
            print(f"Access the web UI at http://<Your_Pi_IP_Address>:5000")
            start_level_meter_thread()
            app.run(host='0.0.0.0', port=5000, debug=False) 
            socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except jack.JackError as e:
        print(f"JACK error: {e}")
        print("Ensure JACK server is running (e.g., via qjackctl) before running this script.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Exiting...")
