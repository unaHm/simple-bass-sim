#!/usr/bin/env python3
"""
jack_engine.py - with granular pitch shifter (numba)
- Packed comb filters (numpy buffers) + numba
- SVF envelope (numba)
- Granular pitch shifter (numba) --- one-octave down by default
- Compressor, Limiter (numba)
- Backwards-compatible API surface
"""

import jack
import numpy as np
import math
import time
import json
import dsp
from pathlib import Path

# ----------------------------
# Basic constants / globals
# ----------------------------
NUM_STRINGS = 6
ROOT = Path(__file__).resolve().parent
PRESET_DIR = ROOT / "presets"
FACTORY_DIR = PRESET_DIR / "factory"
LAST_PRESET_FILE = PRESET_DIR / "last_preset.txt"
GLOBAL_EFFECTS_FILE = PRESET_DIR / "global_effects.json"

NUM_PICKUPS = 2
TAPS_PER_PICKUP = 3
MAX_DELAY_SAMPLES = 512
STRING_FREQS = [30.8677, 41.2034, 55.0, 73.416, 97.999, 130.813]

DSP_PARAMS = np.array([1.0, 1.0, 20000.0, 1.0], dtype=np.float32)

EFFECT_PARAMS = {
    "svf_base_cutoff": 1000.0,
    "svf_env_depth": 0.0,
    "svf_q": 0.8,

    "oct_dry": 0.5,
    "oct_sub_gain": 0.5,

    "comp_threshold": -20.0,
    "comp_ratio": 4.0,
    "comp_makeup": 0.0,

    "acoustic_body_size": 0.5, # 0-1
    "acoustic_tone": 0.5,      # 0-1
    "acoustic_mix": 0.85,      # 0-1
    "acoustic_model_preset": "Classic Violin-Corner", # Default preset
}

EFFECTS_BYPASS = {
    "env_filter": True,
    "octaver": True,
    "comp": False,
}

ACOUSTIC_MODE_ENABLED = False
acoustic_out_buffer = np.zeros(4096, dtype=np.float32) # Buffer for acoustic model output
def _ensure_acoustic_buffer(n_frames):
    global acoustic_out_buffer
    if n_frames > acoustic_out_buffer.shape[0]:
        acoustic_out_buffer = np.zeros(n_frames, dtype=np.float32)

LAST_PEAK_DB = -120.0
LAST_COMP_GR_DB = 0.0
LAST_LIMITER_GR_DB = 0.0

INSTR_MODEL = {
    "num_strings": NUM_STRINGS,
    "pickup_types": ["single", "single"],
    "closest_distance_mm_per_pickup": [40.0, 40.0],
    "comb_delays_samples": None,
}

STATIC_COMB_DELAYS = []

# ----------------------------
# JACK setup
# ----------------------------
try:
    client = jack.Client("jackbasssim_modular")
    IN_JACK = True
except Exception as e:
    client = None
    IN_JACK = False
    print("[jack_engine] JACK client create failed:", e)

if IN_JACK:
    in_ports = [client.inports.register(f"input_{i+1}") for i in range(NUM_STRINGS)]
    outL = client.outports.register("output_1")
    outR = client.outports.register("output_2")
    SAMPLERATE = client.samplerate
    print(f"[jack_engine] registered ports, samplerate={SAMPLERATE}")
else:
    in_ports = []
    outL = None
    outR = None
    SAMPLERATE = 48000

# ----------------------------
# Numba helper
# ----------------------------
try:
    import numba as _numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("[jack_engine] numba available: using JIT kernels")
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap
    def prange(x):
        return range(x)
    print("[jack_engine] numba not available: falling back to Python kernels")

def load_global_effects():
    """Loads global effects from file and applies them at startup."""
    if not GLOBAL_EFFECTS_FILE.exists():
        return

    try:
        with open(GLOBAL_EFFECTS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        if 'comp_bypass' in settings:
            EFFECTS_BYPASS['comp'] = bool(settings['comp_bypass'])
        if 'comp_params' in settings:
            for k, v in settings['comp_params'].items():
                if k in EFFECT_PARAMS:
                    EFFECT_PARAMS[k] = float(v)
        print("[jack_engine] Loaded global compressor settings.")
    except Exception as e:
        print(f"[jack_engine] Warning: Failed to load global effects: {e}")

# ----------------------------
# Helpers mm->samples
# ----------------------------
def mm_to_samples(distance_mm, freq, samplerate):
    if freq <= 0:
        return 1
    wavelength_mm = 343000.0 / freq
    frac = distance_mm / wavelength_mm
    period_samples = samplerate / freq
    delay = int(round(frac * period_samples))
    return max(1, delay)

def compute_comb_delays_from_mm(closest_mm_list, pickup_types, samplerate):
    pickup_widths = {
        "single": 24.0,
        "splitP": 56.0,
        "humbucker": 49.0,
        "soapbar": 34.0,
        "none": 0.0,
    }
    combs = []
    if not isinstance(closest_mm_list, (list, tuple)):
        closest_mm_list = [float(closest_mm_list), float(closest_mm_list)]
    if len(closest_mm_list) < 2:
        closest_mm_list = [closest_mm_list[0], closest_mm_list[0]]
    if not isinstance(pickup_types, (list, tuple)):
        pickup_types = [pickup_types, pickup_types]
    if len(pickup_types) < 2:
        pickup_types = [pickup_types[0], pickup_types[0]]
    for si in range(NUM_STRINGS):
        f = STRING_FREQS[si] if si < len(STRING_FREQS) else 100.0
        p1_mm = float(closest_mm_list[0])
        p2_mm = float(closest_mm_list[1])
        w1 = pickup_widths.get(pickup_types[0], 24.0)
        w2 = pickup_widths.get(pickup_types[1], 24.0)
        p1_positions = [p1_mm, p1_mm + w1*0.5, p1_mm + w1]
        p2_positions = [p2_mm, p2_mm + w2*0.5, p2_mm + w2]
        p1_samples = [mm_to_samples(pos, f, samplerate) for pos in p1_positions]
        p2_samples = [mm_to_samples(pos, f, samplerate) for pos in p2_positions]
        if pickup_types[0] == "none":
            p1_samples = [1,1,1]
        if pickup_types[1] == "none":
            p2_samples = [1,1,1]
        combs.append([p1_samples, p2_samples])
    return combs

# ----------------------------
# Packed comb buffers
# ----------------------------
MAX_INIT_DELAY = MAX_DELAY_SAMPLES
bufs = np.zeros((NUM_STRINGS, NUM_PICKUPS, TAPS_PER_PICKUP, MAX_INIT_DELAY), dtype=np.float32)
lens = np.ones((NUM_STRINGS, NUM_PICKUPS, TAPS_PER_PICKUP), dtype=np.int32)
idxs = np.zeros((NUM_STRINGS, NUM_PICKUPS, TAPS_PER_PICKUP), dtype=np.int32)
_max_delay = MAX_INIT_DELAY

def _ensure_buffer_size(required_max):
    global bufs, _max_delay
    if required_max <= _max_delay:
        return
    new_max = max(required_max, _max_delay * 2)
    newbuf = np.zeros((NUM_STRINGS, NUM_PICKUPS, TAPS_PER_PICKUP, new_max), dtype=np.float32)
    newbuf[:, :, :, :_max_delay] = bufs
    bufs = newbuf
    _max_delay = new_max
    print(f"[jack_engine] comb buffer resized to {_max_delay}")

def set_comb_delays(delays):
    global bufs, lens, idxs, _max_delay
    maxd = 1
    for s in range(NUM_STRINGS):
        for p in range(NUM_PICKUPS):
            for t in range(TAPS_PER_PICKUP):
                d = int(max(1, delays[s][p][t]))
                if d > maxd:
                    maxd = d
    _ensure_buffer_size(maxd)
    for s in range(NUM_STRINGS):
        for p in range(NUM_PICKUPS):
            for t in range(TAPS_PER_PICKUP):
                d = int(max(1, delays[s][p][t]))
                lens[s,p,t] = d
                idxs[s,p,t] = 0
                bufs[s,p,t,:d] = 0.0
    INSTR_MODEL["comb_delays_samples"] = delays
    print("[jack_engine] comb delays set (max delay = {})".format(maxd))

if INSTR_MODEL.get("comb_delays_samples") is None:
    default_combs = compute_comb_delays_from_mm(INSTR_MODEL["closest_distance_mm_per_pickup"], INSTR_MODEL["pickup_types"], SAMPLERATE)
    set_comb_delays(default_combs)

# Load global effects after defaults are set
load_global_effects()

# ----------------------------
# Numba comb processor
# ----------------------------
if NUMBA_AVAILABLE:
    @njit
    def comb_process_all_numba(inputs, bufs_local, lens_local, idxs_local, gains_local):
        nstrings = inputs.shape[0]
        nframes = inputs.shape[1]
        combined = np.zeros(nframes, dtype=np.float32)
        for s in range(nstrings):
            out_s = np.zeros(nframes, dtype=np.float32)
            for p in range(gains_local.shape[0]):
                g = gains_local[p]
                for t in range(lens_local.shape[2]):
                    blen = int(lens_local[s,p,t])
                    idx = int(idxs_local[s,p,t])
                    buf = bufs_local[s,p,t]
                    for i in range(nframes):
                        y = buf[idx]
                        tmp = inputs[s,i] - y
                        out_s[i] += tmp * g
                        buf[idx] = inputs[s,i]
                        idx += 1
                        if idx >= blen:
                            idx = 0
                    idxs_local[s,p,t] = idx
            for i in range(nframes):
                combined[i] += out_s[i]
        return combined
else:
    def comb_process_all_numba(inputs, bufs_local, lens_local, idxs_local, gains_local):
        nstrings = inputs.shape[0]
        nframes = inputs.shape[1]
        combined = np.zeros(nframes, dtype=np.float32)
        for s in range(nstrings):
            out_s = np.zeros(nframes, dtype=np.float32)
            for p in range(gains_local.shape[0]):
                g = gains_local[p]
                for t in range(lens_local.shape[2]):
                    blen = int(lens_local[s,p,t])
                    idx = int(idxs_local[s,p,t])
                    buf = bufs_local[s,p,t]
                    for i in range(nframes):
                        y = buf[idx]
                        tmp = inputs[s,i] - y
                        out_s[i] += tmp * g
                        buf[idx] = inputs[s,i]
                        idx += 1
                        if idx >= blen:
                            idx = 0
                    idxs_local[s,p,t] = idx
            combined += out_s
        return combined

# ----------------------------
# SVF, Compressor, Limiter (numba)
# ----------------------------
import math as _math

@njit
def svf_process_block_numba(frame, lp_arr, bp_arr, env_arr, base_cutoff, env_depth, q, samplerate):
    n = frame.shape[0]
    out = np.empty(n, dtype=np.float32)
    attack = 0.0003
    release = 0.02
    a = _math.exp(-1.0 / (samplerate * attack))
    r = _math.exp(-1.0 / (samplerate * release))
    for i in range(n):
        x = frame[i]
        rect = abs(x)
        if rect > env_arr[0]:
            env_arr[0] = a * env_arr[0] + (1.0 - a) * rect
        else:
            env_arr[0] = r * env_arr[0]
        cutoff = base_cutoff + env_arr[0] * env_depth
        if cutoff < 20.0:
            cutoff = 20.0
        f = 2.0 * _math.sin(_math.pi * (cutoff / samplerate))
        hp = x - lp_arr[0] - bp_arr[0]
        bp_arr[0] += f * hp
        lp_arr[0] += f * bp_arr[0]
        q_clamped = q
        if q_clamped < 0.1:
            q_clamped = 0.1
        if q_clamped > 10.0:
            q_clamped = 10.0
        out[i] = lp_arr[0]
    return out

@njit
def compressor_process_block_numba(frame, threshold_db, ratio, makeup_db):
    n = frame.shape[0]
    out = np.empty(n, dtype=np.float32)
    thr = 10.0 ** (threshold_db / 20.0)
    makeup_lin = 10.0 ** (makeup_db / 20.0)
    gr_db = 0.0
    for i in range(n):
        x = frame[i]
        mag = abs(x) + 1e-12
        if mag > thr:
            excess = mag / thr
            gain = 1.0 / (excess ** (1.0 - (1.0/ratio)))
            if gain <= 0.0:
                gain = 1e-6
            out[i] = x * gain * makeup_lin
            gr_db = 20.0 * _math.log10(gain)
        else:
            out[i] = x * makeup_lin
    return out, gr_db

@njit
def limiter_block_numba(frame):
    n = frame.shape[0]
    peak = 1e-12
    for i in range(n):
        a = abs(frame[i])
        if a > peak:
            peak = a
    if peak > 1.0:
        gain = 1.0 / peak
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = frame[i] * gain
        gr_db = 20.0 * _math.log10(gain)
        return out, gr_db
    else:
        return frame, 0.0

# ----------------------------
# Acoustic model state
# (Modal Synthesis)
# ----------------------------
# Freq, Q, Gain for each mode, per preset
# These presets are designed to emulate different types of upright bass bodies.
ACOUSTIC_MODEL_PRESETS = {
    "Classic Violin-Corner": np.array([
        [62.0,  60.0, 1.0],   # Main air resonance (A0)
        [110.0, 50.0, 0.9],   # Main top resonance (T1)
        [195.0, 40.0, 0.7],
        [330.0, 35.0, 0.6],
        [490.0, 30.0, 0.5],
    ], dtype=np.float32),
    "Gamba-Style": np.array([
        [65.0,  55.0, 1.0],   # Slightly higher A0, more 'open'
        [115.0, 50.0, 0.85],
        [240.0, 38.0, 0.75],  # Different mid-range character
        [420.0, 33.0, 0.6],
        [580.0, 28.0, 0.5],
    ], dtype=np.float32),
    "German Flat-Back": np.array([
        [58.0,  70.0, 1.0],   # Lower, boomier A0 with high Q
        [95.0,  60.0, 0.95],  # Strong, low main wood resonance
        [180.0, 45.0, 0.8],
        [390.0, 30.0, 0.6],
    ], dtype=np.float32),
    "Soloist Carved-Back": np.array([
        [68.0,  50.0, 1.0],   # Tighter, more focused A0
        [125.0, 45.0, 0.9],   # Higher, articulate T1
        [260.0, 40.0, 0.7],
        [440.0, 35.0, 0.65],  # Clear upper-mids for melodic playing
        [620.0, 30.0, 0.5],
    ], dtype=np.float32),
}
MAX_MODES = max(p.shape[0] for p in ACOUSTIC_MODEL_PRESETS.values())
modal_filter_states = np.zeros((MAX_MODES, 4), dtype=np.float32) # x1, x2, y1, y2 for each filter

def modal_synthesis_fallback_py(frames, in_buf, out_buf, *args):
    """Placeholder python version if numba is not available."""
    if frames > 0:
        print("[jack_engine] Warning: Modal synthesis requires Numba, but it's not available. Bypassing.")
    out_buf[:frames] = in_buf[:frames]

# ----------------------------
# Granular pitch shifter (dual-grain, 50% overlap) - numba
# ----------------------------
# Parameters (tweakable)
GRAIN_SIZE = 1024
HOP = GRAIN_SIZE // 2
PITCH_RATIO = 0.5  # one octave down

# ring buffer for shifter
PITCH_BUF_LEN = 65536
pitch_buf = np.zeros(PITCH_BUF_LEN, dtype=np.float32)
pitch_write_idx = np.array([0], dtype=np.int32)

# read positions for two grains (float)
rpos0 = np.array([0.0], dtype=np.float32)
rpos1 = np.array([0.0], dtype=np.float32)
# counters for grain progress
cnt0 = np.array([GRAIN_SIZE], dtype=np.int32)  # force immediate grain start
cnt1 = np.array([GRAIN_SIZE], dtype=np.int32)
# window
hann_win = np.hanning(GRAIN_SIZE).astype(np.float32)
# position of next grain start relative to write_idx
# We'll schedule alternating grains every HOP samples
next_grain_trigger = np.array([0], dtype=np.int32)  # samples until next grain start

@njit
def _read_linear(buf, buf_len, pos):
    # pos is float >=0
    i0 = int(_math.floor(pos)) % buf_len
    i1 = (i0 + 1) % buf_len
    frac = pos - _math.floor(pos)
    return buf[i0] * (1.0 - frac) + buf[i1] * frac

@njit
def granular_pitch_shift_numba(in_frame, buf, buf_len, write_idx_arr,
                               rpos0_arr, rpos1_arr, cnt0_arr, cnt1_arr,
                               hann_window, grain_size, hop, ratio,
                               next_trigger_arr):
    n = in_frame.shape[0]
    out = np.zeros(n, dtype=np.float32)
    write_idx = write_idx_arr[0]
    r0 = rpos0_arr[0]
    r1 = rpos1_arr[0]
    c0 = cnt0_arr[0]
    c1 = cnt1_arr[0]
    next_t = next_trigger_arr[0]

    # We'll alternate grains: grain A active when c0<grain_size, grain B when c1<grain_size
    for i in range(n):
        x = in_frame[i]
        # write incoming to buffer
        buf[write_idx] = x
        write_idx += 1
        if write_idx >= buf_len:
            write_idx = 0

        # check trigger: if next_t == 0, start a new grain in the inactive slot
        if next_t <= 0:
            # choose which grain to start: if c0 >= grain_size -> start grain0 else start grain1
            if c0 >= grain_size:
                # start grain0: set read pos behind write_idx by grain_size (so reading older samples)
                start_pos = write_idx - grain_size
                if start_pos < 0:
                    start_pos += buf_len
                r0 = float(start_pos)
                c0 = 0
            elif c1 >= grain_size:
                start_pos = write_idx - grain_size
                if start_pos < 0:
                    start_pos += buf_len
                r1 = float(start_pos)
                c1 = 0
            # reset trigger to hop
            next_t = hop
        else:
            next_t -= 1

        s0 = 0.0
        s1 = 0.0
        w0 = 0.0
        w1 = 0.0

        # grain0 contribution
        if c0 < grain_size:
            # index into window
            win_idx = c0
            w0 = hann_window[win_idx]
            s0 = _read_linear(buf, buf_len, r0 % buf_len) * w0
            r0 += ratio
            c0 += 1
        # grain1 contribution
        if c1 < grain_size:
            win_idx = c1
            w1 = hann_window[win_idx]
            s1 = _read_linear(buf, buf_len, r1 % buf_len) * w1
            r1 += ratio
            c1 += 1

        # normalize by overlap (sum of windows can be up to ~1.0+ depending on overlap),
        # but for 50% Hann overlap the sum is nearly constant; we apply a simple scaling
        out[i] = s0 + s1

    # write back state
    write_idx_arr[0] = write_idx
    rpos0_arr[0] = r0
    rpos1_arr[0] = r1
    cnt0_arr[0] = c0
    cnt1_arr[0] = c1
    next_trigger_arr[0] = next_t
    return out

# Python fallback granular (simple)
def granular_pitch_shift_py(in_frame):
    global pitch_buf, pitch_write_idx, rpos0, rpos1, cnt0, cnt1, hann_win, next_grain_trigger
    n = in_frame.shape[0]
    out = np.zeros(n, dtype=np.float32)
    buf_len = pitch_buf.shape[0]
    write_idx = int(pitch_write_idx[0])
    r0 = float(rpos0[0])
    r1 = float(rpos1[0])
    c0 = int(cnt0[0])
    c1 = int(cnt1[0])
    next_t = int(next_grain_trigger[0])
    for i in range(n):
        x = in_frame[i]
        pitch_buf[write_idx] = x
        write_idx += 1
        if write_idx >= buf_len:
            write_idx = 0
        if next_t <= 0:
            if c0 >= GRAIN_SIZE:
                start_pos = write_idx - GRAIN_SIZE
                if start_pos < 0:
                    start_pos += buf_len
                r0 = float(start_pos)
                c0 = 0
            elif c1 >= GRAIN_SIZE:
                start_pos = write_idx - GRAIN_SIZE
                if start_pos < 0:
                    start_pos += buf_len
                r1 = float(start_pos)
                c1 = 0
            next_t = HOP
        else:
            next_t -= 1
        s0 = 0.0
        s1 = 0.0
        if c0 < GRAIN_SIZE:
            w0 = hann_win[c0]
            # linear read
            i0 = int(math.floor(r0)) % buf_len
            i1 = (i0 + 1) % buf_len
            frac = r0 - math.floor(r0)
            s0 = (pitch_buf[i0] * (1.0 - frac) + pitch_buf[i1] * frac) * w0
            r0 += PITCH_RATIO
            c0 += 1
        if c1 < GRAIN_SIZE:
            w1 = hann_win[c1]
            i0 = int(math.floor(r1)) % buf_len
            i1 = (i0 + 1) % buf_len
            frac = r1 - math.floor(r1)
            s1 = (pitch_buf[i0] * (1.0 - frac) + pitch_buf[i1] * frac) * w1
            r1 += PITCH_RATIO
            c1 += 1
        out[i] = s0 + s1
    pitch_write_idx[0] = write_idx
    rpos0[0] = r0
    rpos1[0] = r1
    cnt0[0] = c0
    cnt1[0] = c1
    next_grain_trigger[0] = next_t
    return out

# ----------------------------
# Python fallbacks for SVF/Comp/Limiter (kept minimal)
# ----------------------------
def envelope_follower_sample(x, st, attack=0.0003, release=0.02):
    a = math.exp(-1.0 / (SAMPLERATE * max(attack, 1e-6)))
    r = math.exp(-1.0 / (SAMPLERATE * max(release, 1e-6)))
    rect = abs(x)
    if rect > st["env"]:
        st["env"] = a * st["env"] + (1 - a) * rect
    else:
        st["env"] = r * st["env"] + (1 - r) * rect
    return st["env"]

def svf_process_block_py(frame, st, base_cutoff, env_depth, q):
    out = np.empty_like(frame)
    for i, x in enumerate(frame):
        env = envelope_follower_sample(x, st, attack=0.0003, release=0.05)
        cutoff = max(20.0, base_cutoff + env * env_depth)
        f = 2.0 * math.sin(math.pi * (cutoff / SAMPLERATE))
        hp = x - st["lp"] - st["bp"]
        st["bp"] += f * hp
        st["lp"] += f * st["bp"]
        q_clamped = max(0.1, min(q, 10.0))
        out[i] = st["lp"] * (1.0 + (q_clamped - 0.8) * 0.1)
    return out

def compressor_process_block_py(frame, threshold_db, ratio, makeup_db):
    out = np.empty_like(frame)
    thr = 10.0 ** (threshold_db / 20.0)
    makeup_lin = 10.0 ** (makeup_db / 20.0)
    gr_db = 0.0
    for i, x in enumerate(frame):
        mag = abs(x) + 1e-12
        if mag > thr:
            excess = mag / thr
            gain = 1.0 / (excess ** (1.0 - (1.0/ratio)))
            if gain <= 0.0:
                gain = 1e-6
            out[i] = x * gain * makeup_lin
            gr_db = 20.0 * math.log10(gain)
        else:
            out[i] = x * makeup_lin
    return out, gr_db

def limiter_block_py(frame):
    peak = np.max(np.abs(frame))
    if peak > 1.0:
        gain = 1.0 / peak
        return frame * gain, 20.0 * math.log10(gain)
    return frame, 0.0

# ----------------------------
# State adapters & warm-up
# ----------------------------
# SVF state arrays
svf_lp_l = np.array([0.0], dtype=np.float32)
svf_bp_l = np.array([0.0], dtype=np.float32)
svf_env_l = np.array([0.0], dtype=np.float32)
svf_lp_r = np.array([0.0], dtype=np.float32)
svf_bp_r = np.array([0.0], dtype=np.float32)
svf_env_r = np.array([0.0], dtype=np.float32)

# pitch shifter state arrays are defined earlier: pitch_buf, pitch_write_idx, rpos0, rpos1, cnt0, cnt1, next_grain_trigger
next_grain_trigger = np.array([0], dtype=np.int32)

USE_NUMBA_KERNELS = False
if NUMBA_AVAILABLE:
    try:
        dummy = np.zeros(64, dtype=np.float32)
        test_inputs = np.zeros((NUM_STRINGS, 64), dtype=np.float32)
        gains_test = np.array([float(DSP_PARAMS[0]), float(DSP_PARAMS[1])], dtype=np.float32)
        _ = comb_process_all_numba(test_inputs, bufs, lens, idxs, gains_test)
        _ = svf_process_block_numba(dummy, svf_lp_l, svf_bp_l, svf_env_l, float(EFFECT_PARAMS['svf_base_cutoff']), float(EFFECT_PARAMS['svf_env_depth']), float(EFFECT_PARAMS['svf_q']), float(SAMPLERATE))
        _ = compressor_process_block_numba(dummy, float(EFFECT_PARAMS['comp_threshold']), float(EFFECT_PARAMS['comp_ratio']), float(EFFECT_PARAMS['comp_makeup']))
        _ = limiter_block_numba(dummy)
        _ = granular_pitch_shift_numba(dummy, pitch_buf, pitch_buf.shape[0], pitch_write_idx, rpos0, rpos1, cnt0, cnt1, hann_win, GRAIN_SIZE, HOP, PITCH_RATIO, next_grain_trigger)
        if dsp.process_modal_synthesis_numba:
            dummy_coeffs = np.zeros((MAX_MODES, 5), dtype=np.float32)
            dummy_gains = np.zeros(MAX_MODES, dtype=np.float32)
            dsp.process_modal_synthesis_numba(
                dummy.shape[0], dummy, dummy, modal_filter_states, dummy_coeffs, dummy_gains, 0.15, 0.85
            )

        USE_NUMBA_KERNELS = True
        print("[jack_engine] numba kernels warmed up")
    except Exception as e:
        USE_NUMBA_KERNELS = False
        print("[jack_engine] numba warmup failed, falling back to Python kernels:", e)

# ----------------------------
# Main process callback
# ----------------------------
_last_log_time = 0.0

svf_state_l = {"lp":0.0, "bp":0.0, "env":0.0}
svf_state_r = {"lp":0.0, "bp":0.0, "env":0.0}

def _process_callback(frames):
    global LAST_PEAK_DB, LAST_COMP_GR_DB, LAST_LIMITER_GR_DB, _last_log_time
    if not IN_JACK:
        return

    # read inputs
    inputs = []
    for p in in_ports:
        try:
            arr = p.get_array()
        except Exception:
            arr = None
        if arr is None or len(arr) != frames:
            arr = np.zeros(frames, dtype=np.float32)
        inputs.append(arr)

    # mirror first input if others silent
    try:
        mags = [float(np.max(np.abs(x))) for x in inputs]
        if mags[0] > 1e-6 and all(m <= 1e-6 for m in mags[1:]):
            for j in range(1, len(inputs)):
                inputs[j] = inputs[0]
    except Exception:
        pass

    # prepare matrix for comb processor
    in_mat = np.zeros((NUM_STRINGS, frames), dtype=np.float32)
    for s in range(NUM_STRINGS):
        in_mat[s,:] = inputs[s] if s < len(inputs) else np.zeros(frames, dtype=np.float32)

    # --- Acoustic Model OR Standard Effects ---
    if ACOUSTIC_MODE_ENABLED:
        # For acoustic mode, we use a static virtual pickup configuration, ignoring the UI.
        # Pickup 1 is at 350mm, Pickup 2 is disabled.
        acoustic_pickup_dists = [350.0, 40.0] # P2 dist doesn't matter as type is 'none'
        acoustic_pickup_types = ["single", "none"]
        
        # Compute temporary comb delays for this static configuration
        static_combs = compute_comb_delays_from_mm(acoustic_pickup_dists, acoustic_pickup_types, SAMPLERATE)
        static_lens = np.array(static_combs, dtype=np.int32)
        
        # Ensure our main buffer is large enough for these static delays
        max_static_delay = np.max(static_lens)
        _ensure_buffer_size(max_static_delay)
        
        # Use only pickup 1 volume, pickup 2 is off.
        acoustic_gains = np.array([float(DSP_PARAMS[0]), 0.0], dtype=np.float32)

        # Run comb processing with the static acoustic configuration
        if USE_NUMBA_KERNELS:
            combined = comb_process_all_numba(in_mat, bufs, static_lens, idxs, acoustic_gains)
        else:
            combined = comb_process_all_numba(in_mat, bufs, static_lens, idxs, acoustic_gains)

        out_l = combined.copy()
        out_r = combined.copy()

        # Now, process the result through the modal synthesizer
        body_size = float(EFFECT_PARAMS.get("acoustic_body_size", 0.5))
        tone = float(EFFECT_PARAMS.get("acoustic_tone", 0.5))
        mix = float(EFFECT_PARAMS.get("acoustic_mix", 0.85))
        preset_name = EFFECT_PARAMS.get("acoustic_model_preset", "Classic Violin-Corner")

        # Get modes from the selected preset
        base_modes = ACOUSTIC_MODEL_PRESETS.get(preset_name, ACOUSTIC_MODEL_PRESETS["Classic Violin-Corner"])
        num_modes = base_modes.shape[0]
        
        # Calculate filter coefficients on the fly based on UI params
        freq_scale = 0.8 + (0.4 * body_size) # Scale frequencies by +/- 20%
        q_scale = 0.5 + (1.5 * tone) # Scale Q-factor

        filter_coeffs = np.zeros((num_modes, 5), dtype=np.float32)
        filter_gains = np.zeros(num_modes, dtype=np.float32)

        for i in range(num_modes):
            freq = base_modes[i, 0] * freq_scale
            q = base_modes[i, 1] * q_scale
            filter_gains[i] = base_modes[i, 2]

            w0 = 2.0 * math.pi * freq / SAMPLERATE
            alpha = math.sin(w0) / (2.0 * q)
            
            # Band-pass filter coefficients
            filter_coeffs[i, 0] = alpha       # b0
            filter_coeffs[i, 1] = 0.0         # b1
            filter_coeffs[i, 2] = -alpha      # b2
            filter_coeffs[i, 3] = -2.0 * math.cos(w0) # a1
            filter_coeffs[i, 4] = 1.0 - alpha # a2
            # Normalize by a0 = 1 + alpha
            filter_coeffs[i, :] /= (1.0 + alpha)

        mono_in = 0.5 * (out_l + out_r)
        _ensure_acoustic_buffer(frames)
        
        kernel = dsp.process_modal_synthesis_numba if dsp.process_modal_synthesis_numba else modal_synthesis_fallback_py
        kernel(frames, mono_in, acoustic_out_buffer, modal_filter_states, filter_coeffs, filter_gains, 1.0 - mix, mix)
        out_l = acoustic_out_buffer[:frames]
        out_r = acoustic_out_buffer[:frames].copy()

    else:
        # Standard processing path using UI-defined pickup settings
        gains = np.array([float(DSP_PARAMS[0]), float(DSP_PARAMS[1])], dtype=np.float32)

        # Comb processing with dynamic delays from UI
        if USE_NUMBA_KERNELS:
            try:
                combined = comb_process_all_numba(in_mat, bufs, lens, idxs, gains)
            except Exception as e:
                print("[jack_engine] comb_process_all_numba error:", e)
                combined = np.zeros(frames, dtype=np.float32)
        else:
            combined = comb_process_all_numba(in_mat, bufs, lens, idxs, gains)

        out_l = combined.copy()
        out_r = combined.copy()
        # Standard Effects Chain (SVF, Octaver)
        if not EFFECTS_BYPASS.get("env_filter", True):
            base = EFFECT_PARAMS.get("svf_base_cutoff", 1000.0)
            depth = EFFECT_PARAMS.get("svf_env_depth", 0.0)
            q = EFFECT_PARAMS.get("svf_q", 0.8)
            if USE_NUMBA_KERNELS:
                out_l = svf_process_block_numba(out_l, svf_lp_l, svf_bp_l, svf_env_l, float(base), float(depth), float(q), float(SAMPLERATE))
                out_r = svf_process_block_numba(out_r, svf_lp_r, svf_bp_r, svf_env_r, float(base), float(depth), float(q), float(SAMPLERATE))
            else:
                out_l = svf_process_block_py(out_l, svf_state_l, base, depth, q)
                out_r = svf_process_block_py(out_r, svf_state_r, base, depth, q)

        if not EFFECTS_BYPASS.get("octaver", True):
            dry = EFFECT_PARAMS.get("oct_dry", 1.0)
            subg = EFFECT_PARAMS.get("oct_sub_gain", 0.0)
            mono = 0.5 * (out_l + out_r)
            if USE_NUMBA_KERNELS:
                sub = granular_pitch_shift_numba(mono, pitch_buf, pitch_buf.shape[0], pitch_write_idx,
                                                 rpos0, rpos1, cnt0, cnt1, hann_win, GRAIN_SIZE, HOP, PITCH_RATIO, next_grain_trigger)
            else:
                sub = granular_pitch_shift_py(mono)
            out_l = dry * mono + subg * sub
            out_r = out_l.copy()

    # Compressor
    if not EFFECTS_BYPASS.get("comp", True):
        thr = EFFECT_PARAMS.get("comp_threshold", -20.0)
        ratio = EFFECT_PARAMS.get("comp_ratio", 4.0)
        makeup_db = EFFECT_PARAMS.get("comp_makeup", 0.0)
        if USE_NUMBA_KERNELS:
            out_l, gr_l = compressor_process_block_numba(out_l, float(thr), float(ratio), float(makeup_db))
            out_r, gr_r = compressor_process_block_numba(out_r, float(thr), float(ratio), float(makeup_db))
            LAST_COMP_GR_DB = max(float(gr_l), float(gr_r))
        else:
            out_l, gr_l = compressor_process_block_py(out_l, thr, ratio, makeup_db)
            out_r, gr_r = compressor_process_block_py(out_r, thr, ratio, makeup_db)
            LAST_COMP_GR_DB = max(gr_l, gr_r)

    # Master gain
    master_gain = float(DSP_PARAMS[3]) if DSP_PARAMS is not None else 1.0
    out_l *= master_gain
    out_r *= master_gain

    # Limiter
    if USE_NUMBA_KERNELS:
        out_l, lim_gr_l = limiter_block_numba(out_l)
        out_r, lim_gr_r = limiter_block_numba(out_r)
        LAST_LIMITER_GR_DB = max(float(lim_gr_l), float(lim_gr_r))
    else:
        out_l, lim_gr_l = limiter_block_py(out_l)
        out_r, lim_gr_r = limiter_block_py(out_r)
        LAST_LIMITER_GR_DB = max(lim_gr_l, lim_gr_r)

    # Meters
    pk = max(np.max(np.abs(out_l)), np.max(np.abs(out_r)), 1e-12)
    LAST_PEAK_DB = 20.0 * math.log10(pk + 1e-12)

    # write outputs
    try:
        outL.get_array()[:] = out_l
        outR.get_array()[:] = out_r
    except Exception as e:
        now = time.time()
        if now - _last_log_time > 1.0:
            _last_log_time = now
            print("[jack_engine] output write failed:", e)

    now = time.time()
    if now - _last_log_time > 1.0:
        _last_log_time = now
        try:
            inp_peaks = ["{:.6f}".format(float(np.max(np.abs(x)))) for x in inputs]
        except Exception:
            inp_peaks = []
        print("[jack_engine] input peaks:", inp_peaks, "out peak {:.6f}".format(pk))

# install callback
if IN_JACK and client is not None:
    client.set_process_callback(_process_callback)
    print("[jack_engine] process callback installed")

# ----------------------------
# API helpers
# ----------------------------
def start_jack_client():
    if IN_JACK and client is not None:
        if INSTR_MODEL.get("comb_delays_samples") is not None:
            set_comb_delays(INSTR_MODEL["comb_delays_samples"])
        client.activate()
        print("[jack_engine] JACK client activated")
        return True
    print("[jack_engine] JACK not available")
    return False

def run_dummy_audio_loop(samplerate=48000, frames=128):
    print("[jack_engine] running dummy audio loop (no JACK)")
    while True:
        time.sleep(0.1)

def get_instrument_state():
    return {
        "num_strings": INSTR_MODEL.get("num_strings", NUM_STRINGS),
        "pickup_types": INSTR_MODEL.get("pickup_types"),
        "closest_distance_mm_per_pickup": INSTR_MODEL.get("closest_distance_mm_per_pickup"),
        "comb_delays_samples": INSTR_MODEL.get("comb_delays_samples"),
        "dsp_params": DSP_PARAMS.tolist(),
    }

def apply_instrument_state(state, samplerate=None):
    try:
        if "pickup_types" in state:
            INSTR_MODEL["pickup_types"] = state["pickup_types"]
        if "closest_distance_mm_per_pickup" in state:
            INSTR_MODEL["closest_distance_mm_per_pickup"] = state["closest_distance_mm_per_pickup"]
        if "comb_delays_samples" in state and state["comb_delays_samples"] is not None:
            delays = state["comb_delays_samples"]
        else:
            delays = compute_comb_delays_from_mm(
                INSTR_MODEL["closest_distance_mm_per_pickup"],
                INSTR_MODEL["pickup_types"],
                SAMPLERATE if samplerate is None else samplerate
            )
        set_comb_delays(delays)
        if "dsp_params" in state:
            dp = state["dsp_params"]
            try:
                DSP_PARAMS[0] = float(dp[0])
                DSP_PARAMS[1] = float(dp[1])
                DSP_PARAMS[3] = float(dp[3])
            except Exception:
                pass
        return True
    except Exception as e:
        print("apply_instrument_state error:", e)
        return False

def _atomic_write(path: Path, obj):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)

def get_full_state(force_full_update=False):
    """Return a comprehensive state dictionary for the UI."""
    state = {
        'p1': DSP_PARAMS[0],
        'p2': DSP_PARAMS[1],
        'mg': DSP_PARAMS[3],
        'peak_db': LAST_PEAK_DB,
        'limiter_gr_db': LAST_LIMITER_GR_DB,
        'comp_gr_db': LAST_COMP_GR_DB,
        'cpu_load': 0.0, # Placeholder
    }
    state.update(EFFECT_PARAMS)
    state.update(EFFECTS_BYPASS)
    state['acoustic_mode'] = ACOUSTIC_MODE_ENABLED

    # Octaver mix is derived
    state['oct_mix'] = EFFECT_PARAMS.get('oct_sub_gain', 0.0)

    # Pickup state
    state['p1_type'] = INSTR_MODEL['pickup_types'][0]
    state['p2_type'] = INSTR_MODEL['pickup_types'][1]
    state['p1_dist'] = INSTR_MODEL['closest_distance_mm_per_pickup'][0]
    state['p2_dist'] = INSTR_MODEL['closest_distance_mm_per_pickup'][1]

    # Add available acoustic presets for the UI dropdown
    state['acoustic_model_presets'] = list(ACOUSTIC_MODEL_PRESETS.keys())

    if force_full_update:
        user_presets = sorted([p.stem for p in PRESET_DIR.glob("*.json") if p.is_file() and p.parent==PRESET_DIR])
        factory_presets = sorted([p.stem for p in FACTORY_DIR.glob("*.json") if p.is_file()])
        state['user_presets'] = [u for u in user_presets if u not in set(factory_presets)]
        state['factory_presets'] = factory_presets

    return state

def update_controls(data):
    """Update DSP_PARAMS from a dictionary."""
    if 'p1' in data: DSP_PARAMS[0] = float(data['p1'])
    if 'p2' in data: DSP_PARAMS[1] = float(data['p2'])
    if 'mg' in data: DSP_PARAMS[3] = float(data['mg'])

def update_effects_params(data):
    """Update EFFECT_PARAMS from a dictionary."""
    param = data.get("param")
    value = data.get("value")
    if param in EFFECT_PARAMS:
        if isinstance(value, str):
            EFFECT_PARAMS[param] = value
        else:
            EFFECT_PARAMS[param] = float(value)
    elif param == 'oct_mix': # Special handling for oct_mix slider
        EFFECT_PARAMS['oct_dry'] = 1.0 - float(value)
        EFFECT_PARAMS['oct_sub_gain'] = float(value)

def set_bypass(data):
    """Update EFFECTS_BYPASS from a dictionary."""
    name = data.get("name")
    state = data.get("state")
    if name in EFFECTS_BYPASS:
        EFFECTS_BYPASS[name] = bool(state)
    elif name == 'acoustic_mode':
        global ACOUSTIC_MODE_ENABLED
        ACOUSTIC_MODE_ENABLED = bool(state)
        # When enabling acoustic mode, also bypass the other effects
        if ACOUSTIC_MODE_ENABLED:
            EFFECTS_BYPASS['env_filter'] = True
            EFFECTS_BYPASS['octaver'] = True

def set_pickup_distance(data):
    """Update pickup distance and recompute comb delays."""
    slot = int(data['pickup_slot'])
    mm = float(data['distance_mm'])
    INSTR_MODEL['closest_distance_mm_per_pickup'][slot] = mm
    combs = compute_comb_delays_from_mm(
        INSTR_MODEL["closest_distance_mm_per_pickup"],
        INSTR_MODEL["pickup_types"],
        SAMPLERATE
    )
    set_comb_delays(combs)

def set_pickup_type(data):
    """Update pickup type and recompute comb delays."""
    slot = int(data['pickup_slot'])
    ptype = data['type']
    INSTR_MODEL['pickup_types'][slot] = ptype
    combs = compute_comb_delays_from_mm(
        INSTR_MODEL["closest_distance_mm_per_pickup"],
        INSTR_MODEL["pickup_types"],
        SAMPLERATE
    )
    set_comb_delays(combs)
    # If disabling pickup 2, also zero its volume
    if slot == 1 and ptype == 'none':
        DSP_PARAMS[1] = 0.0

def load_preset_file(path):
    """Loads a preset from a JSON file path."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read preset file {path}: {e}")
        return None

def load_preset(name):
    """Load a preset by name from user or factory directories."""
    user_path = PRESET_DIR / f"{name}.json"
    factory_path = FACTORY_DIR / f"{name}.json"
    preset_path = user_path if user_path.exists() else (factory_path if factory_path.exists() else None)
    if preset_path is None:
        return False

    inst = load_preset_file(preset_path)
    if not inst:
        return False

    # User presets can contain effects settings
    if 'effects' in inst and preset_path.parent != FACTORY_DIR:
        effects_block = inst['effects']
        if 'effects_params' in effects_block:
            for k, v in effects_block['effects_params'].items():
                if k in EFFECT_PARAMS:
                    EFFECT_PARAMS[k] = float(v)
        if 'effects_bypass' in effects_block:
            for k, v in effects_block['effects_bypass'].items():
                if k in EFFECTS_BYPASS:
                    EFFECTS_BYPASS[k] = bool(v)

    ok = apply_instrument_state(inst, samplerate=SAMPLERATE)
    if ok:
        try:
            LAST_PRESET_FILE.write_text(name, encoding='utf-8')
        except Exception as e:
            print(f"Warning: could not write to last_preset.txt: {e}")
    return ok

def save_preset(name):
    """Save the current state as a user preset."""
    if not name: return False
    preset_path = PRESET_DIR / f"{name}.json"

    inst = get_instrument_state()
    effects_block = {
        "effects_params": {k: float(v) for k, v in EFFECT_PARAMS.items()},
        "effects_bypass": {k: bool(v) for k, v in EFFECTS_BYPASS.items()}
    }
    preset = {
        "name": name,
        "dsp_params": inst.get("dsp_params"),
        "pickup_types": inst.get("pickup_types"),
        "closest_distance_mm_per_pickup": inst.get("closest_distance_mm_per_pickup"),
        "comb_delays_samples": inst.get("comb_delays_samples"),
        "num_strings": inst.get("num_strings"),
        "effects": effects_block
    }
    try:
        _atomic_write(preset_path, preset)
        LAST_PRESET_FILE.write_text(name, encoding='utf-8')
        return True
    except Exception as e:
        print(f"Error saving preset {name}: {e}")
        return False

def delete_preset(name):
    """Delete a user preset."""
    preset_path = PRESET_DIR / f"{name}.json"
    if preset_path.exists() and preset_path.parent == PRESET_DIR:
        try:
            preset_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting preset {name}: {e}")
            return False
    return False

def save_global_effects():
    """Saves the current compressor settings to a global file."""
    try:
        settings = {
            "comp_bypass": EFFECTS_BYPASS.get("comp", False),
            "comp_params": {
                "comp_threshold": EFFECT_PARAMS.get("comp_threshold", -20.0),
                "comp_ratio": EFFECT_PARAMS.get("comp_ratio", 4.0),
                "comp_makeup": EFFECT_PARAMS.get("comp_makeup", 0.0),
            }
        }
        _atomic_write(GLOBAL_EFFECTS_FILE, settings)
        print("[jack_engine] Global compressor settings saved.")
        return True
    except Exception as e:
        print(f"Error saving global effects: {e}")
        return False

# expose flags
NUMBA_AVAILABLE = NUMBA_AVAILABLE
USE_NUMBA_KERNELS = USE_NUMBA_KERNELS
LAST_PEAK_DB = LAST_PEAK_DB
LAST_COMP_GR_DB = LAST_COMP_GR_DB
LAST_LIMITER_GR_DB = LAST_LIMITER_GR_DB
