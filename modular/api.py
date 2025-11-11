#!/usr/bin/env python3
"""
api.py - Flask + SocketIO API to control jackbasssim

This file is written to be compatible with the current jack_engine.py you've loaded:
it relies on je.get_instrument_state(), je.apply_instrument_state(), je.SAMPLERATE,
je.EFFECT_PARAMS and je.EFFECTS_BYPASS where possible, but it will fall back to
reasonable defaults if something is missing.
"""

import os
import json
import time
from pathlib import Path
from threading import Thread, Lock

from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO

import jack_engine as je
from ui_templates import HTML_TEMPLATE

APP = Flask(__name__)
APP.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret')
SOCKET = SocketIO(APP, cors_allowed_origins="*", async_mode='threading')

ROOT = Path(__file__).resolve().parent
PRESET_DIR = ROOT / "presets"
FACTORY_DIR = PRESET_DIR / "factory"
GLOBAL_EFFECTS_FILE = PRESET_DIR / "global_effects.json"
LAST_PRESET_FILE = PRESET_DIR / "last_preset.txt"

PRESET_DIR.mkdir(exist_ok=True)
FACTORY_DIR.mkdir(exist_ok=True)

fs_lock = Lock()

# --------------------------
# Utilities
# --------------------------
def _atomic_write(path: Path, obj):
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)

# local helper: compute comb delays in samples from distances in mm
# Uses simple model: for each string frequency, samples = round((distance_mm / wavelength_mm) * period_samples)
# wavelength_mm = 343000 / freq (sound speed approx 343 m/s)
STRING_FREQS = [30.8677, 41.2034, 55.0, 73.416, 97.999, 130.813]  # B E A D G C

def mm_to_samples(distance_mm, freq, samplerate):
    if freq <= 0:
        return 1
    wavelength_mm = 343000.0 / freq
    frac = distance_mm / wavelength_mm
    period_samples = samplerate / freq
    delay = int(round(frac * period_samples))
    return max(1, delay)

def compute_comb_delays_samples_from_mm(closest_mm_per_pickup, pickup_types, samplerate):
    # pickup_types arg ignored for widths here; we'll assume widths based on type for better spacing
    PICKUP_WIDTHS = {
        "single": 24.0,
        "splitP": 56.0,
        "humbucker": 49.0,
        "soapbar": 34.0,
        "none": 0.0,
    }
    nstrings = len(STRING_FREQS)
    combs = []
    # ensure lists are length 2
    if not isinstance(closest_mm_per_pickup, (list,tuple)):
        closest_mm_per_pickup = [float(closest_mm_per_pickup), float(closest_mm_per_pickup)]
    if len(closest_mm_per_pickup) < 2:
        closest_mm_per_pickup = [closest_mm_per_pickup[0], closest_mm_per_pickup[0]]
    if not isinstance(pickup_types, (list,tuple)):
        pickup_types = [pickup_types, pickup_types]
    if len(pickup_types) < 2:
        pickup_types = [pickup_types[0], pickup_types[0]]

    for si in range(nstrings):
        f = STRING_FREQS[si]
        p1_mm = float(closest_mm_per_pickup[0])
        p2_mm = float(closest_mm_per_pickup[1])
        w1 = PICKUP_WIDTHS.get(pickup_types[0], 24.0)
        w2 = PICKUP_WIDTHS.get(pickup_types[1], 24.0)
        p1_positions = [p1_mm, p1_mm + 0.5*w1, p1_mm + w1]
        p2_positions = [p2_mm, p2_mm + 0.5*w2, p2_mm + w2]
        p1_samples = [mm_to_samples(pos, f, samplerate) for pos in p1_positions]
        p2_samples = [mm_to_samples(pos, f, samplerate) for pos in p2_positions]
        combs.append([p1_samples, p2_samples])
    return combs

# --------------------------
# Global effects persistence
# --------------------------
def load_global_effects():
    if not GLOBAL_EFFECTS_FILE.exists():
        return None
    try:
        with open(GLOBAL_EFFECTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print("load_global_effects failed:", e)
        return None

def save_global_effects(obj):
    try:
        _atomic_write(GLOBAL_EFFECTS_FILE, obj)
        return True
    except Exception as e:
        print("save_global_effects failed:", e)
        return False

# Apply persisted global effects at startup (if present)
g = load_global_effects()
if g:
    try:
        # apply general effect params (svf/oct) if present
        if 'effects_params' in g:
            for k,v in g['effects_params'].items():
                if k in je.EFFECT_PARAMS:
                    je.EFFECT_PARAMS[k] = float(v)
        # apply compressor params and bypass
        if 'comp_params' in g:
            for k,v in g['comp_params'].items():
                if k in je.EFFECT_PARAMS:
                    je.EFFECT_PARAMS[k] = float(v)
        if 'comp_bypass' in g:
            je.EFFECTS_BYPASS['comp'] = bool(g['comp_bypass'])
    except Exception as e:
        print("apply_global_effects failed:", e)


# --------------------------
# Factory presets (create defaults if absent)
# --------------------------
def ensure_factory_presets(samplerate):
    # only create if missing
    default_defs = {
        "Precision Bass": {"pickup_types":["splitP","none"], "closest_distance_mm_per_pickup":[70.0, 0.0]},
        "60s Jazz Bass": {"pickup_types":["single","single"], "closest_distance_mm_per_pickup":[74.0, 32.0]},
        "MusicMan Stingray": {"pickup_types":["humbucker","none"], "closest_distance_mm_per_pickup":[17.0, 0.0]},
        "70s Jazz Bass": {"pickup_types":["single","single"], "closest_distance_mm_per_pickup":[74.0, 28.0]},
        "Hofner 'Beatle' Bass": {"pickup_types":["single","single"], "closest_distance_mm_per_pickup":[125.0, 28.0]},
        "Warwick Thumb Bass": {"pickup_types":["soapbar","soapbar"], "closest_distance_mm_per_pickup":[25.0, 8.0]},
    }
    for name, info in default_defs.items():
        p = FACTORY_DIR / f"{name}.json"
        if p.exists():
            continue
        try:
            combs = compute_comb_delays_samples_from_mm(info["closest_distance_mm_per_pickup"], info["pickup_types"], samplerate)
            preset = {
                "name": name,
                "dsp_params": [1.0, 1.0, 20000.0, 1.0],
                "pickup_types": info["pickup_types"],
                "closest_distance_mm_per_pickup": info["closest_distance_mm_per_pickup"],
                "comb_delays_samples": combs,
                "num_strings": len(combs)
            }
            _atomic_write(p, preset)
        except Exception as e:
            print("ensure_factory_presets: failed to write", p, e)


# ensure factory presets exist using engine samplerate if available
samplerate = getattr(je, 'SAMPLERATE', 48000)
ensure_factory_presets(samplerate)

# --------------------------
# Flask routes
# --------------------------
@APP.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@APP.route('/api/get_state')
def get_state():
    """
    Returns:
    {
      instrument_state: { num_strings, pickup_types, closest_distance_mm_per_pickup, comb_delays_samples, ... },
      dsp_params: [...],   # backwards-compatible small array (p1,p2,tone,master)
      effects: {..},      # EFFECT_PARAMS
      effects_bypass: {..}, # EFFECTS_BYPASS
      peak_db, comp_gr_db, limiter_gr_db, numba
    }
    """
    try:
        inst = je.get_instrument_state() if hasattr(je, 'get_instrument_state') else {
            "num_strings": getattr(je, "NUM_STRINGS", 6),
            "pickup_types": getattr(je, "INSTR_MODEL", {}).get("pickup_types", ["single","single"]),
            "closest_distance_mm_per_pickup": getattr(je, "INSTR_MODEL", {}).get("closest_distance_mm_per_pickup", [40.0,40.0]),
            "comb_delays_samples": getattr(je, "INSTR_MODEL", {}).get("comb_delays_samples", None)
        }

        # dsp_params back-compat (UI expects indexes: 0=p1vol,1=p2vol,2=tone,3=master)
        # If engine doesn't expose DSP_PARAMS, synthesize reasonable defaults from EFFECT_PARAMS
        try:
            dsp = getattr(je, 'DSP_PARAMS')
            dsp_list = list(dsp) if hasattr(dsp, 'tolist') else list(dsp)
        except Exception:
            dsp_list = [
                float(1.0),                         # pickup1 vol default
                float(1.0),                         # pickup2 vol default
                float(je.EFFECT_PARAMS.get('svf_base_cutoff', 20000.0)),
                float(1.0)
            ]

        response = {
            "instrument_state": inst,
            "dsp_params": dsp_list,
            "effects": je.EFFECT_PARAMS if hasattr(je, 'EFFECT_PARAMS') else {},
            "effects_bypass": je.EFFECTS_BYPASS if hasattr(je, 'EFFECTS_BYPASS') else {},
            "peak_db": float(getattr(je, 'LAST_PEAK_DB', -120.0)),
            "comp_gr_db": float(getattr(je, 'LAST_COMP_GR_DB', 0.0)),
            "limiter_gr_db": float(getattr(je, 'LAST_LIMITER_GR_DB', 0.0)),
            "numba": bool(getattr(je, 'NUMBA_AVAILABLE', False))
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@APP.route('/api/get_instrument_params')
def get_instrument_params():
    try:
        inst = je.get_instrument_state() if hasattr(je, 'get_instrument_state') else {}
        return jsonify(inst)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@APP.route('/api/set_pickup_distance', methods=['POST'])
def set_pickup_distance():
    data = request.get_json(force=True)
    slot = data.get("pickup_slot")
    mm = data.get("distance_mm")

    if slot is None or mm is None:
        return jsonify({"error":"missing fields"}), 400

    try:
        slot = int(slot)
        mmf = float(mm)

        with fs_lock:
            instr = getattr(je, 'INSTR_MODEL', None)
            if instr is None:
                return jsonify({"error": "engine missing INSTR_MODEL"}), 500

            # Ensure list length
            cur = instr.get("closest_distance_mm_per_pickup", [40.0, 40.0])
            if len(cur) < 2:
                cur = [float(cur[0]), float(cur[0])]

            # Update selected pickup
            cur[slot] = mmf
            instr["closest_distance_mm_per_pickup"] = cur

            # Compute new comb delays (correct function name!)
            samplerate = getattr(je, 'SAMPLERATE', 48000)
            combs = je.compute_comb_delays_from_mm(
                instr["closest_distance_mm_per_pickup"],
                instr.get("pickup_types", ["single","single"]),
                samplerate
            )
            instr["comb_delays_samples"] = combs

            # Apply to engine
            if hasattr(je, 'set_comb_delays'):
                je.set_comb_delays(combs)
            elif hasattr(je, 'apply_instrument_state'):
                try:
                    je.apply_instrument_state(instr)
                except Exception as e:
                    print("Warning: apply_instrument_state failed:", e)

        return jsonify({"status":"ok", "pickup_slot": slot, "distance_mm": mmf})

    except Exception as e:
        print("ERROR in set_pickup_distance:", e)
        return jsonify({"error": str(e)}), 500

@APP.route('/api/set_pickup_type', methods=['POST'])
def set_pickup_type():
    data = request.get_json(force=True)
    slot = data.get("pickup_slot")
    ptype = data.get("type")
    if slot is None or ptype is None:
        return jsonify({"error":"missing fields"}), 400
    try:
        slot = int(slot)
        with fs_lock:
            instr = getattr(je, 'INSTR_MODEL', None)
            if instr is None:
                return jsonify({"error":"engine has no INSTR_MODEL"}), 500
            pts = instr.get("pickup_types", ["single","single"])
            if len(pts) < 2:
                pts = [pts[0] if len(pts)>0 else "single", pts[0] if len(pts)>0 else "single"]
            pts[slot] = ptype
            instr["pickup_types"] = pts
            # recompute combs
            samplerate = getattr(je, 'SAMPLERATE', samplerate)
            combs = compute_comb_delays_samples_from_mm(instr.get("closest_distance_mm_per_pickup",[40.0,40.0]), pts, samplerate)
            instr["comb_delays_samples"] = combs
            if hasattr(je, 'set_comb_delays'):
                je.set_comb_delays(combs)
            elif hasattr(je, 'apply_instrument_state'):
                try:
                    je.apply_instrument_state(instr)
                except Exception:
                    pass
            # if disabling second pickup, set DSP gain 0 if available
            if slot == 1 and ptype == "none" and hasattr(je, 'DSP_PARAMS'):
                try:
                    je.DSP_PARAMS[1] = 0.0
                except Exception:
                    pass
        return jsonify({"status":"ok","pickup_slot":slot,"type":ptype})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@APP.route('/api/set_pickup_enabled', methods=['POST'])
def set_pickup_enabled():
    data = request.get_json(force=True)
    enabled = bool(data.get("enabled", True))
    try:
        with fs_lock:
            instr = getattr(je, 'INSTR_MODEL', None)
            if instr is None:
                return jsonify({"error":"engine has no INSTR_MODEL"}), 500
            pts = instr.get("pickup_types", ["single","single"])
            cur = instr.get("closest_distance_mm_per_pickup", [40.0,40.0])
            if enabled:
                if pts[1] == "none":
                    pts[1] = "single"
                if cur[1] == 0.0:
                    cur[1] = 40.0
                # restore DSP gain if present
                if hasattr(je, 'DSP_PARAMS'):
                    try: je.DSP_PARAMS[1] = 1.0
                    except Exception: pass
            else:
                pts[1] = "none"
                cur[1] = 0.0
                if hasattr(je, 'DSP_PARAMS'):
                    try: je.DSP_PARAMS[1] = 0.0
                    except Exception: pass
            instr["pickup_types"] = pts
            instr["closest_distance_mm_per_pickup"] = cur
            samplerate = getattr(je, 'SAMPLERATE', samplerate)
            combs = compute_comb_delays_samples_from_mm(cur, pts, samplerate)
            instr["comb_delays_samples"] = combs
            if hasattr(je, 'set_comb_delays'):
                je.set_comb_delays(combs)
            elif hasattr(je, 'apply_instrument_state'):
                try:
                    je.apply_instrument_state(instr)
                except Exception:
                    pass
        return jsonify({"status":"ok","enabled":enabled})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@APP.route('/api/set_bypass', methods=['POST'])
def set_bypass():
    data = request.get_json(force=True)
    name = data.get("name"); state = data.get("state")
    if name is None or state is None:
        return jsonify({"error":"missing fields"}), 400
    try:
        if hasattr(je, 'EFFECTS_BYPASS'):
            je.EFFECTS_BYPASS[name] = bool(state)
        else:
            return jsonify({"error":"engine has no EFFECTS_BYPASS"}), 500
        return jsonify({"status":"ok","name":name,"bypassed":bool(state)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@APP.route('/api/update_controls', methods=['POST'])
def update_controls():
    data = request.get_json(force=True) or {}
    try:
        # update DSP-like params (pickup volumes, master gain) if engine provides DSP_PARAMS, otherwise ignore
        if hasattr(je, 'DSP_PARAMS'):
            try:
                if 'pickup1_volume' in data:
                    je.DSP_PARAMS[0] = float(data['pickup1_volume'])
                if 'pickup2_volume' in data:
                    je.DSP_PARAMS[1] = float(data['pickup2_volume'])
                if 'master_gain' in data:
                    je.DSP_PARAMS[3] = float(data['master_gain'])
            except Exception:
                pass
        return jsonify({"status":"ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@APP.route('/api/update_effects_params', methods=['POST'])
def update_effects_params():
    data = request.get_json(force=True) or {}
    param = data.get("param"); value = data.get("value")
    if not param:
        return jsonify({"error":"no param"}), 400
    if not hasattr(je, 'EFFECT_PARAMS'):
        return jsonify({"error":"engine missing EFFECT_PARAMS"}), 500
    if param not in je.EFFECT_PARAMS:
        return jsonify({"error":"invalid param"}), 400
    try:
        je.EFFECT_PARAMS[param] = float(value)
        # optional engine hook
        if hasattr(je, 'apply_effect_param_update'):
            try:
                je.apply_effect_param_update(param, float(value))
            except Exception:
                pass
        return jsonify({"status":"ok","param":param,"value":value})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@APP.route('/api/set_comp_state', methods=['POST'])
def api_set_comp_state():
    """
    Persist global compressor state and params.
    Body: { "bypass": true|false, "comp_params": { "comp_threshold": -18, "comp_ratio": 4, "comp_makeup": 1.2 } }
    """
    data = request.get_json(force=True) or {}
    bypass = data.get("bypass")
    comp_params = data.get("comp_params", {})
    try:
        if hasattr(je, 'EFFECTS_BYPASS'):
            if bypass is not None:
                je.EFFECTS_BYPASS['comp'] = bool(bypass)
        for k, v in comp_params.items():
            if k in je.EFFECT_PARAMS:
                je.EFFECT_PARAMS[k] = float(v)
        # persist global state
        save_global_effects({
            "effects_params": {k: float(je.EFFECT_PARAMS.get(k, 0.0)) for k in ["svf_base_cutoff","svf_env_depth","oct_dry","oct_sub_gain"]},
            "comp_bypass": bool(je.EFFECTS_BYPASS.get("comp", False)),
            "comp_params": {k: float(je.EFFECT_PARAMS.get(k, 0.0)) for k in ["comp_threshold","comp_ratio","comp_makeup"]}
        })
        return jsonify({"status":"ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------
# Preset management
# --------------------------
@APP.route('/api/list_presets')
def list_presets():
    try:
        user = sorted([p.stem for p in PRESET_DIR.glob("*.json") if p.is_file() and p.parent==PRESET_DIR])
        factory = sorted([p.stem for p in FACTORY_DIR.glob("*.json") if p.is_file()])
        user = [u for u in user if u not in set(factory)]
        return jsonify({"user": user, "factory": factory})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@APP.route('/api/save_preset', methods=['POST'])
def save_preset():
    data = request.get_json(force=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error":"missing name"}), 400
    preset_path = PRESET_DIR / f"{name}.json"
    try:
        # get instrument state from engine
        inst = je.get_instrument_state() if hasattr(je, 'get_instrument_state') else {
            "num_strings": getattr(je, "NUM_STRINGS", 6),
            "pickup_types": getattr(je, "INSTR_MODEL", {}).get("pickup_types", ["single","single"]),
            "closest_distance_mm_per_pickup": getattr(je, "INSTR_MODEL", {}).get("closest_distance_mm_per_pickup", [40.0,40.0]),
            "comb_delays_samples": getattr(je, "INSTR_MODEL", {}).get("comb_delays_samples", None)
        }

        # include all effect params (user presets store effects)
        effects_block = {
            "effects_params": { k: float(je.EFFECT_PARAMS.get(k, 0.0)) for k in je.EFFECT_PARAMS.keys() },
            "effects_bypass": { k: bool(je.EFFECTS_BYPASS.get(k, True)) for k in je.EFFECTS_BYPASS.keys() }
        }

        preset = {
            "name": name,
            "dsp_params": [1.0,1.0, je.EFFECT_PARAMS.get("svf_base_cutoff",20000.0), 1.0],
            "pickup_types": inst.get("pickup_types"),
            "closest_distance_mm_per_pickup": inst.get("closest_distance_mm_per_pickup"),
            "comb_delays_samples": inst.get("comb_delays_samples"),
            "num_strings": inst.get("num_strings"),
            "effects": effects_block
        }

        _atomic_write(preset_path, preset)
        with open(LAST_PRESET_FILE, "w", encoding='utf-8') as lf:
            lf.write(name)
        return jsonify({"status":"ok", "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import traceback

@APP.route('/api/load_preset', methods=['POST'])
def load_preset():
    data = request.get_json(force=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error":"missing name"}), 400

    user_path = PRESET_DIR / f"{name}.json"
    factory_path = FACTORY_DIR / f"{name}.json"
    preset_path = user_path if user_path.exists() else (factory_path if factory_path.exists() else None)
    if preset_path is None:
        return jsonify({"error":"preset not found"}), 404

    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            inst = json.load(f)
    except Exception as e:
        print("load_preset: failed to read preset file:", preset_path, e)
        traceback.print_exc()
        return jsonify({"error": "failed to read preset", "exc": str(e)}), 500

    try:
        # Factory presets: enforce SVF & Octaver bypass
        if preset_path.parent == FACTORY_DIR:
            if hasattr(je, 'EFFECTS_BYPASS'):
                je.EFFECTS_BYPASS['env_filter'] = True
                je.EFFECTS_BYPASS['octaver'] = True
            # compressor remains global (do not change)
        else:
            # user preset: apply effects block if present (but do not override global compressor state)
            eff = inst.get('effects', {})
            if eff:
                try:
                    for k,v in eff.get('effects_params', {}).items():
                        if k in je.EFFECT_PARAMS:
                            je.EFFECT_PARAMS[k] = float(v)
                    eb = eff.get('effects_bypass', {})
                    if 'env_filter' in eb:
                        je.EFFECTS_BYPASS['env_filter'] = bool(eb['env_filter'])
                    if 'octaver' in eb:
                        je.EFFECTS_BYPASS['octaver'] = bool(eb['octaver'])
                    # do NOT change 'comp' bypass here
                except Exception as e:
                    print("load_preset: warning applying effects block:", e)
                    traceback.print_exc()

        applied = False
        samplerate_to_use = getattr(je, 'SAMPLERATE', samplerate)

        # Try calling apply_instrument_state() in multiple ways to be robust.
        if hasattr(je, 'apply_instrument_state'):
            try:
                # 1) attempt keyword samplerate (preferred)
                applied = je.apply_instrument_state(inst, samplerate=samplerate_to_use)
            except TypeError as te_kw:
                # function may not accept keyword 'samplerate' - try positional
                try:
                    applied = je.apply_instrument_state(inst, samplerate_to_use)
                except TypeError as te_pos:
                    # maybe it only accepts a single arg
                    try:
                        applied = je.apply_instrument_state(inst)
                    except Exception as e3:
                        print("load_preset: apply_instrument_state all attempts failed")
                        traceback.print_exc()
                        applied = False
                except Exception as epos:
                    print("load_preset: apply_instrument_state (positional) raised:", epos)
                    traceback.print_exc()
                    applied = False
            except Exception as e:
                print("load_preset: apply_instrument_state raised:", e)
                traceback.print_exc()
                applied = False
        else:
            print("load_preset: engine has no apply_instrument_state()")

        # Fallback: try to update INSTR_MODEL and set_comb_delays directly
        if not applied:
            try:
                instr = getattr(je, 'INSTR_MODEL', None)
                if instr is None:
                    raise RuntimeError("engine missing INSTR_MODEL")

                # copy known fields safely
                if 'pickup_types' in inst:
                    instr['pickup_types'] = inst['pickup_types']
                if 'closest_distance_mm_per_pickup' in inst:
                    instr['closest_distance_mm_per_pickup'] = inst['closest_distance_mm_per_pickup']

                combs = inst.get('comb_delays_samples')
                if combs is None:
                    # compute from mm if present
                    if 'closest_distance_mm_per_pickup' in instr:
                        combs = compute_comb_delays_samples_from_mm(instr['closest_distance_mm_per_pickup'],
                                                                    instr.get('pickup_types', instr.get('pickup_types', ['single','single'])),
                                                                    getattr(je, 'SAMPLERATE', samplerate_to_use))
                        instr['comb_delays_samples'] = combs
                else:
                    instr['comb_delays_samples'] = combs

                # apply into engine structures
                if hasattr(je, 'set_comb_delays'):
                    try:
                        je.set_comb_delays(instr['comb_delays_samples'])
                        applied = True
                    except Exception as e:
                        print("load_preset: set_comb_delays failed:", e)
                        traceback.print_exc()
                        applied = False
                else:
                    # final fallback: try to reassign INSTR_MODEL (best-effort)
                    try:
                        # already modified instr in place
                        applied = True
                    except Exception:
                        applied = False
            except Exception as e:
                print("load_preset: fallback apply failed:", e)
                traceback.print_exc()
                applied = False

        if not applied:
            msg = "failed to apply preset to engine (no successful apply)"
            print("load_preset:", msg)
            return jsonify({"error": msg}), 500

        # Save last preset name
        try:
            with open(LAST_PRESET_FILE, "w", encoding='utf-8') as lf:
                lf.write(name)
        except Exception:
            pass

        # Return instrument state for the UI
        inst_state = je.get_instrument_state() if hasattr(je, 'get_instrument_state') else inst
        return jsonify({"status":"ok","name":name,"instrument_state": inst_state})

    except Exception as e:
        # This should print a full traceback to the server console for debugging
        print("load_preset: unexpected exception:", e)
        traceback.print_exc()
        # Also return a short traceback snippet to the client for visibility
        tb = traceback.format_exc()
        snippet = "\n".join(tb.splitlines()[-10:])
        return jsonify({"error": str(e), "trace": snippet}), 500

@APP.route('/api/delete_preset', methods=['POST'])
def delete_preset():
    data = request.get_json(force=True) or {}
    name = data.get("name")
    if not name:
        return jsonify({"error":"missing name"}), 400
    path = PRESET_DIR / f"{name}.json"
    if path.exists():
        try:
            path.unlink()
            return jsonify({"status":"ok"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error":"preset not found"}), 404

# --------------------------
# Meter emitter thread
# --------------------------
def start_meter_thread():
    def meter_loop():
        while True:
            try:
                payload = {
                    "peak_db": float(getattr(je, 'LAST_PEAK_DB', -120.0)),
                    "comp_gr_db": float(getattr(je, 'LAST_COMP_GR_DB', 0.0)),
                    "limiter_gr_db": float(getattr(je, 'LAST_LIMITER_GR_DB', 0.0))
                }
                SOCKET.emit('meter_data', payload)
                SOCKET.sleep(1.0/20.0)
            except Exception as e:
                print("meter thread exception:", e)
                time.sleep(0.25)
    t = Thread(target=meter_loop, daemon=True)
    t.start()

# --------------------------
# Server runner
# --------------------------
def run(host='0.0.0.0', port=5000):
    start_meter_thread()
    # flask-socketio run - don't use reloader as this file is not prepared for double-import
    SOCKET.run(APP, host=host, port=port, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)

# If executed as the main module for testing
if __name__ == '__main__':
    run(host='0.0.0.0', port=5000)
