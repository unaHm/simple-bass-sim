#!/usr/bin/env python3
"""
main.py — modern startup for jackbasssim modular

Behavior:
- Start jack engine (real JACK if available, otherwise run a dummy audio loop)
- Autoload last preset (presets/last_preset.txt) if present
- Start API (Flask + SocketIO) in a background thread (api.run)
- Handle Ctrl+C clean shutdown
"""

import os
import sys
import threading
import time
from pathlib import Path

import jack_engine as je
import api

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

        print(f"autoload: loading preset '{name}' from {preset_path}")
        inst = je.load_preset_file(str(preset_path))
        if not inst:
            print("autoload: failed to read preset file")
            return False

        ok = je.apply_instrument_state(inst, samplerate=je.SAMPLERATE if hasattr(je, 'SAMPLERATE') else 48000)
        print("autoload:", "success" if ok else "failed to apply instrument state")
        return ok
    except Exception as e:
        print("autoload failed:", e)
        return False


def start_api_server_in_thread(host='0.0.0.0', port=5000):
    """Start api.run in a daemon thread so main can continue."""
    def target():
        try:
            api.run(host=host, port=port)
        except Exception as e:
            print("API server thread exited with error:", e)

    t = threading.Thread(target=target, daemon=True, name="api-server")
    t.start()
    return t


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


def main():
    print("main.py starting...")

    # 1) Start JACK or dummy audio loop
    audio_thread = start_jack_or_dummy()

    # 2) Autoload last preset (best-effort)
    try:
        autoload_last_preset()
    except Exception as e:
        print("autoload_last_preset error:", e)

    # 3) Start API server
    api_thread = start_api_server_in_thread(host='0.0.0.0', port=5000)
    print("API server started in background thread.")

    # 4) Block until killed
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received — shutting down.")
    except Exception as e:
        print("main loop error:", e)
    finally:
        # Attempt graceful shutdown
        try:
            if hasattr(je, 'client') and je.client is not None:
                try:
                    je.client.deactivate()
                    print("JACK client deactivated.")
                except Exception:
                    pass
        except Exception:
            pass
        print("Exiting.")

if __name__ == '__main__':
    main()
