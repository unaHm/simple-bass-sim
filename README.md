# simple-bass-sim
Simulating various  Bass Guitar instruments in software

The software in this repo attempts to emulate certain Bass Guitar types. The idea stemmed from the Roland GK series of processors, and the research undertaken by Joel de Guzman in his cycfi project.

The idea takes a number of feed-forward comb filters to approximate the effect of a Bass Guitar's pickup and width, while also taking into account any comb filtering that will occur from the position of the GK pickup itself.

A simple reverb was added for the simulation of an acoustic Bass body, which is activated in preset #4.

The presets are approximations of the following:
1. Jazz Bass (two pickups, with the opportunity to blend between them)
2. P-Bass
3. Stingray
4. Acoustic Bass

## 🎛️ System Architecture Overview

The **Simple Bass Simulator** system is a hybrid real-time bass guitar simulator built for the Linux audio environment using JACK.  
It models magnetic pickup behavior, pickup placement, and instrument body resonance through a chain of DSP modules optimized with Numba.

The goal was to build this code in Python to make it portable between different platforms and architectures. I wanted to target this for final use on a Raspberry Pi 5 using an XMOS XK-AUDIO-316-MC-AB board that I could then build into a 'stage box'. The Pi would be running PatchBoxOS, and I could connect to the DSP through the web UI.

## 🚀 Quickstart

Clone the repository and run the setup script:

```bash
git clone https://github.com/unahm/simple-bass-sim.git
cd simple-bass-sim/modular
./setup.sh install
```

### Signal Flow

<svg xmlns="http://www.w3.org/2000/svg" width="760" height="560">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#555"/>
    </marker>
  </defs>

  <rect x="40" y="20" width="260" height="60" fill="#f8f9fb" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  <text x="60" y="50" style="font:14px sans-serif; fill:#111">GK-3B Hex Pickup</text>
  <text x="60" y="68" style="font:12px sans-serif; fill:#444">(6 mono inputs)</text>

  <line x1="170" y1="80" x2="170" y2="110" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="40" y="110" width="680" height="70" fill="#f8f9fb" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  <text x="60" y="150" style="font:14px sans-serif; fill:#111">Per-string Comb Filters (bridge → pickup geometry)</text>
  <text x="60" y="168" style="font:12px sans-serif; fill:#444">Feedback path — static geometry</text>

  <line x1="350" y1="180" x2="350" y2="210" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="200" y="210" width="320" height="54" fill="#f8f9fb" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  <text x="220" y="242" style="font:14px sans-serif; fill:#111">Summing Mixer → Stereo</text>

  <line x1="350" y1="264" x2="350" y2="290" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="200" y="290" width="320" height="54" fill="#f8f9fb" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  <text x="220" y="322" style="font:14px sans-serif; fill:#111">SVF (Envelope-controlled)</text>

  <line x1="350" y1="344" x2="350" y2="370" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="200" y="370" width="320" height="54" fill="#f8f9fb" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  <text x="220" y="402" style="font:14px sans-serif; fill:#111">Octaver / Pitch Shifter (optional)</text>

  <line x1="350" y1="424" x2="350" y2="450" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="200" y="450" width="320" height="54" fill="#f8f9fb" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  <text x="220" y="482" style="font:14px sans-serif; fill:#111">Compressor → Limiter (global)</text>

  <line x1="350" y1="504" x2="350" y2="520" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="200" y="520" width="320" height="40" fill="#f8f9fb" stroke="#cbd5e1" stroke-width="1.5" rx="6"/>
  <text x="290" y="546" style="font:14px sans-serif; fill:#111">JACK Stereo Out</text>
</svg>


### Software Architecture

| Component | Role | Key Technologies |
|------------|------|------------------|
| **`jack_engine.py`** | Real-time DSP core, JACK callback management, Numba-compiled processing kernels | Python, `jack-client`, `numba`, `numpy` |
| **`dsp.py`** | DSP effect models (SVF, Octaver, Compressor, Limiter, Comb Filters) | `numpy`, `numba` |
| **`api.py`** | RESTful Flask API for preset management, DSP parameter control, and state serialization | Flask, JSON |
| **`ui_templates.py`** | Web-based control interface (sliders, meters, and parameter monitoring) | HTML5, JavaScript, CSS |
| **`instrument_geometry.json`** | Static definition of physical pickup geometry in millimeters | JSON |
| **`presets/`** | Factory and user preset definitions (pickup distances, EQ profiles, etc.) | JSON |
| **`main.py`** | Application launcher (initializes JACK, Flask server, and auto-loads presets) | Python |

### Runtime Integration

- **Audio Thread:**  
  Low-latency callback managed by `jack_engine`, processing each block with Numba-compiled kernels.
- **Control Thread:**  
  Flask web server exposes `/api/*` routes for real-time control from the web UI.
- **UI Thread:**  
  A lightweight HTML+JS frontend interacts with the Flask API for live parameter updates.
- **Persistence:**  
  User presets are stored in JSON; fixed hardware geometry (bridge–pickup distances) is read-only and defined externally.

---

## 📚 Technical References

### Core Frameworks and Libraries

- **Python Software Foundation.**  
  *The Python Language Reference, v3.12.*  
  <https://docs.python.org/3.12/>

- **NumPy Developers.**  
  *NumPy Reference Documentation.*  
  <https://numpy.org/doc/stable/>

- **Numba Developers.**  
  *Numba: A High Performance Python Compiler.*  
  <https://numba.pydata.org/>

- **Flask Framework.**  
  *Flask Web Development Framework Documentation.*  
  <https://flask.palletsprojects.com/>

- **JACK Audio Connection Kit.**  
  *Official JACK API Documentation and Python Bindings (`jack-client`).*  
  <https://jackaudio.org/api/>  
  <https://pypi.org/project/JACK-Client/>

### Digital Signal Processing Theory

- Smith, Julius O. (2010).  
  *Physical Audio Signal Processing.*  
  W3K Publishing, Stanford University.  
  [https://ccrma.stanford.edu/~jos/pasp/](https://ccrma.stanford.edu/~jos/pasp/)

- Zölzer, Udo (ed.) (2011).  
  *DAFX: Digital Audio Effects (2nd Edition).*  
  Wiley.  
  ISBN 978-0-470-74368-3.

- Pirkle, Will (2019).  
  *Designing Audio Effect Plug-Ins in C++: For AAX, AU, and VST3 with DSP Theory (2nd Edition).*  
  Routledge.  
  ISBN 978-1-138-05717-4.

### Modular Audio and Synthesis Frameworks

- *Pure Data (Pd) Documentation and Source Examples.*  
  [https://puredata.info/docs/](https://puredata.info/docs/)

- *Cycling ’74 Max/MSP Reference.*  
  [https://docs.cycling74.com/](https://docs.cycling74.com/)

### Hardware and Physical Modeling Context

- **Roland GK-3B Bass Divided Pickup – Owner’s Manual.**  
  Roland Corporation.  
  (for understanding string-to-channel architecture and magnetic pickup geometry)

- **General Bass Guitar Reference Materials.**  
  Various manufacturers’ technical drawings (Fender, MusicMan, Warwick, Hofner)  
  used for pickup spacing and bridge-to-pickup distance calibration.

---

## 🧠 Development Acknowledgments

- **DSP and Core System Architecture:**  
  Developed by Dan Swain with algorithmic assistance from **OpenAI GPT-5**, using  
  general DSP principles, JACK integration, and Numba optimization techniques.

- **User Interface (Web UI) Enhancements and Layout Improvements:**  
  Designed and refined with support from **Google Gemini**,  
  incorporating accessibility, numeric readouts, and real-time update optimizations.

- **No proprietary or copyrighted code** was used;  
  all DSP and control implementations are based on publicly documented, academically standard methods.

---

## 📄 Citation

If you reference this project in academic, research, or open-source documentation, please cite:

> Swain, D. (2025). *Simple Bass Simulator: Real-Time Physical Bass Guitar Simulation with Numba and JACK.*  
> Developed with algorithmic assistance from OpenAI GPT-5 and Google Gemini.  
> Retrieved from: [https://github.com/unahm/simple-bass-sim](https://github.com/unahm/simple-bass-sim)

---
