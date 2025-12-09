"""
dsp.py - DSP kernels and helpers
"""
import math
import numpy as np

# Optional numba
try:
    import numba as nb
    NUMBA_AVAILABLE = True
except Exception:
    nb = None
    NUMBA_AVAILABLE = False

def db_to_amp(db):
    return 10.0 ** (db / 20.0)

def amp_to_db(amp):
    if amp <= 1e-12:
        return -120.0
    return 20.0 * math.log10(abs(amp))

# --- simplified string DSP (numba optional) ---
if NUMBA_AVAILABLE:
    @nb.njit(cache=True)
    def run_string_dsp_numba(
        frames_in, sample_in, sample_out, delayed_sample_lpf, is_pickup2_enabled, one_third,
        fb_delay_len, fb_gain, fb_buffer, fb_idx,
        ff1_p1_delay_len, ff1_p1_gain, ff1_p1_buffer, ff1_p1_idx,
        ff2_p1_delay_len, ff2_p1_gain, ff2_p1_buffer, ff2_p1_idx,
        ff3_p1_delay_len, ff3_p1_gain, ff3_p1_buffer, ff3_p1_idx,
        ff1_p2_delay_len, ff1_p2_gain, ff1_p2_buffer, ff1_p2_idx,
        ff2_p2_delay_len, ff2_p2_gain, ff2_p2_buffer, ff2_p2_idx,
        ff3_p2_delay_len, ff3_p2_gain, ff3_p2_buffer, ff3_p2_idx,
        dsp_params, dummy_arg_36
    ):
        p1_vol = dsp_params[0]
        p2_vol = dsp_params[1]
        master_gain = dsp_params[3]
        for n in range(frames_in):
            delayed_sample = fb_buffer[fb_idx[0]]
            input_sample = sample_in[n]
            dws_sample = (delayed_sample * fb_gain) + input_sample
            p1_out = (
                (ff1_p1_gain * dws_sample) +
                (ff2_p1_gain * ff2_p1_buffer[ff2_p1_idx[0]]) +
                (ff3_p1_gain * ff3_p1_buffer[ff3_p1_idx[0]])
            ) * one_third
            if is_pickup2_enabled:
                p2_out = (
                    (ff1_p2_gain * dws_sample) +
                    (ff2_p2_gain * ff2_p2_buffer[ff2_p2_idx[0]]) +
                    (ff3_p2_gain * ff3_p2_buffer[ff3_p2_idx[0]])
                ) * one_third
            else:
                p2_out = 0.0
            final_sample = (p1_out * p1_vol) + (p2_out * p2_vol)
            final_sample *= master_gain
            sample_out[n] = final_sample
            fb_idx[0] = (fb_idx[0] + 1) % fb_delay_len
            ff1_p1_idx[0] = (ff1_p1_idx[0] + 1) % ff1_p1_delay_len
            ff2_p1_idx[0] = (ff2_p1_idx[0] + 1) % ff2_p1_delay_len
            ff3_p1_idx[0] = (ff3_p1_idx[0] + 1) % ff3_p1_delay_len
            if is_pickup2_enabled:
                ff1_p2_idx[0] = (ff1_p2_idx[0] + 1) % ff1_p2_delay_len
                ff2_p2_idx[0] = (ff2_p2_idx[0] + 1) % ff2_p2_delay_len
                ff3_p2_idx[0] = (ff3_p2_idx[0] + 1) % ff3_p2_delay_len
        return 0

    @nb.njit(cache=True)
    def run_lpf_numba(input_array, a0, b1, z1_state, dsp_params, dummy_arg):
        out = input_array.copy()
        z = z1_state[0]
        for i in range(len(input_array)):
            v = a0 * input_array[i] + b1 * z
            z = v
            out[i] = v
        z1_state[0] = z
        return out

else:
    def run_string_dsp_numba(
        frames_in, sample_in, sample_out, delayed_sample_lpf, is_pickup2_enabled, one_third,
        fb_delay_len, fb_gain, fb_buffer, fb_idx,
        ff1_p1_delay_len, ff1_p1_gain, ff1_p1_buffer, ff1_p1_idx,
        ff2_p1_delay_len, ff2_p1_gain, ff2_p1_buffer, ff2_p1_idx,
        ff3_p1_delay_len, ff3_p1_gain, ff3_p1_buffer, ff3_p1_idx,
        ff1_p2_delay_len, ff1_p2_gain, ff1_p2_buffer, ff1_p2_idx,
        ff2_p2_delay_len, ff2_p2_gain, ff2_p2_buffer, ff2_p2_idx,
        ff3_p2_delay_len, ff3_p2_gain, ff3_p2_buffer, ff3_p2_idx,
        dsp_params, dummy_arg_36
    ):
        p1_vol = float(dsp_params[0])
        p2_vol = float(dsp_params[1])
        master_gain = float(dsp_params[3])
        for n in range(frames_in):
            delayed_sample = float(fb_buffer[fb_idx[0]])
            input_sample = float(sample_in[n])
            dws_sample = (delayed_sample * float(fb_gain)) + input_sample
            p1_out = (
                (float(ff1_p1_gain) * dws_sample) +
                (float(ff2_p1_gain) * float(ff2_p1_buffer[ff2_p1_idx[0]])) +
                (float(ff3_p1_gain) * float(ff3_p1_buffer[ff3_p1_idx[0]]))
            ) * float(one_third)
            if is_pickup2_enabled:
                p2_out = (
                    (float(ff1_p2_gain) * dws_sample) +
                    (float(ff2_p2_gain) * float(ff2_p2_buffer[ff2_p2_idx[0]])) +
                    (float(ff3_p2_gain) * float(ff3_p2_buffer[ff3_p2_idx[0]]))
                ) * float(one_third)
            else:
                p2_out = 0.0
            final_sample = (p1_out * p1_vol) + (p2_out * p2_vol)
            final_sample *= master_gain
            sample_out[n] = final_sample
            fb_idx[0] = (fb_idx[0] + 1) % int(fb_delay_len)
            ff1_p1_idx[0] = (ff1_p1_idx[0] + 1) % int(ff1_p1_delay_len)
            ff2_p1_idx[0] = (ff2_p1_idx[0] + 1) % int(ff2_p1_delay_len)
            ff3_p1_idx[0] = (ff3_p1_idx[0] + 1) % int(ff3_p1_delay_len)
            if is_pickup2_enabled:
                ff1_p2_idx[0] = (ff1_p2_idx[0] + 1) % int(ff1_p2_delay_len)
                ff2_p2_idx[0] = (ff2_p2_idx[0] + 1) % int(ff2_p2_delay_len)
                ff3_p2_idx[0] = (ff3_p2_idx[0] + 1) % int(ff3_p2_delay_len)
        return 0

# --- SVF, Octaver, Compressor ---
if NUMBA_AVAILABLE:
    @nb.njit(cache=True)
    def process_svf_numba(frames, in_block, out_block, z1_state, z2_state, env_state,
                          alpha_atk, alpha_rel, base_cutoff, env_depth, mix, samplerate):
        sr = float(samplerate)
        env = float(env_state[0])
        z1 = float(z1_state[0]); z2 = float(z2_state[0])
        for i in range(frames):
            x = in_block[i]
            a = abs(x)
            if a > env:
                env = alpha_atk * env + (1.0 - alpha_atk) * a
            else:
                env = alpha_rel * env + (1.0 - alpha_rel) * a
            cutoff = base_cutoff + env * env_depth
            cutoff = max(20.0, min(cutoff, sr * 0.49))
            g = math.tan(math.pi * cutoff / sr)
            R = 1.0
            v3 = x - z2
            v1_new = z1 + g * z2
            v2_new = (z2 + g * v3 - R * z2) / (1.0 + g * R)
            z1 = v1_new; z2 = v2_new
            out_block[i] = v2_new * mix + x * (1.0 - mix)
        z1_state[0] = z1; z2_state[0] = z2; env_state[0] = env
        return 0
else:
    def process_svf_numba(frames, in_block, out_block, z1_state, z2_state, env_state,
                          alpha_atk, alpha_rel, base_cutoff, env_depth, mix, samplerate):
        sr = float(samplerate)
        env = float(env_state[0]); z1 = float(z1_state[0]); z2 = float(z2_state[0])
        for i in range(frames):
            x = float(in_block[i])
            a = abs(x)
            if a > env:
                env = alpha_atk * env + (1.0 - alpha_atk) * a
            else:
                env = alpha_rel * env + (1.0 - alpha_rel) * a
            cutoff = base_cutoff + env * env_depth
            cutoff = max(20.0, min(cutoff, sr * 0.49))
            g = math.tan(math.pi * cutoff / sr)
            R = 1.0
            v3 = x - z2
            v1_new = z1 + g * z2
            v2_new = (z2 + g * v3 - R * z2) / (1.0 + g * R)
            z1 = v1_new; z2 = v2_new
            out_block[i] = v2_new * mix + x * (1.0 - mix)
        z1_state[0] = z1; z2_state[0] = z2; env_state[0] = env
        return 0

if NUMBA_AVAILABLE:
    @nb.njit(cache=True)
    def process_octaver_numba(frames, in_block, out_block, prev_sign_arr, phase_state, lp_state, lp_alpha, dry, sub_gain):
        ps = int(phase_state[0]); prev_sign = int(prev_sign_arr[0]); lp = float(lp_state[0])
        for i in range(frames):
            s = in_block[i]
            sign = 1 if s >= 0 else -1
            if sign != prev_sign:
                ps ^= 1; prev_sign = sign
            raw_sub = (1.0 if ps else -1.0) * abs(s)
            lp = lp + lp_alpha * (raw_sub - lp)
            out_block[i] = s * dry + lp * sub_gain
        prev_sign_arr[0] = prev_sign; phase_state[0] = ps; lp_state[0] = lp
        return 0
else:
    def process_octaver_numba(frames, in_block, out_block, prev_sign_arr, phase_state, lp_state, lp_alpha, dry, sub_gain):
        ps = int(phase_state[0]); prev_sign = int(prev_sign_arr[0]); lp = float(lp_state[0])
        for i in range(frames):
            s = float(in_block[i])
            sign = 1 if s >= 0 else -1
            if sign != prev_sign:
                ps ^= 1; prev_sign = sign
            raw_sub = (1.0 if ps else -1.0) * abs(s)
            lp = lp + lp_alpha * (raw_sub - lp)
            out_block[i] = s * dry + lp * sub_gain
        prev_sign_arr[0] = prev_sign; phase_state[0] = ps; lp_state[0] = lp
        return 0

if NUMBA_AVAILABLE:
    @nb.njit(cache=True)
    def process_compressor_numba(frames, in_block, out_block, env_state, alpha_atk, alpha_rel, threshold_lin, ratio, makeup, gain_state, gain_alpha_atk, gain_alpha_rel):
        env = float(env_state[0]); gstate = float(gain_state[0])
        for i in range(frames):
            x = in_block[i]
            lvl = abs(x)
            if lvl > env:
                env = alpha_atk * env + (1.0 - alpha_atk) * lvl
            else:
                env = alpha_rel * env + (1.0 - alpha_rel) * lvl
            if env <= threshold_lin:
                desired_gain = 1.0
            else:
                in_db = 20.0 * math.log10(env + 1e-12)
                thr_db = 20.0 * math.log10(threshold_lin + 1e-12)
                gain_db = (thr_db - in_db) * (1.0 - 1.0/ratio)
                desired_gain = 10.0 ** (gain_db / 20.0)
            if desired_gain < gstate:
                gstate = gain_alpha_atk * gstate + (1.0 - gain_alpha_atk) * desired_gain
            else:
                gstate = gain_alpha_rel * gstate + (1.0 - gain_alpha_rel) * desired_gain
            out_block[i] = x * gstate * makeup
        env_state[0] = env; gain_state[0] = gstate
        return 0
else:
    def process_compressor_numba(frames, in_block, out_block, env_state, alpha_atk, alpha_rel, threshold_lin, ratio, makeup, gain_state, gain_alpha_atk, gain_alpha_rel):
        env = float(env_state[0]); gstate = float(gain_state[0])
        for i in range(frames):
            x = float(in_block[i])
            lvl = abs(x)
            if lvl > env:
                env = alpha_atk * env + (1.0 - alpha_atk) * lvl
            else:
                env = alpha_rel * env + (1.0 - alpha_rel) * lvl
            if env <= threshold_lin:
                desired_gain = 1.0
            else:
                in_db = 20.0 * math.log10(env + 1e-12)
                thr_db = 20.0 * math.log10(threshold_lin + 1e-12)
                gain_db = (thr_db - in_db) * (1.0 - 1.0/ratio)
                desired_gain = 10.0 ** (gain_db / 20.0)
            if desired_gain < gstate:
                gstate = gain_alpha_atk * gstate + (1.0 - gain_alpha_atk) * desired_gain
            else:
                gstate = gain_alpha_rel * gstate + (1.0 - gain_alpha_rel) * desired_gain
            out_block[i] = x * gstate * makeup
        env_state[0] = env; gain_state[0] = gstate
        return 0
