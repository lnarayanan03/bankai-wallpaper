#!/usr/bin/env python3
"""Sound track for the Blood Chain Bankai wallpaper (v8).

Everything is built on a circular 12 s timeline (noise made in the frequency
domain, events wrapped with modulo, reverb as circular convolution), so the
audio loops with no seam, exactly like the video.

Event times come from render.py itself, so whooshes land on the drawn slashes.

Usage: sound.py [out.wav]
"""
import math
import sys
import wave

import numpy as np

import render

SR = 48000
LOOP = render.LOOP
N = int(SR * LOOP)
T = np.arange(N) / SR
rng = np.random.default_rng(12)


def colored_noise(lo, hi, tilt=0.0):
    """Circular noise band-limited to lo..hi Hz, spectrum ~ f^-tilt."""
    f = np.fft.rfftfreq(N, 1 / SR)
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
    shape = ((f >= lo) & (f <= hi)) * np.power(np.maximum(f, 1), -tilt / 2)
    x = np.fft.irfft(spec * shape, N)
    return x / (np.abs(x).max() + 1e-9)


def env_at(t0, attack, decay):
    """Envelope starting at t0 (wrapped round the loop): linear attack, exp decay."""
    d = (T - t0) % LOOP
    a = np.clip(d / max(attack, 1e-4), 0, 1)
    rel = np.clip(d - attack, 0, None)
    return np.where(d < LOOP * 0.9, a * np.exp(-rel / decay), 0.0)


def ramp_to(t0, t1, power=2.0):
    """Rises from 0 at t0 to 1 at t1, then cuts."""
    d = (T - t0) % LOOP
    span = t1 - t0
    return np.where(d < span, (d / span) ** power, 0.0)


def boom(t0, f0, f1, dur, amp):
    d = (T - t0) % LOOP
    on = np.clip((dur * 4 - d) / dur, 0, 1)
    freq = f1 + (f0 - f1) * np.exp(-d / (dur * 0.25))
    ph = 2 * np.pi * np.cumsum(freq) / SR
    ph -= ph[int(t0 * SR) % N]
    return amp * np.sin(ph) * np.exp(-d / dur) * on * np.clip(d / 0.004, 0, 1)


def bandpass(x, centre, width):
    f = np.fft.rfftfreq(N, 1 / SR)
    g = np.exp(-((np.log2(np.maximum(f, 1)) - math.log2(centre)) / width) ** 2)
    return np.fft.irfft(np.fft.rfft(x) * g, N)


def reverb(x, secs, seed):
    r = np.random.default_rng(seed)
    n = int(SR * secs * 6)
    k = np.arange(n)
    ir = r.normal(size=n) * np.exp(-k / (SR * secs)) * np.cos(np.pi / 2 * k / n) ** 2
    ir[: int(SR * 0.012)] = 0
    ir = np.pad(ir, (0, N - n)) if n < N else ir[:N]
    y = np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(ir), N)
    return y / (np.abs(y).max() + 1e-9)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "out/v8_audio.wav"
    render.init()
    render.init_v8()
    G = render.G
    burst_times = [i * LOOP / render.BURSTS for i in range(render.BURSTS)]   # 0, 4, 8
    getsuga = render.T_MERGE

    # ---- ambient bed: low drone, rumbling fire, crackle
    drone = (0.5 * np.sin(2 * np.pi * 41 * T) + 0.35 * np.sin(2 * np.pi * 55 * T + 1.3)
             + 0.15 * np.sin(2 * np.pi * 82.5 * T)) * (0.8 + 0.2 * np.sin(2 * np.pi * T / LOOP * 3))
    fire = colored_noise(30, 900, tilt=1.6)
    fire *= 0.7 + 0.3 * np.sin(2 * np.pi * T * 2 / LOOP + colored_noise(0.1, 2) * 2)
    crackle = np.zeros(N)
    for i in rng.integers(0, N, 900):
        L = int(rng.integers(30, 220))
        seg = rng.normal(size=L) * np.exp(-np.arange(L) / (L / 4)) * rng.uniform(0.2, 1)
        idx = (i + np.arange(L)) % N
        crackle[idx] += seg
    crackle = bandpass(crackle, 3000, 1.2)
    crackle /= np.abs(crackle).max()
    amb = 0.35 * drone + 0.4 * fire + 0.12 * crackle

    # ---- hits on every burst: charge riser, then thump + crack
    hits = np.zeros(N)
    riser = np.zeros(N)
    hiss = colored_noise(200, 12000, tilt=0.5)
    for tb in burst_times:
        r = ramp_to(tb - 1.1, tb, 2.5)
        riser += bandpass(hiss, 400, 1.5) * r * 0.5 + np.sin(2 * np.pi * (60 + 140 * r) * T) * r * 0.15
        big = 1.0 if tb == getsuga else 0.85
        hits += np.tanh(2.5 * boom(tb, 160, 40, 0.5, 1.0)) * big
        hits += bandpass(hiss, 1800, 1.4) * env_at(tb, 0.002, 0.09) * 1.2 * big

    # ---- slashes of the shadow forms (6.5-7.8 s): fast whooshes
    swish = np.zeros(N)
    air = colored_noise(150, 9000, tilt=0.8)
    for s in G["strokes"]:
        d = (T - s["ts"]) % LOOP
        e = np.where(d < 0.35, np.sin(np.pi * np.clip(d / 0.3, 0, 1)) ** 2, 0)
        swish += bandpass(air * e, 1200 + 1500 * rng.random(), 1.0) * 0.8

    # ---- the Getsuga: build, a hole of silence, then the blast
    pre = ramp_to(getsuga - 1.3, getsuga, 3.0)
    scream = bandpass(hiss, 2500, 0.7) * pre * 0.6 + np.sin(2 * np.pi * (180 + 520 * pre ** 2) * T) * pre * 0.12
    blast = np.tanh(2 * boom(render.T_SB, 110, 26, 1.1, 1.0)) * 1.1
    blast += np.tanh(colored_noise(40, 6000, tilt=1.0) * 4) * env_at(render.T_SB, 0.003, 0.35) * 0.7
    # metallic ring: inharmonic partials
    for f, a in ((233, 0.3), (377, 0.22), (611, 0.16), (989, 0.1)):
        blast += np.sin(2 * np.pi * f * T) * env_at(render.T_SB, 0.002, 1.4) * a * 0.5
    # close-up: dark low chord and wind
    cu = ((T >= render.T_CU0) & (T < render.T_BACK)).astype(float)
    cu = np.convolve(cu, np.ones(2400) / 2400, mode="same")
    pad = sum(a * np.sin(2 * np.pi * f * T + p) for f, a, p in
              ((36.7, 0.5, 0), (73.4, 0.35, 1), (110.0, 0.25, 2), (146.8, 0.12, 0.5), (73.9, 0.2, 2.5)))
    pad = pad * cu * 0.6 + bandpass(air, 500, 1.2) * cu * 0.35
    for s in G["cu_streaks"]:
        d = (T - s["ts"]) % LOOP
        e = np.where(d < s["dur"], np.sin(np.pi * np.clip(d / s["dur"], 0, 1)) ** 3, 0)
        swish += bandpass(air * e, 2500, 0.8) * 0.7
    # cut back to the wide shot
    back = boom(render.T_BACK, 120, 32, 0.7, 1.0) + hiss * env_at(render.T_BACK, 0.002, 0.12) * 0.7

    # duck the bed during the blackout and the close-up
    duck = np.ones(N)
    duck[(T >= render.T_BO) & (T < render.T_BACK)] = 0.25
    duck[(T >= render.T_BO) & (T < render.T_SB)] = 0.0
    duck = np.convolve(duck, np.ones(480) / 480, mode="same")
    gap = np.ones(N)
    gap[(T >= render.T_BO) & (T < render.T_SB)] = 0.0
    gap = np.convolve(gap, np.ones(240) / 240, mode="same")

    dry = amb * duck + (riser + scream) * gap + hits + swish * 0.6 + blast + pad + back
    mix = []
    for ch in range(2):
        wet = reverb(dry, 0.9, 100 + ch)
        wide = np.roll(swish, 90 * ch) * 0.15
        mix.append(dry + 0.16 * wet * np.abs(dry).max() + wide)
    st = np.stack(mix, 1)
    st = np.tanh(st / np.abs(st).max() * 1.6) / math.tanh(1.6) * 0.89
    pcm = (st * 32767).astype(np.int16)
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("wrote", out)


if __name__ == "__main__":
    main()
