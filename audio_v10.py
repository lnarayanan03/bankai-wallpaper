#!/usr/bin/env python3
"""Soundtrack for v10 (48 s = 2 video loops): fitted music + effects on the cuts.

Uses out/_seg_st.wav (the bankai theme cut to 18 bars and stretched to 90 BPM by
music.py) and the effect helpers in sound.py, re-timed for the 24 s loop.

Usage: audio_v10.py out.wav
"""
import sys

import numpy as np

import music
import render_v10 as V
import sound as S

SR = S.SR
LOOP = V.LOOP
N = int(SR * LOOP)
# the helpers in sound.py read these module globals
S.LOOP, S.N = LOOP, N
S.T = np.arange(N) / SR
T = S.T


def sfx():
    out = np.zeros(N)
    # low fire bed
    fire = S.bandpass(S.colored_noise(30, 4000, tilt=1.0), 300, 2.2)
    out += fire * 0.10 * (0.8 + 0.2 * np.sin(2 * np.pi * T * 3 / LOOP))
    # idle power bursts
    for t0 in (0.0, 4.0, 8.0, 12.0):
        out += S.boom(t0, 90, 35, 0.6, 0.35)
    # push-in riser: noise that climbs in pitch and level, cut dead at the cut
    rise = S.ramp_to(V.T_PUSH, V.T_CUT, power=2.2)
    sweep = S.bandpass(S.colored_noise(200, 12000, tilt=0.5), 2500, 2.5)
    tone = np.sin(2 * np.pi * np.cumsum(60 + 140 * rise) / SR)
    out += (sweep * 0.5 + tone * 0.35) * rise
    # eye flash: a high metallic ring
    for f, a in ((2960, 0.25), (4440, 0.15), (6100, 0.08)):
        out += np.sin(2 * np.pi * f * T) * S.env_at(V.T_CUT + 0.06, 0.002, 0.35) * a
    # moon reveal: a deep boom with a hiss
    hiss = S.colored_noise(500, 16000, tilt=0.2)
    out += np.tanh(2 * S.boom(V.T_MOON, 120, 28, 1.0, 1.0)) * 0.9
    out += hiss * S.env_at(V.T_MOON, 0.002, 0.25) * 0.35
    # wind-up and slash whooshes
    wnoise = S.bandpass(S.colored_noise(100, 9000, tilt=0.8), 900, 1.5)
    out += wnoise * S.env_at(V.T_RAISE, 0.3, 0.25) * 0.4
    out += S.bandpass(S.colored_noise(300, 14000, tilt=0.3), 3500, 1.8) * S.env_at(V.T_SLASH, 0.01, 0.12) * 0.9
    # the Getsuga flying: a rushing roar that builds into the hit
    fly = S.ramp_to(V.T_FLY, V.T_HIT, power=1.5)
    out += S.bandpass(S.colored_noise(60, 6000, tilt=1.0), 500, 2.0) * fly * 0.9
    # impact on the downbeat
    out += np.tanh(3 * S.boom(V.T_HIT, 140, 30, 1.3, 1.0)) * 1.1
    out += np.tanh(S.colored_noise(40, 9000, tilt=1.0) * 4) * S.env_at(V.T_HIT, 0.002, 0.4) * 0.6
    # eclipse closing: a low swell, then a bright chime at totality
    out += S.bandpass(S.colored_noise(30, 400, tilt=1.2), 70, 1.5) * (S.ramp_to(V.T_ECL0, V.T_ECL1, 1.0) * 0.5)
    for f, a in ((1320, 0.2), (1980, 0.14), (2640, 0.08)):
        out += np.sin(2 * np.pi * f * T) * S.env_at(V.T_ECL1, 0.004, 0.9) * a
    rv = S.reverb(out, 0.9, 3)
    out = out + rv * 0.25
    return out / (np.abs(out).max() + 1e-9)


def main():
    dst = sys.argv[1]
    loops = 2
    Nn = N * loops
    y = music.read_wav("out/_seg_st.wav")
    xf = int(SR * 60 / music.BPM_OUT)
    if len(y) < Nn + xf:
        y = np.pad(y, ((0, Nn + xf - len(y)), (0, 0)))
    w = np.sin(np.linspace(0, np.pi / 2, xf))[:, None] ** 2
    m = y[:Nn].copy()
    m[:xf] = y[:xf] * w + y[Nn:Nn + xf] * (1 - w)
    # bar 0 of the section lands on the cut (16 s, a bar line)
    m = np.roll(m, int(V.T_CUT * SR), axis=0)
    m = m / (np.abs(m).max() + 1e-9)

    Tl = np.arange(Nn) / SR % LOOP
    duck = np.ones(Nn)
    duck[(Tl >= V.T_CUT - 0.02) & (Tl < V.T_MOON)] = 0.0        # silence on the black cut
    duck[(Tl >= V.T_PUSH) & (Tl < V.T_CUT)] *= 0.75
    duck = np.convolve(duck, np.ones(240) / 240, "same")[:, None]

    fx = np.tile(sfx(), loops)[:, None].repeat(2, 1)
    mix = m * 0.75 * duck + fx * 0.5
    mix = np.tanh(mix * 1.3) / np.tanh(1.3) * 0.89
    music.write_wav(dst, mix)
    print("wrote", dst, f"({Nn / SR:.0f} s)")


if __name__ == "__main__":
    main()
