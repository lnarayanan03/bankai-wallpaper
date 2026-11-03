#!/usr/bin/env python3
"""Fit a music track to the wallpaper and mix it with the sound effects.

The video loops every 12 s. At 90 BPM a bar is 2.667 s, so 9 bars fill 24 s
exactly. This script:
  1. finds the track's tempo and beat grid,
  2. picks the loudest stretch of 18 bars whose end flows back into its start,
  3. time-stretches it to exactly 90 BPM (48 s = 4 video loops),
  4. crossfades the seam so it loops cleanly,
  5. lines a downbeat up with the Getsuga blast and mixes the effects on top.

Usage: music.py track.wav sfx.wav out.wav
The track itself is not in the repo (see README).
"""
import subprocess
import sys
import wave

import numpy as np

import render

BPM_OUT = 90.0
BARS = 18
VIDEO_LOOPS = 4
SR = 48000


def read_wav(p):
    w = wave.open(p)
    assert w.getframerate() == SR
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels()) / 32768.0
    return x if x.shape[1] == 2 else np.repeat(x, 2, 1)


def write_wav(p, x):
    with wave.open(p, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def onset_env(mono, hop=512):
    n = len(mono) // hop
    S = np.abs(np.fft.rfft(mono[:n * hop].reshape(n, hop) * np.hanning(hop), axis=1))
    flux = np.r_[0, np.clip(np.diff(np.log1p(S * 10), axis=0), 0, None).sum(1)]
    flux -= np.convolve(flux, np.ones(43) / 43, "same")
    return np.clip(flux, 0, None), SR / hop, S


def beat_grid(flux, fps, lo, hi):
    """Best tempo and beat phase on the section lo..hi seconds."""
    i0, i1 = int(lo * fps), int(hi * fps)
    seg = flux[i0:i1]
    best = (-1, 0, 0)
    for bpm in np.arange(91.0, 93.5, 0.01):
        per = fps * 60 / bpm
        for ph in np.linspace(0, per, 48, endpoint=False):
            idx = (ph + per * np.arange(int((len(seg) - ph) / per))).astype(int)
            v = seg[idx].mean()
            if v > best[0]:
                best = (v, bpm, ph)
    _, bpm, ph = best
    return bpm, (i0 + ph) / fps


def main():
    track, sfx_p, out = sys.argv[1:4]
    x = read_wav(track)
    mono = x.mean(1)
    flux, fps, S = onset_env(mono)
    bpm, b0 = beat_grid(flux, fps, 140, 240)
    beat = 60 / bpm
    # downbeat: the beat phase (of 4) with the strongest low-end onsets
    low = np.clip(np.diff(np.log1p(S[:, 1:12] * 10), axis=0), 0, None).sum(1)
    beats = b0 + beat * np.arange(-int(b0 / beat), int((len(mono) / SR - b0) / beat))
    beats = beats[(beats > 0) & (beats < len(mono) / SR - 1)]
    strength = [low[(beats[k::4] * fps).astype(int)].mean() for k in range(4)]
    bars = beats[int(np.argmax(strength))::4]
    print(f"tempo {bpm:.2f} BPM, {len(bars)} bars")

    # loud 18-bar sections whose last bar sounds like the bar before the first
    L = BARS * 4 * beat
    cs = np.r_[0, np.cumsum(mono ** 2)]
    rms_of = lambda a, b: np.sqrt((cs[int(b * SR)] - cs[int(a * SR)]) / max(int(b * SR) - int(a * SR), 1))
    best = None
    for k, s0 in enumerate(bars):
        if k == 0 or k + BARS >= len(bars):
            continue
        e0 = bars[k + BARS]
        a = S[int(bars[k - 1] * fps):int(s0 * fps)].mean(0)
        b = S[int(bars[k + BARS - 1] * fps):int(e0 * fps)].mean(0)
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        loud = rms_of(s0, e0)
        score = sim * loud
        if best is None or score > best[0]:
            best = (score, s0, e0, sim, loud)
    _, s0, e0, sim, loud = best
    print(f"section {s0:.2f}-{e0:.2f} s (similarity {sim:.3f})")

    # stretch to exactly 90 BPM, plus one bar of tail for the crossfade
    tail = 4 * beat
    seg = x[int(s0 * SR):int((e0 + tail) * SR)]
    write_wav("out/_seg.wav", seg)
    tempo = BPM_OUT / bpm
    subprocess.run(["./ffmpeg", "-v", "error", "-y", "-i", "out/_seg.wav", "-af", f"atempo={tempo:.6f}",
                    "-ar", str(SR), "out/_seg_st.wav"], check=True)
    y = read_wav("out/_seg_st.wav")
    N = int(round(render.LOOP * VIDEO_LOOPS * SR))
    xf = int(SR * 60 / BPM_OUT)                       # one-beat crossfade at the seam
    if len(y) < N + xf:
        y = np.pad(y, ((0, N + xf - len(y)), (0, 0)))
    w = np.sin(np.linspace(0, np.pi / 2, xf))[:, None] ** 2
    loop = y[:N].copy()
    loop[:xf] = y[:xf] * w + y[N:N + xf] * (1 - w)

    # put a downbeat on the Getsuga blast (bar 0 of the section -> T_SB)
    loop = np.roll(loop, int(render.T_SB * SR), axis=0)

    sfx = read_wav(sfx_p)
    sfx = np.tile(sfx, (VIDEO_LOOPS, 1))[:N]
    T = np.arange(N) / SR % render.LOOP
    # a hole of silence in the music on each blackout, then it slams back
    duck = np.ones(N)
    duck[(T >= render.T_BO - 0.02) & (T < render.T_SB)] = 0.0
    duck = np.convolve(duck, np.ones(240) / 240, "same")[:, None]
    m = loop / (np.abs(loop).max() + 1e-9)
    mix = m * 0.8 * duck + sfx * 0.45
    mix = np.tanh(mix * 1.3) / np.tanh(1.3) * 0.89
    write_wav(out, mix)
    print("wrote", out, f"({N / SR:.0f} s)")


if __name__ == "__main__":
    main()
