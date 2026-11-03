#!/usr/bin/env python3
"""v9: Blood Chain Bankai wallpaper built from the real TYBW animation.

Timeline (about 51 s, loops seamlessly through black):
  idle     14 s   the red crescent moon, slowly breathing, embers drifting (OST)
  bankai   25.6 s "The... moon?" ... "...kai." -> eyes, molten coat, black blades,
                  Ichigo standing in the red chamber (episode audio)
  getsuga  5.75 s "Getsuga... tensho!!" -> blackout, starburst, crescent close-up
  moon     9.3 s  the explosion clears to the black moon eclipse, then a slow fade to black

Sources (not in the repo; download with yt-dlp into assets/clips/):
  tuNgPgCToNY  bankai transformation, 1080p60, burned-in subtitles painted out
  7RIb93r3cuM  Getsuga Tensho, 1080p
  J98sYIUTCw0  clean 1080p upload: idle moon frame
  vGQBzLBs3nY  the black moon eclipse after the Getsuga
  0mornRTkHFg  official "Blood Chains Theme" OST (audio)

Usage: montage.py out.mp4
"""
import math
import subprocess
import sys

import cv2
import numpy as np

W, H, FPS, SR = 1920, 1080, 60, 48000
C = "assets/clips/"
FF = "./ffmpeg"
OST = "assets/ost_blood_chains.wav"

# (source, start, end, subtitle window end or 0)
SEGS = [
    ("tuNgPgCToNY.mp4", 2.0, 27.6, 12.0),
    ("7RIb93r3cuM.mp4", 94.6, 100.35, 97.85),
    ("vGQBzLBs3nY.mp4", 16.0, 21.8, 0),
]
IDLE = 14.0
HOLD = 2.5      # black moon held after its shot
FADE = 1.0      # fade to black at the end


def frames(src, t0, t1):
    cmd = [FF, "-v", "error", "-ss", f"{t0}", "-t", f"{t1 - t0}", "-i", C + src,
           "-vf", f"fps={FPS},scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3).copy()
    p.wait()


def grab(src, t):
    return next(frames(src, t, t + 0.05))


def unsub(img):
    """Paint out a burned-in subtitle (white text, black outline, bottom centre)."""
    y0, y1, x0, x1 = 935, 1050, 520, 1400
    box = img[y0:y1, x0:x1]
    white = (box.min(2) > 190).astype(np.uint8)
    if white.sum() < 120:
        return img
    m = cv2.dilate(white, np.ones((3, 3), np.uint8), iterations=5) * 255
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = m
    return cv2.inpaint(img, full, 5, cv2.INPAINT_TELEA)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class Idle:
    """The red crescent moon from the episode, brought to life."""

    def __init__(self):
        self.base = grab("J98sYIUTCw0.mp4", 6.5).astype(np.float32) / 255
        red = self.base[..., 2]
        self.moon = (red > 0.25).astype(np.float32)
        ys, xs = np.nonzero(self.moon)
        self.c = (float(xs.mean()), float(ys.mean()))
        self.glow = cv2.GaussianBlur(self.base * self.moon[..., None], (0, 0), 25)
        self.glow2 = cv2.GaussianBlur(self.base * self.moon[..., None], (0, 0), 90)
        rng = np.random.default_rng(9)
        n = 260
        self.emb = dict(x=rng.uniform(0, W, n), y=rng.uniform(0, H, n), v=rng.uniform(15, 70, n),
                        sway=rng.uniform(5, 30, n), ph=rng.uniform(0, 6.28, n),
                        r=rng.choice([1, 1, 2, 2, 3], n), b=rng.uniform(0.3, 1, n), fl=rng.uniform(2, 7, n))
        self.mist = [cv2.resize(rng.uniform(0, 1, (H // 60, W // 60)).astype(np.float32), (W, H),
                                interpolation=cv2.INTER_CUBIC) for _ in range(2)]

    def frame(self, t):
        k = t / IDLE
        pulse = 0.5 + 0.5 * math.sin(2 * math.pi * t / 3.5)
        img = self.base * (0.95 + 0.08 * pulse) + self.glow * 0.22 * (0.6 + 0.4 * pulse) + self.glow2 * 0.25
        a = 2 * math.pi * t / IDLE
        mist = self.mist[0] * math.cos(a) ** 2 + self.mist[1] * math.sin(a) ** 2
        img += (mist * 0.05)[..., None] * np.array([0.05, 0.05, 1.0], np.float32)
        e = self.emb
        lay = np.zeros((H, W), np.float32)
        for i in range(len(e["x"])):
            x = e["x"][i] + e["sway"][i] * math.sin(t * 0.8 + e["ph"][i])
            y = (e["y"][i] - e["v"][i] * t) % H
            f = 0.6 + 0.4 * math.sin(t * e["fl"][i] + e["ph"][i])
            cv2.circle(lay, (int(x * 4), int(y * 4)), int(e["r"][i] * 4), float(e["b"][i] * f), -1, cv2.LINE_AA, 2)
        img += (cv2.GaussianBlur(lay, (0, 0), 1.2) + cv2.GaussianBlur(lay, (0, 0), 6) * 0.8)[..., None] \
            * np.array([0.15, 0.25, 1.0], np.float32)
        z = 1.0 + 0.06 * smoothstep(0, 1, k)
        M = cv2.getRotationMatrix2D(self.c, 0.6 * k, z)
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
        img *= smoothstep(0.0, 1.5, t)          # fade in from the black of the loop seam
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def video(out):
    enc = subprocess.Popen([FF, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    idle = Idle()
    n = 0
    for i in range(int(IDLE * FPS)):
        enc.stdin.write(idle.frame(i / FPS).tobytes())
        n += 1
    print("idle done", n, flush=True)
    last = None
    for src, t0, t1, sub_end in SEGS:
        for j, f in enumerate(frames(src, t0, t1)):
            if t0 + j / FPS < sub_end:
                f = unsub(f)
            enc.stdin.write(f.tobytes())
            last = f
            n += 1
        print(src, "done", n, flush=True)
    # hold the black moon: slow push-in, then fade to black
    base = last.astype(np.float32)
    nh = int((HOLD + FADE) * FPS)
    for i in range(nh):
        t = i / FPS
        M = cv2.getRotationMatrix2D((W / 2, H * 0.42), 0, 1.0 + 0.04 * t / (HOLD + FADE))
        f = cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        f *= 1 - smoothstep(HOLD, HOLD + FADE, t)
        enc.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    return n


def audio_seg(src, t0, t1):
    raw = subprocess.run([FF, "-v", "error", "-ss", f"{t0}", "-t", f"{t1 - t0}", "-i", src, "-vn", "-ac", "2",
                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32).reshape(-1, 2).copy()
    want = int(round((t1 - t0) * SR))
    return np.pad(a, ((0, max(0, want - len(a))), (0, 0)))[:want]


def audio(out, nframes):
    N = int(round(nframes / FPS * SR))
    idle = audio_seg(OST, 0.0, IDLE + 0.3)
    parts = [audio_seg(C + s, t0, t1) for s, t0, t1, _ in SEGS]
    tail = audio_seg(C + SEGS[-1][0], SEGS[-1][2], SEGS[-1][2] + HOLD + FADE)
    ti = np.arange(len(tail)) / SR
    tail *= np.exp(-ti / 0.9)[:, None]
    # level-match the episode clips
    gains = [0.12 / (np.sqrt((p ** 2).mean()) + 1e-9) for p in parts]
    parts = [p * g for p, g in zip(parts, gains)]
    tail *= gains[-1]
    idle = idle / (np.sqrt((idle ** 2).mean()) + 1e-9) * 0.08
    ii = np.arange(len(idle)) / SR
    idle *= smoothstep(0.0, 1.5, ii)[:, None]
    mix = np.zeros((N + SR, 2), np.float32)
    pos = 0
    xf = int(0.3 * SR)
    mix[:len(idle)] += idle * (1 - smoothstep(IDLE - 0.3, IDLE + 0.3, ii))[:, None]
    pos = int(IDLE * SR)
    for k, p in enumerate(parts):
        r = np.ones(len(p))
        s = int(0.03 * SR) if k else xf     # short crossfades on the hard cuts
        r[:s] = np.linspace(0, 1, s)
        mix[pos:pos + len(p)] += p * r[:, None]
        pos += len(p)
    mix[pos:pos + len(tail)] += tail
    mix = mix[:N]
    mix = np.tanh(mix * 1.6) / np.tanh(1.6) * 0.95
    raw = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    import wave
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(raw.tobytes())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "out/v9.mp4"
    n = video("out/_v9_video.mp4")
    audio("out/_v9_audio.wav", n)
    subprocess.run([FF, "-y", "-v", "error", "-i", "out/_v9_video.mp4", "-i", "out/_v9_audio.wav",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
                   check=True)
    print("wrote", out, f"{n / FPS:.1f} s")


if __name__ == "__main__":
    main()
