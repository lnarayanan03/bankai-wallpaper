#!/usr/bin/env python3
"""Blood Chain Bankai - procedural motion wallpaper.

Takes the still artwork from the original wallpaper and adds animated layers:
outward-flowing reiatsu, rotating eclipse corona, swaying blood chains,
rising embers, black-red lightning and periodic power bursts (shockwave,
camera punch, chromatic aberration). Every effect is periodic in LOOP
seconds, so the output loops seamlessly.

Usage:
  render.py preview            # writes a few still frames to out/preview_*.png
  render.py video [out.mp4]    # renders the full loop
"""
import math
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np

SRC = "/home/ranga/Downloads/ichigo-bankai-blood-chains-eclipse-moewalls-com.mp4"
BASE = "frames/base.png"  # median frame of the source (static artwork)
W, H = 1920, 1080
FPS = 60
LOOP = 12.0                     # seconds; all motion repeats exactly
NF = int(LOOP * FPS)
BURSTS = 3                      # power bursts per loop
CX, CY, R = 973.0, 478.0, 399.0  # eclipse centre / radius at 1080p

TAU = 2 * math.pi
CY_C = np.array([CX, CY])
G = {}  # per-process precomputed data


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def value_noise(shape, scale, seed):
    """Smooth random field in [-1, 1]."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(-1, 1, (shape[0] // scale + 3, shape[1] // scale + 3)).astype(np.float32)
    big = cv2.resize(small, (shape[1] + 3 * scale, shape[0] + 3 * scale), interpolation=cv2.INTER_CUBIC)
    return big[scale:scale + shape[0], scale:scale + shape[1]]


SWORD_POLY = np.array([(472, 768), (478, 818), (498, 846), (808, 622), (840, 540), (770, 560), (715, 562)], np.int32)


def sword_mask():
    """1 on the sword (with a soft feathered edge), 0 elsewhere."""
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [SWORD_POLY], 255)
    m = cv2.dilate(m, np.ones((15, 15), np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 5)


CORONA_HARM = ((5, 1, 0.3, 0.5), (9, -2, 1.7, 0.35), (14, 3, 4.1, 0.3), (23, -4, 2.2, 0.2), (37, 5, 0.9, 0.15))


def init():
    base = cv2.imread(BASE).astype(np.float32) / 255.0
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_AREA)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xs - CX, ys - CY
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    th = np.arctan2(dy, dx)
    G.update(base=base, xs=xs, ys=ys, r=r, th=th, ux=dx / r, uy=dy / r)
    G["harm"] = [(np.cos(n * th).astype(np.float32), np.sin(n * th).astype(np.float32)) for n, *_ in CORONA_HARM]
    # how strongly the background flows: 0 on the eclipse, 1 further out
    G["flow_w"] = smoothstep(R * 1.08, R * 1.45, r)
    # keep Ichigo's sword solid: the flow warp and energy waves must not touch it
    G["sword"] = sword_mask()
    G["flow_w"] = G["flow_w"] * (1 - G["sword"])
    # the flow samples pixels from nearer the eclipse, which would drag ghost
    # copies of the sword outward. Flow from a copy with the sword painted out.
    hole = cv2.dilate((G["sword"] > 0.02).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    b8 = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    G["flow_src"] = cv2.inpaint(b8, hole, 9, cv2.INPAINT_TELEA).astype(np.float32) / 255
    G["phase_noise"] = value_noise((H, W), 160, 1) * 0.5
    G["wob_x"] = value_noise((H, W), 90, 2)
    G["wob_y"] = value_noise((H, W), 90, 3)
    yy = (ys / H - 0.5) * 1.6
    xx = (xs / W - 0.5) * 1.3
    G["vignette"] = np.clip(1.15 - (xx * xx + yy * yy) * 0.9, 0.35, 1)[..., None]

    rng = np.random.default_rng(7)
    # embers: vertical speed is an integer number of screen heights per loop
    n = 420
    G["emb"] = dict(
        x=rng.uniform(0, W, n), y=rng.uniform(0, H + 60, n),
        k=rng.integers(1, 4, n), amp=rng.uniform(5, 40, n),
        m=rng.integers(1, 4, n), ph=rng.uniform(0, TAU, n),
        size=rng.choice([1, 1, 2, 2, 3], n), bri=rng.uniform(0.4, 1.0, n),
        fl=rng.integers(3, 9, n),
    )
    # chains: anchor off-screen, end near the figure
    chains = []
    # (anchor x, y, rim angle in degrees, bow direction); none crosses the sword (lower left)
    anchors = [(-80, 140, 200, -1), (-80, 520, 188, -1), (W + 80, 60, -25, -1), (W + 80, 860, 30, 1), (1550, -80, -60, -1)]
    specs = []
    for ax, ay, deg, _ in anchors:  # chain end sits on the rim, angle in degrees
        a = math.radians(deg)
        specs.append((ax, ay, CX + math.cos(a) * R * 1.03, CY + math.sin(a) * R * 1.03))
    for i, (ax, ay, bx, by) in enumerate(specs):
        chains.append(dict(a=np.array([ax, ay], float), b=np.array([bx, by], float),
                           sag=rng.uniform(60, 140) * anchors[i][3],
                           m=int(rng.integers(1, 3)), ph=rng.uniform(0, TAU),
                           scroll=int(rng.choice([-2, -1, 1, 2]))))
    G["chains"] = chains


def burst_env(t):
    """Seconds since the last burst, and a decaying envelope."""
    period = LOOP / BURSTS
    tau = (t + 0.0) % period
    return tau, math.exp(-tau / 0.35)


def flow_layer(t):
    """Background energy streaming outward from the eclipse (two-phase flow map)."""
    base, fw = G["base"], G["flow_w"]
    P = LOOP / 4                    # flow cycle length
    D = 70.0                        # px travelled per cycle
    out = np.zeros_like(base)
    wob = 6.0 * math.sin(TAU * t / LOOP)
    wob2 = 6.0 * math.cos(TAU * 2 * t / LOOP)
    for off in (0.0, 0.5):
        ph = (t / P + off + G["phase_noise"]) % 1.0
        d = ph * D * fw * (0.6 + G["r"] / 1400)
        swirl = d * 0.35
        mx = G["xs"] - G["ux"] * d + G["uy"] * swirl + G["wob_x"] * wob * fw
        my = G["ys"] - G["uy"] * d - G["ux"] * swirl + G["wob_y"] * wob2 * fw
        s = cv2.remap(G["flow_src"], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        w = (1 - np.abs(1 - 2 * ph))[..., None]
        out += s * w
    # flow blend loses a bit of contrast; restore it
    out = np.clip((out - 0.04) * 1.3, 0, 1)
    fw3 = fw[..., None]
    img = base * (1 - fw3) + out * fw3
    # energy waves radiating outward
    wave = 0.5 + 0.5 * np.sin(TAU * (G["r"] / 420 - 3 * t / LOOP))
    wave = 0.6 + 0.7 * wave ** 4
    return img * (1 + (wave[..., None] - 1) * fw3)


def corona(t, e):
    r, th = G["r"], G["th"]
    # cos(n*th - p) = cos(n*th)cos(p) + sin(n*th)sin(p); the tables are precomputed in init()
    fl = np.zeros_like(r)
    for (n, m, ph, a), (cn, sn) in zip(CORONA_HARM, G["harm"]):
        p = -TAU * m * t / LOOP + ph
        fl += cn * (a * math.cos(p)) - sn * (a * math.sin(p))
    fl = np.clip(fl + 0.25, 0, None)
    beat = 0.55 + 0.25 * math.sin(TAU * BURSTS * 2 * t / LOOP) ** 2 + 0.7 * e
    ring = np.exp(-((r - R) / 9) ** 2) * (0.7 + fl)
    halo = np.exp(-np.clip(r - R, 0, None) / (35 + 120 * fl)) * smoothstep(R * 0.97, R * 1.01, r) * fl * 0.9
    inner = np.exp(-np.clip(R - r, 0, None) / 18) * (r < R) * 0.5
    a = (ring + halo + inner) * beat
    col = np.array([0.08, 0.18, 1.0], np.float32)
    hot = np.array([0.6, 0.75, 1.0], np.float32)
    return a[..., None] * col + (np.clip(ring * beat - 0.9, 0, None))[..., None] * hot


def shockwave(tau):
    if tau > 1.4:
        return None
    rad = R + tau * 1300
    a = math.exp(-tau / 0.5)
    band = np.exp(-((G["r"] - rad) / (14 + tau * 50)) ** 2) * a
    return band[..., None] * np.array([0.15, 0.25, 1.0], np.float32)


def bezier(a, c, b, n):
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * a + 2 * (1 - s) * s * c + s ** 2 * b


def draw_chains(t, e, layer, mask, glow):
    """Chains are rim-lit by the eclipse, hazy far from it, and carry blood-red
    energy pulses that travel along them into the eclipse."""
    spacing = 19.0
    red = np.array([0.06, 0.1, 1.0])
    for ch in G["chains"]:
        a, b = ch["a"], ch["b"]
        v = b - a
        L = np.linalg.norm(v)
        nrm = np.array([-v[1], v[0]]) / L
        sway = ch["sag"] + 45 * math.sin(TAU * ch["m"] * t / LOOP + ch["ph"]) + 60 * e * math.sin(ch["ph"] * 3)
        c = (a + b) / 2 + nrm * sway
        pts = bezier(a, c, b, 400)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0], np.cumsum(seg)])
        total = arc[-1]
        # links slide along the chain; integer links per loop keeps it seamless
        shift = (ch["scroll"] * 6 * spacing * t / LOOP) % (2 * spacing)
        k = -2
        while True:
            s = k * spacing + shift
            k += 1
            if s > total:
                break
            if s < 0:
                continue
            idx = np.searchsorted(arc, s)
            idx = min(max(idx, 1), len(pts) - 1)
            p, q = pts[idx - 1], pts[idx]
            ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
            cxy = (int(p[0] * 4), int(p[1] * 4))
            u = s / total                                  # 0 at anchor, 1 at the eclipse
            sink = min(1.0, (total - s) / 50)              # end melts into the rim light
            alpha = (0.75 + 0.25 * u) * sink
            light = 0.35 + 0.65 * u ** 1.4
            pulse = (0.5 + 0.5 * math.cos(TAU * (s / 260 - 5 * t / LOOP))) ** 8 * (1 + e)
            body = tuple(red * (0.05 + 0.12 * light))
            hi = tuple(np.clip(red * (0.6 * light + 1.1 * pulse) + 0.25 * pulse, 0, 2))
            gl = tuple(red * (0.4 * light + 0.9 * pulse) * sink)
            if (k % 2) == 0:
                axes = (12 * 4, 6 * 4)
                cv2.ellipse(layer, cxy, axes, ang, 0, 360, body, 4 * 4, cv2.LINE_AA, 2)
                cv2.ellipse(layer, cxy, axes, ang, 200, 330, hi, 1 * 4, cv2.LINE_AA, 2)
                cv2.ellipse(mask, cxy, axes, ang, 0, 360, alpha, 4 * 4, cv2.LINE_AA, 2)
                cv2.ellipse(glow, cxy, axes, ang, 0, 360, gl, 7 * 4, cv2.LINE_AA, 2)
            else:
                dx, dy = math.cos(math.radians(ang)) * 12 * 4, math.sin(math.radians(ang)) * 12 * 4
                p1 = (int(cxy[0] - dx), int(cxy[1] - dy))
                p2 = (int(cxy[0] + dx), int(cxy[1] + dy))
                cv2.line(layer, p1, p2, body, 4 * 4, cv2.LINE_AA, 2)
                cv2.line(layer, p1, p2, hi, 1 * 4, cv2.LINE_AA, 2)
                cv2.line(mask, p1, p2, alpha, 4 * 4, cv2.LINE_AA, 2)
                cv2.line(glow, p1, p2, gl, 7 * 4, cv2.LINE_AA, 2)
        # burning contact point where the chain pierces the eclipse
        burn = 0.6 + 0.4 * math.sin(TAU * 4 * t / LOOP + ch["ph"]) + e
        cv2.circle(glow, (int(b[0] * 4), int(b[1] * 4)), 26 * 4, tuple(red * 1.2 * burn), -1, cv2.LINE_AA, 2)


def draw_embers(t, add):
    em = G["emb"]
    Hh = H + 60
    y = (em["y"] - em["k"] * Hh * t / LOOP) % Hh - 30
    x = em["x"] + em["amp"] * np.sin(TAU * em["m"] * t / LOOP + em["ph"])
    fl = 0.6 + 0.4 * np.sin(TAU * em["fl"] * t / LOOP + em["ph"])
    for i in range(len(x)):
        b = em["bri"][i] * fl[i]
        cv2.circle(add, (int(x[i] * 4), int(y[i] * 4)), int(em["size"][i] * 4), (0.25 * b, 0.45 * b, 1.0 * b), -1, cv2.LINE_AA, 2)


def bolt(rng, p0, p1, depth=6, disp=55):
    pts = [np.array(p0, float), np.array(p1, float)]
    for _ in range(depth):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            mid = (a + b) / 2
            d = b - a
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
            new += [mid + n * rng.uniform(-disp, disp), b]
        pts = new
        disp *= 0.5
    return np.array(pts)


def draw_lightning(frame, e, add):
    hold = 4  # each bolt pattern lasts 4 frames
    rng = np.random.default_rng(1000 + frame // hold)
    count = int(rng.integers(0, 3)) + (3 if e > 0.5 else 0)
    for _ in range(count):
        a0 = rng.uniform(0, TAU)
        r0 = rng.uniform(0.1, 0.55) * R
        p0 = (CX + math.cos(a0) * r0, CY + math.sin(a0) * r0)
        a1 = a0 + rng.uniform(-1.2, 1.2)
        r1 = rng.uniform(0.85, 1.25) * R
        p1 = (CX + math.cos(a1) * r1, CY + math.sin(a1) * r1)
        pts = (bolt(rng, p0, p1) * 4).astype(np.int32)
        cv2.polylines(add, [pts], False, (0.08, 0.1, 0.9), 4 * 4, cv2.LINE_AA, 2)
        cv2.polylines(add, [pts], False, (0.7, 0.8, 1.0), 1 * 4, cv2.LINE_AA, 2)
        pf = pts / 4.0
        for _ in range(int(rng.integers(1, 3))):  # side branches
            j = int(rng.integers(8, len(pf) - 8))
            d = pf[j] - CY_C
            ang = math.atan2(d[1], d[0]) + rng.uniform(-0.9, 0.9)
            ln = rng.uniform(40, 110)
            q = (pf[j][0] + math.cos(ang) * ln, pf[j][1] + math.sin(ang) * ln)
            bp = (bolt(rng, pf[j], q, depth=4, disp=18) * 4).astype(np.int32)
            cv2.polylines(add, [bp], False, (0.08, 0.1, 0.8), 3 * 4, cv2.LINE_AA, 2)
            cv2.polylines(add, [bp], False, (0.5, 0.6, 1.0), 1 * 4, cv2.LINE_AA, 2)


def bloom(img, strength=0.5):
    small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bright = np.clip(small - 0.6, 0, None)
    b = cv2.GaussianBlur(bright, (0, 0), 6) + cv2.GaussianBlur(bright, (0, 0), 18)
    return img + cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) * strength


def render(frame):
    if not G:
        cv2.setNumThreads(1)
        init()
    t = frame / FPS
    tau, e = burst_env(t)

    img = flow_layer(t)
    # the sword is in front of the eclipse: ring glow and shockwave go behind it
    behind = (1 - G["sword"])[..., None]
    img = img + corona(t, e) * behind
    sw = shockwave(tau)
    if sw is not None:
        img = img + sw * behind

    # effect layers (drawn with 2-bit subpixel precision -> coords x4)
    glow = np.zeros_like(img)
    add = np.zeros_like(img)
    draw_embers(t, add)
    draw_lightning(frame, e, add)
    chains = np.zeros_like(img)
    chain_mask = np.zeros((H, W), np.float32)
    draw_chains(t, e, chains, chain_mask, glow)
    # soften edges so chains don't look cut out
    chains = cv2.GaussianBlur(chains, (0, 0), 0.9)
    chain_mask = cv2.GaussianBlur(chain_mask, (0, 0), 0.9)[..., None]
    glow = cv2.GaussianBlur(glow, (0, 0), 9) * 0.6
    bg = img
    img = img + glow + add
    img = img * (1 - chain_mask) + chains * chain_mask
    # energy wisps drift in front of the chains
    haze = np.clip(cv2.GaussianBlur(bg, (0, 0), 3) - 0.25, 0, None)
    img = img + haze * chain_mask * 0.3

    img = img * (0.9 + 0.35 * e)  # burst flash
    img = bloom(img)

    # chromatic aberration on bursts
    ca = 7 * e
    if ca > 0.5:
        s = 1 + ca / 1000
        M_r = cv2.getRotationMatrix2D((CX, CY), 0, s)
        M_b = cv2.getRotationMatrix2D((CX, CY), 0, 1 / s)
        img[..., 2] = cv2.warpAffine(img[..., 2], M_r, (W, H), borderMode=cv2.BORDER_REFLECT)
        img[..., 0] = cv2.warpAffine(img[..., 0], M_b, (W, H), borderMode=cv2.BORDER_REFLECT)

    # camera: slow breathing + burst punch + shake
    rng = np.random.default_rng(frame)
    zoom = 1.035 + 0.012 * math.sin(TAU * t / LOOP) + 0.03 * e
    sx, sy = rng.uniform(-1, 1, 2) * 10 * e
    sx += 6 * math.sin(TAU * 2 * t / LOOP)
    sy += 4 * math.cos(TAU * t / LOOP)
    M = cv2.getRotationMatrix2D((CX, CY), 0.4 * math.sin(TAU * t / LOOP), zoom)
    M[0, 2] += sx
    M[1, 2] += sy
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    img = img * G["vignette"]
    img = img / (1 + 0.15 * img)  # soft tone map so highlights don't clip hard
    img = np.clip(img - 0.015, 0, None) ** 1.15 * 1.05
    noise = rng.normal(0, 0.012, (H // 2, W // 2, 1)).astype(np.float32)
    img = img + cv2.resize(noise, (W, H))[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if mode == "preview":
        init()
        for f in (0, 6, 30, 120, 260, 400):
            cv2.imwrite(f"out/preview_{f:03d}.png", render(f))
        return
    out = sys.argv[2] if len(sys.argv) > 2 else "out/blood_chain_bankai_dynamic.mp4"
    cmd = ["./ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(8) as pool:
        for i, fr in enumerate(pool.imap(render, range(NF), chunksize=4)):
            enc.stdin.write(fr.tobytes())
            if i % 60 == 0:
                print(f"frame {i}/{NF}", flush=True)
    enc.stdin.close()
    enc.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
