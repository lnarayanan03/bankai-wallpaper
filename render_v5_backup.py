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


# Ichigo's silhouette (hand-traced at 1080p) and the crop his motion lives in
FIG_POLY = np.array([(940, 240), (966, 230), (993, 250), (1000, 286), (1046, 323), (1080, 360), (1103, 406),
                     (1116, 460), (1120, 506), (1106, 546), (1103, 576), (1070, 576), (1063, 613), (1056, 666),
                     (1040, 706), (986, 733), (970, 766), (953, 806), (940, 850), (916, 850), (916, 790),
                     (893, 753), (856, 713), (843, 666), (850, 613), (863, 573), (836, 556), (806, 553),
                     (793, 520), (810, 473), (836, 420), (876, 376), (916, 323), (920, 266)], np.int32)
FX0, FY0, FX1, FY1 = 600, 150, 1400, 960
BODY_C = np.array([965.0, 420.0])   # breathing centre (chest)
HAND = np.array([800.0, 548.0])     # sword hand, pivot of the blade
EYES = ((953, 290), (971, 290))
EMBLEM = (963, 353)
# ends of the reiatsu flames already painted on his arms; sparks also leap from these
TIPS = ((1240, 330), (1075, 240), (1155, 600), (790, 380), (1110, 210))

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
    init_figure()
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


def charge_env(tau):
    """Builds up over the last second before each burst (the wind-up)."""
    return float(smoothstep(LOOP / BURSTS - 1.1, LOOP / BURSTS - 0.05, tau)) ** 2


def flow_layer(t):
    """Background energy streaming outward from the eclipse (two-phase flow map)."""
    # flow_src has the sword painted out; the sword is pasted back as a moving sprite
    base, fw = G["flow_src"], G["flow_w"]
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


def aniso_noise(h, w, sx, sy, seed):
    """Smooth random field in [-1, 1], stretched sx by sy pixels per cell."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(-1, 1, (h // sy + 3, w // sx + 3)).astype(np.float32)
    big = cv2.resize(small, ((w // sx + 3) * sx, (h // sy + 3) * sy), interpolation=cv2.INTER_CUBIC)
    return big[sy:sy + h, sx:sx + w]


FLAME_S = 160  # px a flame tongue rises per flame cycle


def init_figure():
    ch, cw = FY1 - FY0, FX1 - FX0
    hard = np.zeros((H, W), np.uint8)
    cv2.fillPoly(hard, [FIG_POLY], 255)
    # motion weight: 1 on the body, fading out over the black space around it
    soft = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    G["fig_w"] = cv2.GaussianBlur(soft.astype(np.float32) / 255, (0, 0), 16)
    hc = hard[FY0:FY1, FX0:FX1]
    G["d_out"] = cv2.distanceTransform(255 - hc, cv2.DIST_L2, 5)
    G["d_in"] = cv2.distanceTransform(hc, cv2.DIST_L2, 5)
    ys, xs = np.mgrid[FY0:FY1, FX0:FX1].astype(np.float32)
    G["fxs"], G["fys"] = xs, ys
    G["lx"], G["ly"] = xs - FX0, ys - FY0
    G["hem"] = smoothstep(540, 830, ys)
    G["hair"] = smoothstep(305, 238, ys)
    G["n"] = [value_noise((ch, cw), 70, 20 + i) for i in range(6)]
    # flames: tall stretched noise that scrolls upward
    G["flame_n"] = [aniso_noise(ch + FLAME_S, cw, 14, 38, 30 + i) for i in range(2)]
    # spark origins: points on the silhouette outline + the painted flame tips
    cnt = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0][:, 0, :]
    G["outline"] = cnt[::6].astype(float)
    init_sparks()


def body_motion(t, tau, e, c):
    """Hover offset and breathing scale of the whole figure."""
    bx = 3.0 * math.sin(TAU * t / LOOP + 1.0)
    by = 7.0 * math.sin(TAU * 2 * t / LOOP)
    by += 5.0 * c - 11.0 * e          # sinks while charging, kicks up on release
    s = 0.009 * math.sin(TAU * 4 * t / LOOP) - 0.012 * c + 0.02 * e
    return bx, by, s


def point_offset(p, mot):
    bx, by, s = mot
    w = G["fig_w"][int(min(max(p[1], 0), H - 1)), int(min(max(p[0], 0), W - 1))]
    return np.array([bx + (p[0] - BODY_C[0]) * s, by + (p[1] - BODY_C[1]) * s]) * w


def loop_noise(i, j, cycles, t):
    """Noise field that changes over time and returns to its start every LOOP/cycles."""
    a = TAU * cycles * t / LOOP
    return G["n"][i] * math.cos(a) + G["n"][j] * math.sin(a)


def figure_layer(img, t, e, c, mot):
    """Animate Ichigo inside img (in place): black-reiatsu aura, living outline,
    then a warp for hovering, breathing, coat flutter and hair flicker."""
    fig = img[FY0:FY1, FX0:FX1].copy()
    d_out, d_in = G["d_out"], G["d_in"]

    # flame field rising off his body (two scrolling phases, crossfaded -> loops)
    P = LOOP / 6
    ch = FY1 - FY0
    F = np.zeros_like(d_out)
    for k, off in enumerate((0.0, 0.5)):
        ph = (t / P + off) % 1.0
        o = int(ph * FLAME_S)
        F += G["flame_n"][k][o:o + ch] * (1 - abs(1 - 2 * ph))
    F = F + 0.35 * loop_noise(0, 1, 6, t)

    # the painted glow along his outline flickers like fire
    band = np.exp(-d_out / 28) * (d_in < 14)
    fig *= (1 + (0.35 * F + 0.15 * e) * band)[..., None]

    # black reiatsu: dark flame tongues with blood-red edges
    reach = 14 + 20 * c + 34 * e
    lvl = (0.42 + 0.6 * F - d_out / reach) * (d_in < 10)     # >0 inside a flame tongue
    core = np.clip(lvl * 3, 0, 1)
    rim = np.exp(-(lvl / 0.07) ** 2) * (lvl > -0.3) * smoothstep(0.0, 5.0, d_out)  # thin flame outline
    rim *= np.clip(1.3 - d_out / reach, 0, 1)
    fig *= (1 - 0.8 * core)[..., None]
    fig += rim[..., None] * np.array([0.04, 0.09, 0.9], np.float32) * (0.6 + 0.6 * c + 0.7 * e)

    # warp: hover + breathing on the whole body, flutter at the hem, flicker in the hair
    bx, by, s = mot
    w = G["fig_w"][FY0:FY1, FX0:FX1]
    X, Y = G["fxs"], G["fys"]
    dx = w * (bx + (X - BODY_C[0]) * s)
    dy = w * (by + (Y - BODY_C[1]) * s)
    hem = G["hem"] * w
    wave = np.sin(TAU * (Y / 120 - 3 * t / LOOP) + X / 140)
    shiver = loop_noise(2, 3, 2, t)
    amp = 1 + 1.8 * e + 0.6 * c
    dx += hem * (4.5 * wave + 3.5 * shiver) * amp
    dy += hem * 1.5 * np.cos(TAU * (Y / 120 - 3 * t / LOOP) + X / 140) * amp
    fur = loop_noise(4, 5, 12, t) * (0.6 + 1.5 * e)   # fast fur shiver
    dx += w * fur
    hair = G["hair"] * w
    dy += hair * (-1.5 - 2.0 * loop_noise(1, 4, 9, t) - 4 * e)
    dx += hair * 2.0 * loop_noise(3, 0, 7, t)
    img[FY0:FY1, FX0:FX1] = cv2.remap(fig, (G["lx"] - dx).astype(np.float32), (G["ly"] - dy).astype(np.float32), cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)


def sword_layer(t, e, c, mot):
    """The blade as a rigid sprite that follows his hand (never warped)."""
    off = point_offset(HAND, mot)
    ang = 1.2 * math.sin(TAU * 2 * t / LOOP - 0.7) - 2.5 * e + 1.0 * c
    M = cv2.getRotationMatrix2D((float(HAND[0]), float(HAND[1])), ang, 1.0)
    M[:, 2] += off
    rgb = cv2.warpAffine(G["base"], M, (W, H), flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(G["sword"], M, (W, H), flags=cv2.INTER_LINEAR)
    return rgb, a


def eyes_layer(e, c, t, mot, add, sglow):
    k = 0.55 + 0.25 * math.sin(TAU * 5 * t / LOOP) ** 2 + 0.8 * c + 2.2 * e
    for p in EYES:
        q = np.array(p, float) + point_offset(p, mot)
        cx, cy = int(q[0] * 4), int(q[1] * 4)
        cv2.circle(add, (cx, cy), 2 * 4, (0.35 * k, 0.5 * k, 1.0 * k), -1, cv2.LINE_AA, 2)
        cv2.circle(sglow, (cx, cy), 7 * 4, (0.05 * k, 0.1 * k, 0.9 * k), -1, cv2.LINE_AA, 2)
    q = np.array(EMBLEM, float) + point_offset(EMBLEM, mot)
    pulse = 0.5 + 0.5 * math.sin(TAU * 4 * t / LOOP) ** 4 + 1.2 * e
    cv2.circle(sglow, (int(q[0] * 4), int(q[1] * 4)), 16 * 4, (0.04 * pulse, 0.08 * pulse, 0.8 * pulse),
               -1, cv2.LINE_AA, 2)


def init_sparks():
    """Spark events. Each one lives a few frames; ages wrap modulo the loop so
    the video still loops cleanly."""
    rng = np.random.default_rng(11)
    ev = []

    def add_ev(start, big=False, tip=None):
        if tip is None:
            tip = rng.random() < 0.35
        if tip:
            o = np.array(TIPS[int(rng.integers(len(TIPS)))], float)
        else:
            o = G["outline"][int(rng.integers(len(G["outline"])))]
        d = o - np.array([965.0, 520.0])
        ang = math.atan2(d[1], d[0]) + rng.uniform(-0.8, 0.8)
        ln = rng.uniform(220, 420) if big else rng.uniform(35, 130)
        ev.append(dict(start=int(start) % NF, life=int(rng.integers(6, 11) if big else rng.integers(3, 8)),
                       o=o, ang=ang, ln=ln, seed=int(rng.integers(1 << 30)), big=big,
                       nb=int(rng.integers(2, 5) if big else rng.integers(0, 3))))

    for _ in range(int(NF / 4.5)):            # ambient crackle
        add_ev(rng.uniform(0, NF))
    for b in range(BURSTS):
        fb = b * NF / BURSTS
        for _ in range(14):                   # wind-up: crackle intensifies
            add_ev(fb - rng.uniform(0, 1.0) ** 0.6 * 60)
        for _ in range(16):                   # release
            add_ev(fb + rng.uniform(0, 1) ** 2 * 30)
        for _ in range(4):                    # long arcs reaching for the eclipse rim
            add_ev(fb + rng.uniform(0, 8), big=True)
    G["sparks"] = ev


def draw_tapered(layer, pts, w0, col, frac=1.0):
    n = max(2, int(len(pts) * frac))
    for i in range(n - 1):
        w = max(1, int(round(w0 * (1 - i / n) * 4)))
        cv2.line(layer, tuple(pts[i]), tuple(pts[i + 1]), col, w, cv2.LINE_AA, 2)


def draw_sparks(frame, mot, add, sglow):
    """Thin, branching red crackles that leap off his body and vanish."""
    for ev in G["sparks"]:
        age = (frame - ev["start"]) % NF
        if age >= ev["life"]:
            continue
        rng = np.random.default_rng(ev["seed"] + age // 2)   # re-crackle every 2 frames
        o = ev["o"] + point_offset(ev["o"], mot)
        grow = min(1.0, (age + 1) / 2.0)
        env = (1 - age / ev["life"]) ** 1.3 * (1.6 if age == 0 else 1.0)
        a = ev["ang"] + rng.uniform(-0.12, 0.12)
        p1 = o + np.array([math.cos(a), math.sin(a)]) * ev["ln"]
        pts = bolt(rng, o, p1, depth=6 if ev["big"] else 5, disp=ev["ln"] * 0.22)
        P = (pts * 4).astype(np.int32)
        w0 = 2.4 if ev["big"] else 1.8
        draw_tapered(sglow, P, w0 * 3, (0.02 * env, 0.05 * env, 0.9 * env), grow)
        draw_tapered(add, P, w0, (0.08 * env, 0.16 * env, 1.0 * env), grow)
        draw_tapered(add, P, 1.0, (0.45 * env, 0.6 * env, 1.0 * env), grow * 0.55)  # hot core near the root
        for _ in range(ev["nb"]):           # forks
            j = int(rng.integers(3, max(4, len(pts) * grow - 3)))
            d = pts[min(j + 1, len(pts) - 1)] - pts[j]
            ba = math.atan2(d[1], d[0]) + rng.choice([-1, 1]) * rng.uniform(0.4, 1.0)
            bl = ev["ln"] * rng.uniform(0.2, 0.45)
            q = pts[j] + np.array([math.cos(ba), math.sin(ba)]) * bl
            bp = (bolt(rng, pts[j], q, depth=4, disp=bl * 0.25) * 4).astype(np.int32)
            draw_tapered(add, bp, 1.2, (0.06 * env, 0.12 * env, 0.85 * env), grow)
            draw_tapered(sglow, bp, 3.5, (0.02 * env, 0.04 * env, 0.6 * env), grow)


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

    c = charge_env(tau)
    mot = body_motion(t, tau, e, c)

    img = flow_layer(t)
    figure_layer(img, t, e, c, mot)
    srgb, sa = sword_layer(t, e, c, mot)
    sa3 = sa[..., None]
    img = img * (1 - sa3) + srgb * sa3
    # the sword is in front of the eclipse: ring glow and shockwave go behind it
    behind = 1 - sa3
    img = img + corona(t, e + 0.4 * c) * behind
    sw = shockwave(tau)
    if sw is not None:
        img = img + sw * behind

    # effect layers (drawn with 2-bit subpixel precision -> coords x4)
    glow = np.zeros_like(img)
    add = np.zeros_like(img)
    sglow = np.zeros_like(img)
    draw_embers(t, add)
    draw_sparks(frame, mot, add, sglow)
    eyes_layer(e, c, t, mot, add, sglow)
    add += cv2.GaussianBlur(sglow, (0, 0), 4) * 0.9 + cv2.GaussianBlur(sglow, (0, 0), 14) * 0.5
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
    sx, sy = rng.uniform(-1, 1, 2) * 7 * e
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
        for f in (0, 4, 30, 120, 215, 236, 400):
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
