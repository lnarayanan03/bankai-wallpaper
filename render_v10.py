#!/usr/bin/env python3
"""Blood Chain Bankai v10: "Blood Moon".

An original 24 s loop built on the still art (frames/base.png), cut to the
music's bar grid (90 BPM, one bar = 2.667 s):

  0.00-13.33  idle: the living art from render.py (flowing reiatsu, corona,
              chains, embers, breathing figure, power bursts)
 13.33-16.00  push-in on his face; body turns molten, eyes flare, aura grows
 16.00        hard cut to black, two red eyes flash
 16.20-18.67  the eclipse is gone: a huge blazing blood moon, Ichigo a black
              silhouette in front of it. He raises the blade and slashes.
 18.10-18.67  a black Getsuga crescent with a red rim sweeps across the moon
 18.67        downbeat: impact flash, the cut across the moon burns white
 18.90-21.10  the eclipse slides back over the moon (diamond-ring flash at totality)
 21.10-24.00  idle again

Usage:
  render_v10.py preview [seconds...]   stills to out/v10_<t>.png
  render_v10.py video out.mp4          full loop (no audio)
"""
import math
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np

import render as R
from render import G, W, H, CX, CY, TAU, smoothstep, value_noise

FPS = 60
LOOP = 24.0
NF = int(LOOP * FPS)
BAR = 60 / 90 * 4               # 2.667 s

T_PUSH = 5 * BAR                # 13.33 push-in starts
T_CUT = 6 * BAR                 # 16.00 cut to black
T_MOON = T_CUT + 0.2            # 16.20 blood moon
T_RAISE = 17.25                 # sword wind-up
T_SLASH = 18.02                 # slash
T_FLY = 18.10                   # Getsuga leaves the blade
T_HIT = 7 * BAR                 # 18.67 downbeat impact
T_ECL0 = 18.95                  # eclipse starts sliding back
T_ECL1 = 21.10                  # totality
RM = R.R                        # moon radius (the eclipse ring)

STATE = {"T": 0.0}


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = min(max(x, 0.0), 1.0)
    return x ** 3


def pulse(T, t0, attack, decay):
    d = T - t0
    if d < 0:
        return min(1.0, max(0.0, 1 + d / attack)) ** 2 if attack > 0 else 0.0
    return math.exp(-d / decay)


# ---------------------------------------------------------------- setup

def init10():
    R.init()
    ys, xs, r = G["ys"], G["xs"], G["r"]
    # silhouette mask of the figure (drawn black in front of the moon)
    hard = np.zeros((H, W), np.uint8)
    cv2.fillPoly(hard, [R.FIG_POLY], 255)
    hard = cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    G["sil"] = cv2.GaussianBlur(hard.astype(np.float32) / 255, (0, 0), 1.6)

    # moon surface: maria, craters, sphere shading, colour ramp
    rng = np.random.default_rng(42)
    alb = (0.55 * value_noise((H, W), 260, 50) + 0.3 * value_noise((H, W), 110, 51)
           + 0.15 * value_noise((H, W), 40, 52) + 0.08 * value_noise((H, W), 14, 53))
    maria = np.clip(value_noise((H, W), 300, 54) * 1.6 - 0.15, 0, 1)
    alb = 0.62 + 0.22 * alb - 0.28 * maria
    for _ in range(46):
        a, d = rng.uniform(0, TAU), RM * math.sqrt(rng.uniform(0, 0.92))
        cx, cy = CX + d * math.cos(a), CY + d * math.sin(a)
        cr = rng.choice([8, 12, 16, 22, 30, 44, 60], p=[.25, .22, .18, .14, .1, .07, .04])
        dd = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / cr
        floor = (dd < 1) * (0.10 + 0.06 * rng.uniform())
        rim_l = np.exp(-((dd - 1.0) / 0.13) ** 2)
        # light from the upper left: the far rim catches it, the near wall is shadowed
        side = ((xs - cx) * -0.6 + (ys - cy) * -0.8) / (cr * dd + 1e-3)
        alb += -floor + rim_l * 0.12 * np.clip(-side, 0, 1) - rim_l * 0.08 * np.clip(side, 0, 1)
    nz = np.sqrt(np.clip(1 - (r / RM) ** 2, 0, 1))
    shade = 0.35 + 0.65 * nz ** 0.6
    lum = np.clip(alb, 0, 1.2) * shade
    G["moon_lum"] = lum.astype(np.float32)
    G["moon_disk"] = smoothstep(RM * 1.004, RM * 0.988, r).astype(np.float32)
    # outer glow of the lit moon
    G["moon_halo"] = (np.exp(-np.clip(r - RM, 0, None) / 60) * 0.55
                      + np.exp(-np.clip(r - RM, 0, None) / 260) * 0.35).astype(np.float32) * (r >= RM * 0.99)

    # Getsuga path: from the blade, up and across the upper moon, out to the right
    G["gpath"] = np.array([(600, 700), (620, 330), (1080, 210), (1500, 380)], float)
    # streak noise for the crescent's smoky back edge
    G["smoke"] = value_noise((H, W), 22, 60).astype(np.float32)
    G["streak_n"] = cv2.resize(np.random.default_rng(61).uniform(0, 1, (H // 4, 1)).astype(np.float32),
                               (W, H), interpolation=cv2.INTER_LINEAR)


def moon_colour(lum):
    """Blood-red moon colour ramp (BGR), lum in 0..1.2."""
    l = lum[..., None]
    dark = np.array([0.02, 0.02, 0.22], np.float32)
    mid = np.array([0.03, 0.09, 0.80], np.float32)
    hot = np.array([0.30, 0.55, 1.05], np.float32)
    a = np.clip(l / 0.6, 0, 1)
    b = np.clip((l - 0.6) / 0.5, 0, 1)
    return dark * (1 - a) + mid * a * (1 - b) + hot * b


def molten_level10(_t):
    T = STATE["T"]
    up = ease((T - (T_PUSH + 0.4)) / 2.0)
    down = 1 - ease((T - T_CUT) / 0.05)
    return up * down * 1.1


R.molten_level = molten_level10


# ---------------------------------------------------------------- timeline helpers

def bezier3(p, s):
    s = np.asarray(s, float)[..., None]
    return ((1 - s) ** 3 * p[0] + 3 * (1 - s) ** 2 * s * p[1] + 3 * (1 - s) * s * s * p[2] + s ** 3 * p[3])


def moon_level(T):
    """0 = eclipse (idle art), 1 = fully lit blood moon."""
    if T < T_MOON or T >= T_ECL1 + 0.3:
        return 0.0
    return 1.0


def occluder_x(T):
    """Centre x of the dark disk sliding back over the moon (None = no occluder)."""
    if T < T_ECL0:
        return None
    k = ease((T - T_ECL0) / (T_ECL1 - T_ECL0))
    return CX - 2.15 * RM * (1 - k)


def sword_extra(T):
    """Extra blade rotation (degrees, cv2 convention: + = counter-clockwise)."""
    up = ease_out((T - T_RAISE) / 0.55) * -58           # lift the blade back over his shoulder
    down = ease_in((T - T_SLASH) / 0.09) * 105          # whip it through
    settle = ease((T - (T_SLASH + 0.5)) / 1.4) * -47    # back to rest
    return up + down + settle


def burst10(T):
    """Burst envelope: the idle bursts at 0/4/8/12 s, plus the climax impacts."""
    t = T % 12.0
    tau = t % 4.0
    e = math.exp(-tau / 0.35)
    if T >= T_PUSH - 0.5 and T < T_ECL1 + 0.8:
        e = 0.0
    if T >= 22.7:   # wind-up before the burst at 0
        pass
    e = max(e, 0.8 * pulse(T, T_SLASH + 0.06, 0, 0.18), 1.2 * pulse(T, T_HIT, 0, 0.32),
            0.5 * pulse(T, T_ECL1, 0, 0.4))
    return tau, e


def charge10(T, tau):
    if T_PUSH <= T < T_CUT:
        return ease((T - T_PUSH) / (T_CUT - T_PUSH)) ** 1.5 * 1.3
    if T_RAISE <= T < T_SLASH:
        return 0.8 * ease((T - T_RAISE) / (T_SLASH - T_RAISE))
    if T_PUSH - 0.5 <= T < T_ECL1 + 0.8:
        return 0.0
    return float(R.smoothstep(4.0 - 1.1, 4.0 - 0.05, tau)) ** 2


# ---------------------------------------------------------------- layers

def sword_layer10(t, T, e, c, mot):
    off = R.point_offset(R.HAND, mot)
    ang = 1.2 * math.sin(TAU * 2 * t / R.LOOP - 0.7) - 2.5 * e * (T < T_PUSH or T > T_ECL1) + 1.0 * c
    ang += sword_extra(T)
    M = cv2.getRotationMatrix2D((float(R.HAND[0]), float(R.HAND[1])), ang, 1.0)
    M[:, 2] += off
    rgb = cv2.warpAffine(G["base"], M, (W, H), flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(G["sword"], M, (W, H), flags=cv2.INTER_LINEAR)
    return rgb, a, M


def sil_mask(mot):
    bx, by, s = mot
    M = cv2.getRotationMatrix2D((float(R.BODY_C[0]), float(R.BODY_C[1])), 0, 1 + s)
    M[0, 2] += bx
    M[1, 2] += by
    return cv2.warpAffine(G["sil"], M, (W, H), flags=cv2.INTER_LINEAR)


def blade_tip(M):
    tip = np.array([485.0, 800.0, 1.0])
    return M @ tip


def crescent_mask(c, ang, size, T):
    """Solid crescent (convex side facing direction ang) + rim band, smoky back."""
    xs, ys = G["xs"], G["ys"]
    ux, uy = math.cos(ang), math.sin(ang)
    dx, dy = xs - c[0], ys - c[1]
    # rotate into the crescent's frame: u = along travel, v = across
    u = dx * ux + dy * uy
    v = -dx * uy + dy * ux
    ro = size
    ri = size * 0.93
    off = size * 0.30
    d_out = np.sqrt(u * u + v * v) - ro
    d_in = np.sqrt((u + off) ** 2 + v * v) - ri
    # smoky, ragged back edge
    rag = G["smoke"] * size * 0.03
    sdf = np.maximum(d_out, -(d_in + rag))
    body = smoothstep(1.5, -1.5, sdf)
    rim = np.exp(-(np.abs(d_out) / 3.0) ** 2) * (d_in > -2) * (np.abs(v) < size * 0.98)
    rim += 0.6 * np.exp(-(np.abs(d_in + rag) / 2.0) ** 2) * (d_out < 0)
    return body.astype(np.float32), np.clip(rim, 0, 1.5).astype(np.float32), u, v


def getsuga_layer(img, T, M_sword):
    """Black crescent with a red rim flying along the path. Returns (img, cut_mask)."""
    k = (T - T_FLY) / (T_HIT - T_FLY)
    if k < 0 or k > 1.35:
        return img
    s = gpos(k)
    p = bezier3(G["gpath"], s)
    p2 = bezier3(G["gpath"], min(s + 0.02, 1.0))
    ang = math.atan2(p2[1] - p[1], p2[0] - p[0])
    size = 150 + 260 * ease_out(min(k * 1.6, 1.0))
    fade = 1 - ease((k - 1.0) / 0.35)
    body, rim, u, v = crescent_mask(p, ang, size, T)
    # motion trail: a few stretched echoes behind it
    trail = np.zeros_like(body)
    for i in range(6):
        back = 0.012 * (i + 1)
        q = bezier3(G["gpath"], max(s - back, 0))
        b2, _, _, _ = crescent_mask(q, ang, size * (1 - back), T)
        trail = np.maximum(trail, b2 * (0.5 - i * 0.075))
    trail = cv2.GaussianBlur(trail, (0, 0), 3)
    ink = np.clip(np.maximum(body, trail * 0.85), 0, 1) * fade
    img = img * (1 - ink[..., None] * 0.97)
    red = np.array([0.10, 0.12, 1.0], np.float32)
    rim_glow = cv2.GaussianBlur(rim, (0, 0), 6)
    img = img + (rim[..., None] * np.array([0.6, 0.7, 1.0], np.float32) * 0.9 + rim_glow[..., None] * red * 1.4) * fade
    return img


def gpos(k):
    """Position along the Getsuga path: fast launch, steady sweep, leaves after the hit."""
    k = max(k, 0.0)
    if k <= 1.0:
        return 0.25 * ease_out(min(k / 0.2, 1.0)) + 0.6 * (max(k - 0.2, 0) / 0.8)
    return min(0.85 + 0.15 * (k - 1.0) / 0.35, 1.0)


def cut_line(T):
    """Where the Getsuga passed, the moon burns: a bright arc line that cools."""
    if T < T_FLY + 0.1:
        return None
    k = (T - T_FLY) / (T_HIT - T_FLY)
    n = 160
    s = np.linspace(0, gpos(k), n)
    pts = bezier3(G["gpath"], s)
    m = np.zeros((H, W), np.float32)
    for i in range(n - 1):
        w = 2 + 5 * math.sin(math.pi * i / (n - 1))
        cv2.line(m, tuple((pts[i] * 4).astype(int)), tuple((pts[i + 1] * 4).astype(int)), 1.0,
                 int(w * 4), cv2.LINE_AA, 2)
    return m


def moon_layer(img, T, mot, sa):
    lvl = moon_level(T)
    if lvl <= 0:
        return img
    lum = G["moon_lum"].copy()
    # heat: the moon blazes right after the cut, again at the impact
    heat = 0.25 * pulse(T, T_MOON, 0, 0.5) + 0.5 * pulse(T, T_HIT, 0, 0.35)
    lum = lum * (1.0 + heat) + heat * 0.25
    # slow churning shimmer
    lum = lum + 0.04 * np.sin(TAU * (G["r"] / 180 - T * 0.6) + G["th"] * 3)
    col = moon_colour(lum)
    disk = G["moon_disk"].copy()
    halo = G["moon_halo"].copy()
    ox = occluder_x(T)
    if ox is not None:
        rr = np.sqrt((G["xs"] - ox) ** 2 + (G["ys"] - CY) ** 2)
        occ = smoothstep(RM * 1.003, RM * 0.99, rr)
        cover = 1 - occ
        disk = disk * cover
        # the halo shrinks with the visible sliver
        vis = float(disk.sum() / (G["moon_disk"].sum() + 1))
        halo = halo * (0.15 + 0.85 * vis)
        # diamond ring: a last bright bead where the sliver closes
        dr = pulse(T, T_ECL1 - 0.05, 0.25, 0.35)
        if dr > 0.01:
            bead = np.array([CX + RM, CY])
            d = np.sqrt((G["xs"] - bead[0]) ** 2 + (G["ys"] - bead[1]) ** 2)
            star = np.exp(-d / 18) + 0.5 * np.exp(-d / 90)
            ang = np.abs(np.arctan2(G["ys"] - bead[1], G["xs"] - bead[0]))
            star = star + 0.6 * np.exp(-d / 420) * (np.exp(-(np.sin(ang) * d / 3) ** 2) + np.exp(-(np.cos(ang) * d / 3) ** 2))
            img = img + (dr * star)[..., None] * np.array([0.75, 0.85, 1.0], np.float32)
    sil = np.clip(sil_mask(mot) + sa[..., 0], 0, 1)
    vis = (disk * (1 - sil))[..., None] * lvl
    # Ichigo's flames stay on top as rim light: keep the art's bright pixels
    art_light = np.clip(img - 0.08, 0, None) * sil[..., None]
    img = img * (1 - vis) + col * vis + art_light * 0.0
    img = img + halo[..., None] * np.array([0.04, 0.10, 0.95], np.float32) * lvl * (1 + heat)
    # rim-light on his silhouette edge from the moon behind him
    edge = np.clip(cv2.GaussianBlur(sil, (0, 0), 2.5) - sil, 0, 1) * G["moon_disk"]
    img = img + (edge * 2.2)[..., None] * np.array([0.12, 0.22, 1.0], np.float32) * lvl
    # the Getsuga's burn line across the moon
    cl = cut_line(T)
    if cl is not None:
        heat_c = 0.4 + 1.6 * pulse(T, T_HIT, 0.05, 0.5)
        cool = 1 - ease((T - (T_HIT + 0.3)) / 2.0)
        cl = cl * disk * (1 - sil) * cool
        glow = cv2.GaussianBlur(cl, (0, 0), 10)
        img = img + (cl * heat_c)[..., None] * np.array([0.8, 0.9, 1.0], np.float32)
        img = img + (glow * heat_c * 2.0)[..., None] * np.array([0.1, 0.3, 1.0], np.float32)
    return img


def speed_streaks(img, T):
    """Thin horizontal light streaks for the push-in."""
    k = ease((T - (T_PUSH + 1.0)) / 1.6) if T_PUSH <= T < T_CUT else 0.0
    if k <= 0:
        return img
    n = G["streak_n"]
    lines = np.clip((n - 0.86) * 8, 0, 1)
    shift = int(T * 2400) % W
    lines = np.roll(lines * np.clip(G["streak_n"][:, ::-1] * 1.5, 0, 1), shift, axis=1)
    centre = smoothstep(260, 900, np.abs(G["xs"] - 962) + np.abs(G["ys"] - 300) * 1.4)
    return img + (lines * centre * k * 0.35)[..., None] * np.array([0.1, 0.15, 1.0], np.float32)


def eye_flash(T):
    """Black frame with two red eyes and an anamorphic streak (the cut)."""
    img = np.zeros((H, W, 3), np.float32)
    k = pulse(T, T_CUT + 0.07, 0.06, 0.16)
    xs, ys = G["xs"], G["ys"]
    for ex, tilt in ((770, 9), (1150, -9)):
        ey = 520
        dx, dy = xs - ex, ys - ey
        a = math.radians(tilt)
        u = dx * math.cos(a) + dy * math.sin(a)
        v = -dx * math.sin(a) + dy * math.cos(a)
        # sharp almond eye, narrower towards the nose
        half = 26 * np.clip(1 - (u / 150) ** 2, 0, 1) * (1 - 0.35 * np.clip(u * np.sign(-tilt) / 150, 0, 1))
        inside = smoothstep(1.5, -1.5, np.abs(v) - half)
        glow = np.exp(-np.maximum(np.abs(v) - half, 0) / 22) * np.exp(-np.maximum(np.abs(u) - 150, 0) / 12)
        iris = np.exp(-((u - 10 * np.sign(tilt)) ** 2 + v * v) / (2 * 14 ** 2))
        img += (inside * 0.9 + glow * 0.45)[..., None] * np.array([0.05, 0.10, 1.0], np.float32) * k
        img += (iris * inside)[..., None] * np.array([0.6, 0.75, 1.0], np.float32) * k
    streak = np.exp(-np.abs(ys - 520) / 2.5) * np.exp(-np.abs(xs - 962) / 700) * k
    img += streak[..., None] * np.array([0.08, 0.12, 0.9], np.float32) * 0.7
    return img


# ---------------------------------------------------------------- frame

def camera(T, t, e, frame):
    rng = np.random.default_rng(frame + 1000)
    zoom = 1.035 + 0.012 * math.sin(TAU * t / 12) + 0.03 * e
    ang = 0.4 * math.sin(TAU * t / 12)
    piv = (CX, CY)
    sx, sy = rng.uniform(-1, 1, 2) * 7 * e
    sx += 6 * math.sin(TAU * 2 * t / 12)
    sy += 4 * math.cos(TAU * t / 12)
    if T_PUSH <= T < T_CUT:
        k = ease_in((T - T_PUSH) / (T_CUT - T_PUSH)) * 0.6 + ease((T - T_PUSH) / (T_CUT - T_PUSH)) * 0.4
        zoom = zoom + 0.42 * k
        piv = (CX + (962 - CX) * min(1, k * 3), CY + (300 - CY) * min(1, k * 3))
        sh = 4 * k
        sx += rng.uniform(-1, 1) * sh
        sy += rng.uniform(-1, 1) * sh
    elif T_MOON <= T < T_ECL1 + 0.8:
        # cut in wide-but-close, then a slow pull back
        k = ease_out((T - T_MOON) / (T_ECL1 + 0.8 - T_MOON))
        zoom = 1.13 - 0.095 * k + 0.035 * e
        ang = 0.0
        sx, sy = rng.uniform(-1, 1, 2) * 16 * e
    return cv2.getRotationMatrix2D(piv, ang, zoom), sx, sy


def render10(frame):
    if "moon_lum" not in G:
        cv2.setNumThreads(1)
        init10()
    T = frame / FPS
    STATE["T"] = T
    if T_CUT <= T < T_MOON:
        img = eye_flash(T)
        rng = np.random.default_rng(frame)
        noise = rng.normal(0, 0.012, (H // 2, W // 2, 1)).astype(np.float32)
        img = img + cv2.resize(noise, (W, H))[..., None]
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    t = T % 12.0
    tau, e = burst10(T)
    c = charge10(T, tau)
    mot = R.body_motion(t, tau, e, c)

    img = R.flow_layer(t)
    R.figure_layer(img, t, e, c, mot)
    srgb, sa, M_sword = sword_layer10(t, T, e, c, mot)
    sa3 = sa[..., None]
    img = img * (1 - sa3) + srgb * sa3
    behind = 1 - sa3
    img = img + R.corona(t, e + 0.4 * c) * behind
    for t0, gain in ((T - T_HIT, 1.6), (tau if (T < T_PUSH - 0.5 or T > T_ECL1 + 0.8) else 99, 1.0)):
        sw = R.shockwave(t0) if t0 >= 0 else None
        if sw is not None:
            img = img + sw * behind * gain

    glow = np.zeros_like(img)
    add = np.zeros_like(img)
    sglow = np.zeros_like(img)
    R.draw_embers(t, add)
    R.draw_sparks(frame % R.NF, mot, add, sglow)
    R.eyes_layer(e, c * (1 + 1.5 * (T_PUSH <= T < T_CUT)), t, mot, add, sglow)
    add += cv2.GaussianBlur(sglow, (0, 0), 4) * 0.9 + cv2.GaussianBlur(sglow, (0, 0), 14) * 0.5
    chains = np.zeros_like(img)
    chain_mask = np.zeros((H, W), np.float32)
    R.draw_chains(t, e, chains, chain_mask, glow)
    chains = cv2.GaussianBlur(chains, (0, 0), 0.9)
    chain_mask = cv2.GaussianBlur(chain_mask, (0, 0), 0.9)[..., None]
    glow = cv2.GaussianBlur(glow, (0, 0), 9) * 0.6
    bg = img
    img = img + glow + add
    img = img * (1 - chain_mask) + chains * chain_mask
    haze = np.clip(cv2.GaussianBlur(bg, (0, 0), 3) - 0.25, 0, None)
    img = img + haze * chain_mask * 0.3

    img = moon_layer(img, T, mot, sa3)
    img = getsuga_layer(img, T, M_sword)
    img = speed_streaks(img, T)

    img = img * (0.9 + 0.35 * e)
    img = R.bloom(img)

    # impact: two white frames, then a hot red wash
    if T_HIT <= T < T_HIT + 2 / FPS:
        img = img * 0.25 + 0.9
    else:
        wash = 0.5 * pulse(T, T_HIT, 0, 0.25) + 0.35 * pulse(T, T_MOON, 0, 0.12)
        img = img + wash * np.array([0.25, 0.3, 0.6], np.float32)

    ca = 7 * e
    if ca > 0.5:
        s = 1 + ca / 1000
        M_r = cv2.getRotationMatrix2D((CX, CY), 0, s)
        M_b = cv2.getRotationMatrix2D((CX, CY), 0, 1 / s)
        img[..., 2] = cv2.warpAffine(img[..., 2], M_r, (W, H), borderMode=cv2.BORDER_REFLECT)
        img[..., 0] = cv2.warpAffine(img[..., 0], M_b, (W, H), borderMode=cv2.BORDER_REFLECT)

    M, sx, sy = camera(T, t, e, frame)
    M[0, 2] += sx
    M[1, 2] += sy
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    vig = G["vignette"]
    if T_PUSH <= T < T_CUT:
        k = ease((T - T_PUSH) / (T_CUT - T_PUSH))
        vig = vig ** (1 + 1.2 * k)
    img = img * vig
    img = img / (1 + 0.15 * img)
    img = np.clip(img - 0.015, 0, None) ** 1.15 * 1.05
    rng = np.random.default_rng(frame)
    noise = rng.normal(0, 0.012, (H // 2, W // 2, 1)).astype(np.float32)
    img = img + cv2.resize(noise, (W, H))[..., None]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if mode == "preview":
        ts = [float(a) for a in sys.argv[2:]] or [6.0, 14.8, 16.05, 16.4, 17.6, 18.3, 18.69, 19.3, 20.4, 21.05]
        with Pool(min(8, len(ts))) as pool:
            frs = pool.map(render10, [int(round(x * FPS)) for x in ts])
        for x, fr in zip(ts, frs):
            cv2.imwrite(f"out/v10_{x:05.2f}.png", fr)
        print("wrote", len(ts), "previews")
        return
    out = sys.argv[2] if len(sys.argv) > 2 else "out/blood_chain_bankai_v10.mp4"
    cmd = ["./ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(8) as pool:
        for i, fr in enumerate(pool.imap(render10, range(NF), chunksize=4)):
            enc.stdin.write(fr.tobytes())
            if i % 120 == 0:
                print(f"frame {i}/{NF}", flush=True)
    enc.stdin.close()
    enc.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
