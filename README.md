# Blood Chain Bankai — live wallpaper

A 12-second, seamlessly looping 1080p60 live wallpaper of Ichigo's Blood Chain
bankai, made entirely in code (Python + OpenCV + NumPy) and played on GNOME
with the [Hanabi](https://github.com/jeffshee/gnome-ext-hanabi) extension.

Project log and version history: https://claude.ai/artifact/5grUQ8k12WGeXrSrAz32qj

## What's in the loop (v8)

- 0–6.5 s: the still art, animated: flowing reiatsu, eclipse corona, blood
  chains, embers, sparks, a breathing, hovering figure. Power bursts at 0 s and 4 s.
- 6.5–8 s: shadow forms slash through (black brush strokes with neon-red rims),
  molten cracks spread over his body, then a black/red ray burst.
- 8.0 s: blackout, then a white spiky starburst behind his silhouette.
- 8.13–9.9 s: the Getsuga close-up. A glossy black crescent fills the frame
  against a cropped white moon, with ink smoke and brush streaks.
- 9.9 s: hard cut back to the wide shot with an impact.

The cut order follows the TYBW anime scene
([reference clip](https://www.youtube.com/watch?v=7RIb93r3cuM), about 1:17–1:40).

## Files

| File | What it does |
| --- | --- |
| `render.py` | Renders the video. `render.py preview [frames...]` writes stills to `out/`; `render.py video out.mp4` renders the full loop. |
| `sound.py` | Builds the matching sound track (seamless loop, event times read from `render.py`). |
| `frames/base.png` | The source still art. Everything is built on it. Do not delete. |
| `render_v*_backup.py` | Earlier versions of the renderer. |

## Build

```bash
python3 -m venv .venv && .venv/bin/pip install numpy opencv-python-headless imageio-ffmpeg
ln -s .venv/lib/python3.*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-* ffmpeg
.venv/bin/python render.py video out/v8_silent.mp4      # ~10-15 min on 8 threads
.venv/bin/python sound.py out/v8_audio.wav
./ffmpeg -i out/v8_silent.mp4 -i out/v8_audio.wav -c:v copy -c:a aac -b:a 192k -shortest out/v8.mp4
```

## Set as wallpaper

```bash
gsettings set io.github.jeffshee.hanabi-extension video-path ~/Videos/wallpapers/blood_chain_bankai_dynamic_v8.mp4
gsettings set io.github.jeffshee.hanabi-extension mute false   # to hear the sound
```
