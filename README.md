# Blood Chain Bankai — live wallpaper

**v9 (current):** `montage.py` builds a 55 s loop from the real TYBW animation:
the red crescent moon (idle, animated in code, official Blood Chains Theme OST) →
"The... moon?" ... "...Ban-kai." → eyes, molten coat, black blades → "Getsuga...
tensho!!" → blackout, starburst, crescent close-up → the black moon eclipse → fade
to black. Burned-in subtitles in the source clips are painted out automatically.
The episode clips and OST are not in this repo; `montage.py` lists the YouTube IDs.

Everything below describes the earlier code-drawn versions (v1–v8).

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
| `sound.py` | Builds the sound effects (seamless loop, event times read from `render.py`). `--sfx-only` drops the ambient bed for mixing under music. |
| `music.py` | Fits a music track to the loop: finds the tempo, picks the loudest 18 bars, stretches them to 90 BPM (48 s = 4 video loops), loops the seam, and puts a downbeat on the Getsuga blast. |
| `frames/base.png` | The source still art. Everything is built on it. Do not delete. |
| `render_v*_backup.py` | Earlier versions of the renderer. |

## Build

```bash
python3 -m venv .venv && .venv/bin/pip install numpy opencv-python-headless imageio-ffmpeg
ln -s .venv/lib/python3.*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-* ffmpeg
.venv/bin/python render.py video out/v8_silent.mp4      # ~10-15 min on 8 threads
.venv/bin/python sound.py out/v8_sfx.wav --sfx-only
.venv/bin/python music.py assets/bankai_track.wav out/v8_sfx.wav out/v8_music.wav
./ffmpeg -stream_loop 3 -i out/v8_silent.mp4 -i out/v8_music.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -t 48 out/v8.mp4
```

The music is "Ichigo's True Bankai Theme" (cover by Dude's Cover,
https://www.youtube.com/watch?v=zVt6DzEKpZ0). It is not included in this repo;
download it yourself to `assets/bankai_track.wav`.

## Set as wallpaper

```bash
gsettings set io.github.jeffshee.hanabi-extension video-path ~/Videos/wallpapers/blood_chain_bankai_dynamic_v8.mp4
gsettings set io.github.jeffshee.hanabi-extension mute false   # to hear the sound
```
