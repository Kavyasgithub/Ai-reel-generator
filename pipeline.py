#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║           AI Reels Generator — CPU Pipeline          ║
║                                                      ║
║  Script   → Google Gemini API                        ║
║  Voice    → Coqui TTS  (CPU-friendly)                ║
║  Visuals  → Stable Diffusion 1.5  (CPU mode)         ║
║  Assembly → FFmpeg                                   ║
╚══════════════════════════════════════════════════════╝

Usage:
    python pipeline.py "The mystery of black holes"
    python pipeline.py "Benefits of cold showers" --images 4 --steps 15
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  –  Edit these before running
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = "AIzaSyBelI7_RiYhd1lt6-295I0ap0lc0mhwFf4"   # https://aistudio.google.com/

# Stable Diffusion model (downloaded automatically on first run, ~4 GB)
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Output directories
OUTPUT_DIR  = Path("output")
IMAGES_DIR  = OUTPUT_DIR / "images"
AUDIO_DIR   = OUTPUT_DIR / "audio"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner(msg: str):
    print(f"\n{'─'*56}")
    print(f"  {msg}")
    print(f"{'─'*56}")

def get_ffmpeg_bin() -> str:
    """Return path to FFmpeg binary from imageio-ffmpeg (pip) or system."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"

FFMPEG_BIN = get_ffmpeg_bin()

def check_ffmpeg():
    try:
        subprocess.run([FFMPEG_BIN, "-version"], capture_output=True, check=True)
        print(f"     FFmpeg ready → {FFMPEG_BIN}")
    except FileNotFoundError:
        print("   FFmpeg not found. Run:  pip install imageio-ffmpeg")
        sys.exit(1)

def get_audio_duration(audio_path: str) -> float:
    """Return duration in seconds by parsing ffmpeg stderr (no ffprobe needed)."""
    result = subprocess.run(
        [FFMPEG_BIN, "-i", audio_path, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    for line in result.stderr.splitlines():
        if "Duration:" in line:
            dur_str = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = dur_str.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"Could not read audio duration from: {audio_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SCRIPT + IMAGE PROMPTS  (Gemini)
# ─────────────────────────────────────────────────────────────────────────────

# ── LLM BACKEND SELECTOR ─────────────────────────────────────────────────────
# Set LLM_BACKEND to one of:
#   "ollama"  — free, local, no quota  (recommended — run: ollama pull llama3.2:3b)
#   "gemini"  — Google Gemini API      (requires API key + billing)
LLM_BACKEND   = "ollama"   # "ollama" or "gemini"
OLLAMA_MODEL  = "llama3.2:3b"     # or "mistral" if you pulled that instead
OLLAMA_URL    = "http://localhost:11434/api/generate"

SCRIPT_PROMPT = """You are a viral short-form video scriptwriter.

Topic: "{topic}"

Create content for a 30-40 second social media reel.

Return ONLY a raw JSON object (no markdown, no code fences) with exactly this shape:
{{
  "script": "120-140 word voiceover. Punchy, emotional, hook in first 5 words.",
  "image_prompts": ["prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5", "prompt 6"]
}}

Rules for image_prompts:
- Exactly {num_images} prompts
- Each is a Stable Diffusion prompt: photorealistic, 4k, cinematic, dramatic lighting, vertical portrait
- No text, logos, or watermarks
- Each prompt must be visually distinct
Return ONLY the JSON. No explanation, no markdown fences."""


def _parse_json(raw: str, num_images: int) -> dict:
    """Strip markdown fences and parse JSON, with basic validation."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    data = json.loads(raw)
    prompts = data["image_prompts"][:num_images]
    while len(prompts) < num_images:
        prompts.append(prompts[-1])
    data["image_prompts"] = prompts
    return data


def _generate_with_ollama(topic: str, num_images: int) -> dict:
    import urllib.request
    prompt = SCRIPT_PROMPT.format(topic=topic, num_images=num_images)
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",          # forces JSON output mode
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return _parse_json(result["response"], num_images)


def _generate_with_gemini(topic: str, num_images: int) -> dict:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model  = genai.GenerativeModel("gemini-2.0-flash")   # more generous free quota
    prompt = SCRIPT_PROMPT.format(topic=topic, num_images=num_images)
    response = model.generate_content(prompt)
    return _parse_json(response.text, num_images)


def generate_script(topic: str, num_images: int) -> dict:
    banner(f"   STEP 1 / 4 — Generating Script  [{topic}]  backend: {LLM_BACKEND}")
    t0 = time.time()

    if LLM_BACKEND == "ollama":
        data = _generate_with_ollama(topic, num_images)
    elif LLM_BACKEND == "gemini":
        data = _generate_with_gemini(topic, num_images)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r}. Use 'ollama' or 'gemini'.")

    print(f"     Script  ({len(data['script'].split())} words)  [{time.time()-t0:.1f}s]")
    print(f"\n  Script preview:\n  {data['script'][:120]}…\n")

    # Save script for reference
    (OUTPUT_DIR / "script.txt").write_text(data["script"])
    return data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — VOICEOVER  (Coqui TTS)
# ─────────────────────────────────────────────────────────────────────────────

def generate_voiceover(script: str) -> str:
    banner(" STEP 2 / 4 — Generating Voiceover  (edge-tts)")

    import asyncio
    import edge_tts

    out_path = str(AUDIO_DIR / "voiceover.mp3")

    # Voice options — change VOICE to any from: edge-tts --list-voices
    # Good English voices:
    #   "en-US-GuyNeural"       — male, neutral
    #   "en-US-JennyNeural"     — female, friendly
    #   "en-GB-RyanNeural"      — male, British
    #   "en-IN-NeerjaNeural"    — female, Indian English
    VOICE = "en-US-GuyNeural"
    RATE  = "+5%"   # speed: -20% slower, +20% faster

    async def _synthesize():
        communicate = edge_tts.Communicate(script, VOICE, rate=RATE)
        await communicate.save(out_path)

    asyncio.run(_synthesize())

    duration = get_audio_duration(out_path)
    print(f"     Voiceover saved → {out_path}  ({duration:.1f}s)  voice: {VOICE}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — AI VISUALS  (Stable Diffusion 1.5 on CPU)
# ─────────────────────────────────────────────────────────────────────────────

def generate_images(prompts: list, steps: int) -> list:
    banner(f"    STEP 3 / 4 — Generating {len(prompts)} Images  (SD 1.5 CPU)")

    print("      CPU inference: expect ~3-6 min per image. Total may take 20-40 min.")
    print(f"      Using {steps} inference steps  (lower = faster, higher = better)\n")

    from diffusers import StableDiffusionPipeline
    import torch
    from PIL import Image

    print("  Loading model weights (~4 GB, cached after first run)…")
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float32,   # float32 required for CPU
        safety_checker=None,          # skip for speed; re-enable if needed
    )
    pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()   # reduces peak RAM usage

    NEGATIVE = (
        "blurry, low quality, distorted, ugly, watermark, text, logo, "
        "duplicate, morbid, deformed, bad anatomy"
    )

    image_paths = []
    total = len(prompts)

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{total}]  Generating image…")
        print(f"  Prompt: {prompt[:90]}…")
        t0 = time.time()

        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            width=512,
            height=512,
            num_inference_steps=steps,
            guidance_scale=7.5,
        )

        img = result.images[0]
        # Resize to 1080×1920 (9:16 portrait) for reels
        img = img.resize((1080, 1920), Image.LANCZOS)

        path = str(IMAGES_DIR / f"frame_{i:02d}.png")
        img.save(path)
        image_paths.append(path)

        elapsed = time.time() - t0
        remaining = elapsed * (total - i)
        print(f"     Saved {path}  [{elapsed:.0f}s elapsed, ~{remaining/60:.1f} min remaining]\n")

    return image_paths


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ASSEMBLE REEL  (FFmpeg)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_reel(image_paths: list, audio_path: str, topic: str) -> str:
    banner("    STEP 4 / 4 — Assembling Reel  (FFmpeg)")

    safe_name   = "".join(c if c.isalnum() else "_" for c in topic)[:40]
    output_path = str(OUTPUT_DIR / f"{safe_name}.mp4")

    audio_dur     = get_audio_duration(audio_path)
    time_per_img  = audio_dur / len(image_paths)

    print(f"  Audio duration : {audio_dur:.1f}s")
    print(f"  Images         : {len(image_paths)}")
    print(f"  Time per image : {time_per_img:.2f}s")

    # ── Write concat manifest ────────────────────────────────────────────────
    concat_file = OUTPUT_DIR / "concat.txt"
    with open(concat_file, "w") as f:
        for path in image_paths:
            abs_path = os.path.abspath(path)
            f.write(f"file '{abs_path}'\n")
            f.write(f"duration {time_per_img}\n")
        # FFmpeg concat demuxer needs the last file listed twice (no duration)
        f.write(f"file '{os.path.abspath(image_paths[-1])}'\n")

    # ── FFmpeg command ───────────────────────────────────────────────────────
    #
    #  -f concat         : use concat demuxer (respects 'duration' in manifest)
    #  zoompan           : gentle 1-→1.05 zoom-in per image (Ken-Burns lite)
    #  xfade             : 0.5s crossfade between images
    #  libx264 / aac     : standard web-compatible codecs
    #  -shortest         : stop when audio ends
    #
    # Note: zoompan + concat can be tricky; we use a simple per-frame zoom.
    # ────────────────────────────────────────────────────────────────────────

    fps = 25
    zoom_frames = int(time_per_img * fps)   # frames per image

    vf = (
        f"scale=1080:1920:force_original_aspect_ratio=increase,"
        f"crop=1080:1920,"
        f"zoompan="
        f"z='min(zoom+0.0015,1.05)':"
        f"x='iw/2-(iw/zoom/2)':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={zoom_frames}:"
        f"s=1080x1920:"
        f"fps={fps}"
    )

    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", audio_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path,
    ]

    print("\n  Running FFmpeg…")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n     FFmpeg failed:\n{result.stderr[-2000:]}")
        # Fallback: simple slideshow without zoom
        print("\n      Retrying without zoom effect…")
        cmd_simple = [
            FFMPEG_BIN, "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", audio_path,
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path,
        ]
        result2 = subprocess.run(cmd_simple, capture_output=True, text=True)
        if result2.returncode != 0:
            raise RuntimeError(f"FFmpeg failed:\n{result2.stderr[-1000:]}")

    print(f"     Reel saved → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(topic: str, num_images: int = 6, steps: int = 20):
    check_ffmpeg()

    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)

    t_start = time.time()

    print("\n" + "═"*56)
    print("      AI REELS GENERATOR")
    print(f"  Topic  : {topic}")
    print(f"  Images : {num_images}   Steps: {steps}")
    print("═"*56)

    # ── Step 1: Script ────────────────────────────────────────────────
    content = generate_script(topic, num_images)
    script  = content["script"]
    prompts = content["image_prompts"]

    # ── Step 2: Voiceover ─────────────────────────────────────────────
    audio_path = generate_voiceover(script)

    # ── Step 3: Images ────────────────────────────────────────────────
    image_paths = generate_images(prompts, steps)

    # ── Step 4: Reel ──────────────────────────────────────────────────
    reel_path = assemble_reel(image_paths, audio_path, topic)

    total = time.time() - t_start
    print("\n" + "═"*56)
    print(f"      DONE in {total/60:.1f} min")
    print(f"  Reel  : {reel_path}")
    print(f"  Script: {OUTPUT_DIR / 'script.txt'}")
    print("═"*56 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a 30-40s AI reel from a text topic."
    )
    parser.add_argument("topic", help='Topic for the reel, e.g. "The Great Wall of China"')
    parser.add_argument(
        "--images", type=int, default=6,
        help="Number of images to generate (default: 6, min: 3, max: 10)"
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="SD inference steps (default: 20; use 15 for speed, 30 for quality)"
    )
    args = parser.parse_args()

    num_images = max(3, min(args.images, 10))
    steps      = max(10, min(args.steps,  50))

    run(args.topic, num_images, steps)