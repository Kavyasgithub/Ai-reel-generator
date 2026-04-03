import pyttsx3
import os
import re
from typing import List
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
try:
    from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, VideoFileClip, concatenate_videoclips
except ImportError:
    from moviepy import AudioFileClip, CompositeVideoClip, ImageClip, VideoFileClip, concatenate_videoclips

import google.generativeai as genai


@dataclass
class SubtitleStyle:
    font_size: int = 62
    color: tuple = (255, 255, 255, 255)
    box_fill: tuple = (0, 0, 0, 145)
    box_outline: tuple = (255, 255, 255, 120)
    position: str = "bottom"  # top, center, bottom
    animation: str = "slide_up"  # none, slide_up, pop
    padding_x: int = 44
    padding_y: int = 34
    line_gap: int = 14


SUBTITLE_PRESETS = {
    "clean": SubtitleStyle(),
    "neon": SubtitleStyle(
        font_size=64,
        color=(230, 255, 250, 255),
        box_fill=(8, 24, 40, 165),
        box_outline=(95, 240, 220, 180),
        position="bottom",
        animation="slide_up",
    ),
    "cinema": SubtitleStyle(
        font_size=58,
        color=(255, 241, 212, 255),
        box_fill=(20, 12, 8, 170),
        box_outline=(255, 196, 120, 170),
        position="center",
        animation="pop",
    ),
}


KEYWORD_GROUPS = {
    "docker": {"docker", "container", "image", "kubernetes", "compose"},
    "cloud": {"cloud", "server", "infrastructure", "platform", "scale"},
    "deploy": {"deploy", "delivery", "pipeline", "release", "launch"},
    "ai": {"ai", "model", "machine", "learning", "neural", "llm"},
    "security": {"security", "secure", "threat", "risk", "privacy"},
}


# Optional direct URLs for auto-download (kept empty by default).
# You can add free public MP4 links per keyword if you have preferred sources.
REMOTE_STOCK_CLIP_URLS = {
    "docker": [],
    "cloud": [],
    "deploy": [],
    "ai": [],
    "security": [],
    "general": [],
}

# Configure Gemini API
genai.configure(api_key="")

# Generate script using Gemini
def generate_voiceover_script():
    model = genai.GenerativeModel("gemini-2.5-flash")
    input_data = input("Enter the topic for the voiceover script: ")
    prompt = (
        f"Write a professional voiceover script for a 30-40 second video for {input_data}. "
        "Return only plain narration text as 1-2 short paragraphs. "
        "Do not include headings, timestamps, markdown, stage directions, symbols, or labels like Voiceover."
    )
    
    response = model.generate_content(prompt)
    return response.text


def filter_voiceover_text(raw_text: str) -> str:
    """Keep only narration content suitable for text-to-speech."""
    if not raw_text:
        return ""

    lines = raw_text.replace("\r\n", "\n").split("\n")
    cleaned_lines = []

    skip_patterns = [
        r"^here is\b",
        r"^video\s*title\b",
        r"^\(.*\)$",                 # Full stage-direction lines
        r"^\[.*\]$",                 # Bracketed direction lines
        r"^music\b",
        r"^visuals?\b",
        r"^---+$",
        r"^\*+$",
        r"^audio\s+saved\s+to\b",
    ]

    for line in lines:
        text = line.strip()
        if not text:
            continue

        # Remove markdown emphasis and bullets.
        text = re.sub(r"[*_`#]+", "", text)
        text = re.sub(r"^[-•]\s*", "", text)
        text = text.strip()

        # Keep content after a voiceover label, if present.
        lower = text.lower()
        if lower.startswith("voiceover:"):
            text = text.split(":", 1)[1].strip()
            lower = text.lower()
        else:
            # Remove leading timing cues, then re-check for stage-direction keywords.
            no_timing = re.sub(
                r"^[\(\[][^\)\]]*\bseconds?\b[^\)\]]*[\)\]]\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
            lower_no_timing = no_timing.lower()

            if any(re.match(pattern, lower_no_timing) for pattern in skip_patterns):
                continue

            # Drop non-voiceover stage direction lines.
            if re.search(r"\b(music|visuals?)\b", lower_no_timing):
                continue

            text = no_timing

        # Drop obvious timing cues like "(0-3 seconds)" still left inline.
        text = re.sub(r"\([^)]*\bseconds?\b[^)]*\)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[[^\]]*\bseconds?\b[^\]]*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip(" -")

        if text:
            cleaned_lines.append(text)

    # Join as natural narration for TTS.
    return " ".join(cleaned_lines)

# Convert text to speech
def text_to_speech(text, output_file="output.mp3"):
    engine = pyttsx3.init()
    
    # Configure voice properties
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 0.9)  # Volume (0-1)
    
    # Save to file
    engine.save_to_file(text, output_file)
    engine.runAndWait()
    
    print(f"Audio saved to {output_file}")


def get_subtitle_style() -> SubtitleStyle:
    """Read subtitle style from env vars or preset selection."""
    preset_name = os.getenv("SUBTITLE_PRESET", "neon").strip().lower()
    style = SUBTITLE_PRESETS.get(preset_name, SubtitleStyle())

    style = SubtitleStyle(**style.__dict__)
    if os.getenv("SUBTITLE_POSITION"):
        style.position = os.getenv("SUBTITLE_POSITION", "bottom").strip().lower()
    if os.getenv("SUBTITLE_ANIMATION"):
        style.animation = os.getenv("SUBTITLE_ANIMATION", "slide_up").strip().lower()

    return style


def split_into_scenes(script_text: str, max_words_per_scene: int = 14) -> List[str]:
    """Split narration into short scene lines for visual cards."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script_text) if s.strip()]
    scenes: List[str] = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words_per_scene:
            scenes.append(sentence)
            continue

        for i in range(0, len(words), max_words_per_scene):
            chunk = " ".join(words[i:i + max_words_per_scene]).strip()
            if chunk:
                scenes.append(chunk)

    return scenes or [script_text]


def make_gradient_background(width: int, height: int, color_top: tuple, color_bottom: tuple) -> np.ndarray:
    """Create a soft vertical gradient background frame."""
    gradient = np.linspace(0, 1, height).reshape(height, 1, 1)
    top = np.array(color_top, dtype=np.float32).reshape(1, 1, 3)
    bottom = np.array(color_bottom, dtype=np.float32).reshape(1, 1, 3)
    frame = (top * (1 - gradient) + bottom * gradient).astype(np.uint8)
    return np.repeat(frame, width, axis=1)


def infer_scene_keyword(scene_text: str) -> str:
    text = scene_text.lower()
    for key, words in KEYWORD_GROUPS.items():
        if any(word in text for word in words):
            return key
    return "general"


def download_ai_image(scene_text: str, keyword: str, output_image: Path, seed: int) -> bool:
    """Generate a free AI image using Pollinations API and save it locally."""
    prompt = (
        f"Cinematic vertical 9:16 visual for: {scene_text}. "
        f"Theme: {keyword}. High detail, dramatic lighting, no text, no watermark."
    )
    url = (
        f"https://image.pollinations.ai/prompt/{quote(prompt)}"
        f"?width=1080&height=1920&seed={seed}&nologo=true"
    )

    try:
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_image.write_bytes(response.content)
        return True
    except Exception:
        return False


def find_local_stock_clip(keyword: str) -> Path | None:
    """Find user-provided stock clips by keyword from local folders."""
    search_paths = [
        Path("stock") / keyword,
        Path("stock") / "general",
        Path("output") / "stock" / keyword,
        Path("output") / "stock" / "general",
    ]

    candidates: List[Path] = []
    for folder in search_paths:
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.mp4")))

    return candidates[0] if candidates else None


def maybe_download_stock_clip(keyword: str, output_dir: Path) -> Path | None:
    """Optionally download a stock clip from configured free public URLs."""
    urls = REMOTE_STOCK_CLIP_URLS.get(keyword, []) or REMOTE_STOCK_CLIP_URLS.get("general", [])
    if not urls:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{keyword}.mp4"
    if target.exists():
        return target

    for url in urls:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            target.write_bytes(response.content)
            return target
        except Exception:
            continue

    return None


def wrap_text_for_card(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    """Wrap text to fit within the visual card width."""
    words = text.split()
    if not words:
        return []

    lines: List[str] = []
    current = words[0]

    for word in words[1:]:
        test = f"{current} {word}"
        left, top, right, bottom = draw.textbbox((0, 0), test, font=font)
        if (right - left) <= max_width:
            current = test
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def render_text_card(scene_text: str, width: int, height: int, style: SubtitleStyle) -> np.ndarray:
    """Render a transparent text card for a scene."""
    card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)

    try:
        font = ImageFont.truetype("arial.ttf", style.font_size)
    except OSError:
        font = ImageFont.load_default()

    max_text_width = int(width * 0.78)
    lines = wrap_text_for_card(draw, scene_text, font, max_text_width)
    if not lines:
        return np.array(card)

    line_heights = []
    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bottom - top)

    total_text_height = sum(line_heights) + (len(lines) - 1) * style.line_gap
    box_padding_x = style.padding_x
    box_padding_y = style.padding_y
    box_width = max_text_width + (box_padding_x * 2)
    box_height = total_text_height + (box_padding_y * 2)
    box_x = (width - box_width) // 2

    if style.position == "top":
        box_y = int(height * 0.20) - (box_height // 2)
    elif style.position == "center":
        box_y = int(height * 0.50) - (box_height // 2)
    else:
        box_y = int(height * 0.76) - (box_height // 2)

    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_width, box_y + box_height],
        radius=26,
        fill=style.box_fill,
        outline=style.box_outline,
        width=2,
    )

    y = box_y + box_padding_y
    for i, line in enumerate(lines):
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        text_width = right - left
        x = (width - text_width) // 2
        draw.text((x, y), line, font=font, fill=style.color)
        y += line_heights[i] + style.line_gap

    return np.array(card)


def clip_set_duration(clip, duration: float):
    """Compatibility wrapper for MoviePy v1/v2 duration API."""
    if hasattr(clip, "set_duration"):
        return clip.set_duration(duration)
    return clip.with_duration(duration)


def clip_set_audio(clip, audio_clip):
    """Compatibility wrapper for MoviePy v1/v2 audio API."""
    if hasattr(clip, "set_audio"):
        return clip.set_audio(audio_clip)
    return clip.with_audio(audio_clip)


def clip_set_position(clip, position):
    if hasattr(clip, "set_position"):
        return clip.set_position(position)
    return clip.with_position(position)


def clip_resize(clip, width=None, height=None, factor=None):
    if factor is not None:
        if hasattr(clip, "resize"):
            return clip.resize(factor)
        return clip.resized(factor)

    kwargs = {}
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height

    if hasattr(clip, "resize"):
        return clip.resize(**kwargs)
    return clip.resized(**kwargs)


def clip_crop(clip, x_center=None, y_center=None, width=None, height=None):
    kwargs = {}
    if x_center is not None:
        kwargs["x_center"] = x_center
    if y_center is not None:
        kwargs["y_center"] = y_center
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height

    if hasattr(clip, "crop"):
        return clip.crop(**kwargs)
    return clip.cropped(**kwargs)


def clip_subclip(clip, start_t: float, end_t: float):
    """Compatibility wrapper for MoviePy v1/v2 subclip API."""
    if hasattr(clip, "subclip"):
        return clip.subclip(start_t, end_t)
    return clip.subclipped(start_t, end_t)


def fit_clip_to_vertical(clip, width: int, height: int):
    """Resize and center-crop any clip to 1080x1920 vertical format."""
    clip = clip_resize(clip, height=height)
    if clip.w < width:
        clip = clip_resize(clip, width=width)

    return clip_crop(
        clip,
        x_center=clip.w / 2,
        y_center=clip.h / 2,
        width=width,
        height=height,
    )


def build_background_clip(scene_text: str, keyword: str, duration: float, width: int, height: int, image_path: Path):
    """Create scene background from stock motion clip or AI image fallback."""
    local_stock = find_local_stock_clip(keyword)
    if not local_stock:
        local_stock = maybe_download_stock_clip(keyword, Path("output") / "stock_cache")

    if local_stock and local_stock.exists():
        try:
            stock_source = VideoFileClip(str(local_stock))
            stock = fit_clip_to_vertical(stock_source, width, height)
            if stock.duration >= duration:
                return clip_subclip(stock, 0, duration)
            return clip_set_duration(stock, duration)
        except Exception:
            pass

    if not image_path.exists():
        ok = download_ai_image(scene_text, keyword, image_path, seed=abs(hash(scene_text)) % 100000)
        if not ok:
            return None

    image_clip = clip_set_duration(ImageClip(str(image_path)), duration)
    return fit_clip_to_vertical(image_clip, width, height)


def apply_subtitle_animation(sub_clip, style: SubtitleStyle, height: int):
    """Apply simple motion animation to subtitles."""
    if style.animation == "none":
        pos_y = "bottom"
    elif style.position == "top":
        pos_y = "top"
    elif style.position == "center":
        pos_y = "center"
    else:
        pos_y = "bottom"

    return clip_set_position(sub_clip, ("center", pos_y))


def create_synced_visual_video(script_text: str, audio_file: str, output_video: str) -> None:
    """Generate a free, local vertical visual reel synchronized with narration audio."""
    if not Path(audio_file).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    scenes = split_into_scenes(script_text)
    width, height = 1080, 1920
    style = get_subtitle_style()
    scene_image_dir = Path("output") / "scene_images"
    scene_image_dir.mkdir(parents=True, exist_ok=True)

    with AudioFileClip(audio_file) as audio:
        total_duration = max(audio.duration, 1.0)

        weights = [max(1, len(scene.split())) for scene in scenes]
        weight_sum = float(sum(weights))
        durations = [total_duration * (w / weight_sum) for w in weights]

        # Ensure exact sync by correcting floating-point remainder in the last scene.
        durations[-1] += total_duration - sum(durations)

        palettes = [
            ((18, 28, 44), (40, 98, 148)),
            ((23, 52, 36), (64, 136, 88)),
            ((55, 32, 24), (164, 98, 66)),
            ((38, 24, 62), (98, 60, 156)),
            ((30, 30, 30), (98, 98, 98)),
        ]

        clips = []
        for i, scene in enumerate(scenes):
            keyword = infer_scene_keyword(scene)
            image_path = scene_image_dir / f"scene_{i+1:02d}_{keyword}.jpg"

            bg_clip = build_background_clip(scene, keyword, durations[i], width, height, image_path)
            if bg_clip is None:
                top, bottom = palettes[i % len(palettes)]
                bg_frame = make_gradient_background(width, height, top, bottom)
                bg_clip = clip_set_duration(ImageClip(bg_frame), durations[i])

            card_img = render_text_card(scene, width, height, style)
            card_path = scene_image_dir / f"subtitle_{i+1:02d}.png"
            card_pil = Image.fromarray(card_img)
            card_pil.save(str(card_path), "PNG")
            card_clip = clip_set_duration(ImageClip(str(card_path)), durations[i])
            card_clip = apply_subtitle_animation(card_clip, style, height)

            scene_clip = clip_set_duration(
                CompositeVideoClip([bg_clip, card_clip], size=(width, height)),
                durations[i],
            )
            clips.append(scene_clip)

        final_video = clip_set_audio(
            concatenate_videoclips(clips, method="compose"),
            clip_subclip(audio, 0, total_duration),
        )
        final_video.write_videofile(output_video, fps=24, codec="libx264", audio_codec="aac")


# Main execution
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("stock", exist_ok=True)

    script = generate_voiceover_script()
    filtered_script = filter_voiceover_text(script)
    output_audio_path = "output/output.mp3"
    output_video_path = "output/reel.mp4"

    print("Generated Script:\n", script)
    print("\nFiltered Script for TTS:\n", filtered_script)

    if filtered_script:
        text_to_speech(filtered_script, output_audio_path)
        print("Creating synced visuals...")
        create_synced_visual_video(filtered_script, output_audio_path, output_video_path)
        print(f"Reel saved to {output_video_path}")
        print("Tip: Add keyword MP4 clips inside stock/<keyword>/ for motion backgrounds.")
        print("Supported keywords: docker, cloud, deploy, ai, security, general")
        print("Subtitle options via env vars: SUBTITLE_PRESET, SUBTITLE_POSITION, SUBTITLE_ANIMATION")
    else:
        print("No valid voiceover text found after filtering.")
