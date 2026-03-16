import requests
import edge_tts
import asyncio
import numpy as np
from moviepy.editor import *

#Script (Ollama)
def generate_script(topic):
    prompt = f"""
    Create a 30-second Instagram reel script about {topic}.
    Start with a strong hook.
    Use short punchy lines.
    End with a call to action.
    Keep it under 120 words.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )

        response.raise_for_status()
        return response.json()["response"]

    except Exception as e:
        print("Error generating script:", e)
        return "Stay consistent. Stay focused. Success will follow."

# Generate Voice (Edge TTS)
async def generate_voice(text):
    try:
        communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
        await communicate.save("voice.mp3")
    except Exception as e:
        print("Error generating voice:", e)


#  Prepare Background (9:16)
def prepare_background():
    video = VideoFileClip("background.mp4")

    # Ensure video is long enough
    duration = min(30, video.duration)

    # Resize to vertical
    video = video.resize(height=1920)

    if video.w < 1080:
        video = video.resize(width=1080)

    video = video.crop(
        width=1080,
        height=1920,
        x_center=video.w / 2,
        y_center=video.h / 2
    )

    return video.subclip(0, duration)

# Add Captions
def add_captions(video, text):
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a text image using PIL
    width, height = 1000, 500
    img = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Wrap text
    lines = []
    words = text.split()
    current_line = []
    for word in words:
        current_line.append(word)
        test_line = " ".join(current_line)
        if draw.textbbox((0, 0), test_line, font=font)[2] > width - 20:
            if len(current_line) > 1:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(test_line)
                current_line = []
    if current_line:
        lines.append(" ".join(current_line))
    
    # Draw text
    y_offset = 20
    for line in lines:
        draw.text((10, y_offset), line, fill="white", font=font)
        y_offset += 50
    
    img = img.crop(img.getbbox() or (0, 0, width, height))
    txt_clip = ImageClip(np.array(img)).set_duration(video.duration)
    txt_clip = txt_clip.set_position(("center", "bottom"))

    return CompositeVideoClip([video, txt_clip])

#  Create Reel Pipeline
def create_reel(topic):
    print("Generating script...")
    script = generate_script(topic)
    print("\nGenerated Script:\n")
    print(script)

    print("\nGenerating voice...")
    asyncio.run(generate_voice(script))

    print("Preparing background...")
    video = prepare_background()

    print("Adding audio...")
    audio = AudioFileClip("voice.mp3")
    video = video.set_audio(audio)

    print("Adding captions...")
    final_video = add_captions(video, script)

    print("Exporting reel...")
    final_video.write_videofile(
        "output/reel.mp4",
        fps=24,
        codec="libx264",
        audio_codec="aac"
    )

    print("\n✅ Done! Check output folder.")


#  Main Entry Point
if __name__ == "__main__":
    topic = input("Enter topic: ")
    create_reel(topic)