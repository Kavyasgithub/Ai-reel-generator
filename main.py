from moviepy import TextClip
import requests
import edge_tts
import asyncio
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
    txt_clip = TextClip(
        text,
        fontsize=50,
        color='white',
        size=(1000, None),
        method='caption'
    )

    txt_clip = txt_clip.set_position(
        ("center", "bottom")
    ).set_duration(video.duration)

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