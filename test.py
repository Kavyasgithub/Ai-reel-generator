import pyttsx3
import os

import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyBhPg0WTncWMf48jCKARQQhG68pWBIcVrw")

# Generate script using Gemini
def generate_voiceover_script():
    model = genai.GenerativeModel("gemini-2.5-flash")
    input_data = input("Enter the topic for the voiceover script: ")
    prompt = f"Write a professional voiceover script for a 30-40 second video for {input_data}. Make it engaging and concise. Ensure it is clean and suitable for text to speech conversion with no special characters etc.. "
    
    response = model.generate_content(prompt)
    return response.text

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

# Main execution
if __name__ == "__main__":
    script = generate_voiceover_script()
    print("Generated Script:\n", script)
    text_to_speech(script)