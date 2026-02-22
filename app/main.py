import os
import time
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from dotenv import load_dotenv

from google import genai
from google.genai import types

from langchain_core.messages import HumanMessage, AIMessage
from .graph import graph

# -------- ENV --------
load_dotenv()

# -------- Gemini Client --------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text = ""
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text += part.get("text", "")
        return text

    return str(content)

def speak_with_gemini(text: str):
    if not text or not text.strip():
        return

    print("🔊 Generating speech...")

    styled_text = f"""
You are a loving, sweet, caring girlfriend.

Speak softly and warmly, but with a slight playful anger.
Sound like you're a little upset in a cute way.
Your tone should feel emotionally connected and real.

Add small natural pauses.
Let your voice show a tiny bit of attitude,
like you're pretending to be annoyed because you care.

Do not sound robotic.
Do not sound truly angry — just playful and expressive.

Text to speak:
{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=styled_text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Despina"
                    )
                )
            ),
        )
    )

    audio_bytes = response.candidates[0].content.parts[0].inline_data.data

    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    sd.play(audio_float32, samplerate=24000)
    sd.wait()

def main():
    recognizer = sr.Recognizer()
    messages = []

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.pause_threshold = 2

            while True:
                print("\n🎙️ Speak something...")
                audio = recognizer.listen(source)

                try:
                    print("🧠 Processing (STT)...")
                    user_text = recognizer.recognize_google(audio)
                    print("👤 You said:", user_text)

                except sr.UnknownValueError:
                    print("❌ Could not understand audio")
                    continue

                except sr.RequestError as e:
                    print("❌ STT Error:", e)
                    continue

                messages.append(HumanMessage(content=user_text))
                final_ai_message = None

                for event in graph.stream(
                    {"messages": messages},
                    stream_mode="values"
                ):
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            final_ai_message = last_msg

                if final_ai_message:
                    ai_text = extract_text(final_ai_message.content)
                    # print("🤖 AI:", ai_text)

                    speak_with_gemini(ai_text)
                    messages.append(final_ai_message)

                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n👋 Exiting voice assistant safely...")


if __name__ == "__main__":
    main()

