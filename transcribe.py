import sys
import os
from transformers import pipeline
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = mp3_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(file_path):
    print("Loading Whisper model...")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

    print("Converting MP3 to WAV...")
    wav_path = convert_mp3_to_wav(file_path)

    print("Transcribing...")
    result = asr(wav_path)

    transcript = result["text"]
    print("\n🎧 Transcription:\n", transcript)

    os.makedirs("output", exist_ok=True)
    with open("output/transcript.txt", "w") as f:
        f.write(transcript)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <path_to_mp3_file>")
        sys.exit(1)

    mp3_file = sys.argv[1]
    if not os.path.exists(mp3_file):
        print(f"File not found: {mp3_file}")
        sys.exit(1)

    transcribe_audio(mp3_file)