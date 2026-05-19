from pydub import AudioSegment
import speech_recognition as sr
import os
import math

# Split audio into 15-second chunks
def split_audio_chunks(audio_path, chunk_length_ms=15000):
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    total_length = len(audio)
    for i in range(0, total_length, chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_filename = f"chunk_{i//chunk_length_ms}.wav"
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)
    return chunks

# Transcribe each chunk
def transcribe_chunks(chunk_files):
    recognizer = sr.Recognizer()
    transcript = ""
    for idx, chunk_file in enumerate(chunk_files):
        with sr.AudioFile(chunk_file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data, language="en-IN")
                transcript += f"{text} "
                print(f"✅ Chunk {idx + 1} transcribed.")
            except sr.UnknownValueError:
                print(f"⚠️ Could not understand chunk {idx + 1}")
            except sr.RequestError as e:
                print(f"❌ API error on chunk {idx + 1}: {e}")
        os.remove(chunk_file)  # Clean up
    return transcript.strip()

def transcribe_long_audio(file_path):
    output_file = "long_transcription.txt"

    # Convert .ogg to .wav if needed
    if file_path.lower().endswith(".ogg"):
        print("🎧 Converting OGG to WAV...")
        audio = AudioSegment.from_file(file_path, format="ogg")
        file_path = file_path.replace(".ogg", ".wav")
        audio.export(file_path, format="wav")

    print("🔁 Splitting audio into chunks...")
    chunk_files = split_audio_chunks(file_path)
    
    print("🧠 Transcribing chunks...")
    final_text = transcribe_chunks(chunk_files)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_text + "\n")

    print("✅ Transcription complete.")
    print(f"📄 Saved to: {output_file}")

# Run it
if __name__ == "__main__":
    audio_path = "./audio/sample2.ogg"  # Replace this
    transcribe_long_audio(audio_path)
