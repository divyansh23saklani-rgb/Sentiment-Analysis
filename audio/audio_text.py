# audio_to_text.py

from faster_whisper import WhisperModel

# Settings
audio_file = "recorded_audio.wav"
model_size = "small"  # You can also use "base", "medium", "large"

# Load the model
print("Loading the Whisper model...")
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Transcribe
print("Transcribing the audio...")
segments, info = model.transcribe(audio_file, beam_size=5)

# Write to text file
with open("transcription.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        f.write(segment.text + " ")

print("Transcription complete! Saved as 'transcription.txt'")
