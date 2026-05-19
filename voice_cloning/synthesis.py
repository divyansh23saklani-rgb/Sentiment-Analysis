from TTS.api import TTS

# Load the multilingual XTTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)

# Your custom voice reference audio
reference_audio_path = "my_voice.wav"  # Must be ~6 seconds of clear speech

# Input text to synthesize
text_input = "Hello! This is my cloned voice speaking in English."

# Output path
output_path = "cloned_output.wav"

# Synthesize speech with your voice
tts.tts_to_file(
    text=text_input,
    speaker_wav=reference_audio_path,
    language="en",  # or "hi" for Hindi, etc.
    file_path=output_path
)

print("✅ Speech synthesized successfully and saved to:", output_path)
