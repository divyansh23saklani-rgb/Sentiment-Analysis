import sounddevice as sd
import scipy.io.wavfile as wav
from TTS.api import TTS
import os
import torch

def record_audio(filename="my_voice.wav", duration=5, fs=16000):
    try:
        print("🎙️ Recording... Speak now!")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wav.write(filename, fs, audio)
        print(f"✅ Recording saved as {filename}")
    except Exception as e:
        print(f"❌ Failed to record audio: {e}")
        raise

def load_tts_model():
    use_gpu = torch.cuda.is_available()
    print(f"🔁 Loading TTS model (XTTS-v2)... [GPU: {use_gpu}]")
    tts = TTS(model_name="coqui/xtts-v2", progress_bar=True, gpu=use_gpu)
    print("✅ TTS model loaded.")
    return tts

def synthesize_speech(tts_model, text, reference_audio="my_voice.wav", output_file="output.wav"):
    if not os.path.exists(reference_audio):
        raise FileNotFoundError(f"❌ {reference_audio} not found.")
    
    print("🗣️ Synthesizing text in your voice...")
    tts_model.tts_to_file(
        text=text,
        speaker_wav=reference_audio,
        language="en",
        file_path=output_file
    )
    print(f"✅ Synthesized voice saved to: {output_file}")

def main():
    try:
        record_audio()
        tts_model = load_tts_model()
        text = input("📝 Enter the text you want to synthesize: ")
        synthesize_speech(tts_model, text)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
