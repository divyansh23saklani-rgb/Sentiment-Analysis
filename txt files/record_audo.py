# record_audio.py

import sounddevice as sd
from scipy.io.wavfile import write

# Settings
duration = 10 # 10 minutes = 600 seconds
samplerate = 16000  # 16 kHz is enough for speech
filename = "/audio/recorded_audio.wav"

print("Recording started for 10 minutes... Speak now!")

# Record audio
audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished

# Save the recording
write(filename, samplerate, audio_data)

print(f"Recording finished. Saved as {filename}")
