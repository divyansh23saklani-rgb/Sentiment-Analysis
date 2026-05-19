import time
import hmac
import hashlib
import base64
import requests
import json
import wave
import pyaudio

# 🔑 ACRCloud config
HOST = 'https://identify-ap-southeast-1.acrcloud.com/v1/identify'
ACCESS_KEY = 'cd14d2757fb14a000cd6016a9ad854dc'
ACCESS_SECRET = '0RuaQqJB6WgmZjzvMSra8CpqIoeKQ6KduQgEQ3Rq'

def record_audio(file_name="recorded.wav", duration=10):
    RATE = 44100
    CHANNELS = 1
    FORMAT = pyaudio.paInt16
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print(f"🎙️ Recording {duration}s...")
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * duration))]
    print("✅ Done.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    return file_name

def recognize(audio_path):
    http_method = "POST"
    http_uri = "/v1/identify"
    data_type = "audio"
    signature_version = "1"
    timestamp = str(int(time.time()))

    with open(audio_path, 'rb') as f:
        sample_bytes = f.read()

    string_to_sign = '\n'.join([http_method, http_uri, ACCESS_KEY, data_type, signature_version, timestamp])
    sign = base64.b64encode(hmac.new(ACCESS_SECRET.encode('ascii'), string_to_sign.encode('ascii'), digestmod=hashlib.sha1).digest()).decode('ascii')

    files = {'sample': sample_bytes}
    data = {
        'access_key': ACCESS_KEY,
        'sample_bytes': len(sample_bytes),
        'timestamp': timestamp,
        'signature': sign,
        'data_type': data_type,
        'signature_version': signature_version,
    }

    print("🔍 Sending to ACRCloud...")
    response = requests.post(HOST, files=files, data=data)
    try:
        return response.json()
    except Exception as e:
        print("❌ Failed to parse JSON response. Raw response:")
        print(response.status_code, response.text)
        raise e

if __name__ == "__main__":
    file = record_audio(duration=12)
    result = recognize(file)
    
    
    
    # print("🎵 Result:", json.dumps(result, indent=2))


    try:
        music_info = result['metadata']['music'][0]
        title = music_info.get('title', 'Unknown Title')
        artists = ', '.join(artist['name'] for artist in music_info.get('artists', []))
        album = music_info.get('album', {}).get('name', 'Unknown Album')
        release_date = music_info.get('release_date', 'Unknown Release Date')
        
        print("🎶 Song Identified!")
        print(f"🎵 Title       : {title}")
        print(f"🎤 Artist(s)   : {artists}")
        print(f"💿 Album       : {album}")
        print(f"📅 Release Date: {release_date}")
    except KeyError:
        print("❌ Could not identify the song or unexpected response format.")
