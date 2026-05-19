from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time, hmac, hashlib, base64, requests
import os

# ACRCloud credentials
HOST = 'https://identify-ap-southeast-1.acrcloud.com/v1/identify'
ACCESS_KEY = 'cd14d2757fb14a000cd6016a9ad854dc'
ACCESS_SECRET = '0RuaQqJB6WgmZjzvMSra8CpqIoeKQ6KduQgEQ3Rq'

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify_song():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Send to ACRCloud
    result = recognize(path)
    try:
        music_info = result['metadata']['music'][0]
        return jsonify({
            'title': music_info.get('title', 'Unknown'),
            'artists': [a['name'] for a in music_info.get('artists', [])],
            'album': music_info.get('album', {}).get('name', 'Unknown'),
            'release_date': music_info.get('release_date', 'Unknown')
        })
    except:
        return jsonify({'error': 'Could not recognize song'}), 400

def recognize(audio_path):
    http_method = "POST"
    http_uri = "/v1/identify"
    data_type = "audio"
    signature_version = "1"
    timestamp = str(int(time.time()))

    with open(audio_path, 'rb') as f:
        sample_bytes = f.read()

    string_to_sign = '\n'.join([http_method, http_uri, ACCESS_KEY, data_type, signature_version, timestamp])
    sign = base64.b64encode(hmac.new(ACCESS_SECRET.encode(), string_to_sign.encode(), hashlib.sha1).digest()).decode()

    files = {'sample': sample_bytes}
    data = {
        'access_key': ACCESS_KEY,
        'sample_bytes': len(sample_bytes),
        'timestamp': timestamp,
        'signature': sign,
        'data_type': data_type,
        'signature_version': signature_version,
    }

    response = requests.post(HOST, files=files, data=data)
    return response.json()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True) 