import requests
import json
import os
project_root = os.path.join(os.path.dirname(__file__),'..')
import sys
sys.path.append(project_root)
from config import XI_API_KEY

url = "https://api.elevenlabs.io/v1/voices"


headers = {
  "Accept": "application/json",
  "xi-api-key": XI_API_KEY
}

CHUNK_SIZE = 1024
DANIEL_VOICE_ID = "onwK4e9ZLuTAKqWW03F9"
sts_url = f"https://api.elevenlabs.io/v1/speech-to-speech/{DANIEL_VOICE_ID}/stream"
response = requests.get(url, headers=headers)
data = response.json()

AUDIO_FILE_PATH = "/project/audio/old/voice-hispanic-1.wav"
OUTPUT_PATH = "/project/audio/old/voice-hispanic-1-xilabs-daniel.wav"

data = {
    "model_id": "eleven_multilingual_sts_v2",
    "voice_settings": json.dumps({
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.0,
        "use_speaker_boost": True
    })
}
try:
    files = {
        "audio": open(AUDIO_FILE_PATH, "rb")
    }
except Exception as e:
    print(f"Exception: {e}")
    assert False

response = requests.post(sts_url, headers=headers, data=data, files=files, stream=True)

if response.ok:
    # Open the output file in write-binary mode
    with open(OUTPUT_PATH, "wb") as f:
        # Read the response in chunks and write to the file
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
    print("Audio stream saved successfully.")
else:
    print(response.text)