import os
import requests

# Directory containing sound files
ROOT_DIR = os.path.dirname(__file__)
directory = os.path.join(ROOT_DIR,"xilabs_task_3_wavtokenizer")
# API endpoint
api_url = "https://api.elevenlabs.io/v1/moderation/ai-speech-classification"

# Headers as observed in the request
headers = {
    "Host": "api.elevenlabs.io",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Referer": "https://elevenlabs.io/",
    "Origin": "https://elevenlabs.io",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Priority": "u=4",
}

# Function to upload a file and handle the response
def upload_file(file_path):
    with open(file_path, "rb") as audio_file:
        # Form-data payload
        files = {
            "file": (os.path.basename(file_path), audio_file, "audio/x-flac"),
        }
        response = requests.post(api_url, headers=headers, files=files)
        
        # Handle the response
        if response.status_code == 200:
            try:
                result = response.json()  # Parse JSON response
                print(f"Response for {os.path.basename(file_path)}: {result}")
            except ValueError:
                print(f"Non-JSON response for {os.path.basename(file_path)}: {response.text}")
        else:
            print(f"Failed for {os.path.basename(file_path)}: HTTP {response.status_code}, {response.text}")

# Iterate over all files in the directory
for file_name in os.listdir(directory):
    file_path = os.path.join(directory, file_name)
    if os.path.isfile(file_path):
        print(f"Uploading {file_name}...")
        try:
            upload_file(file_path)
        except Exception as e:
            print(f"Error uploading {file_name}: {e}")
