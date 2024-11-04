from .base_cloner import BaseCloner
from config import XI_LABS_API_KEY
import requests

class ElevenlabsCloner(BaseCloner):
    name = "elevenlabs"
    url = "https://api.elevenlabs.io/v1/"
    def __init__(self):
        self.voice_id = None
    def extract_voice(self,input_path):
        url = self.url + "voices/add"
        headers = {
            "xi-api-key" : XI_API_KEY
        }
        data = {
            'name' : "242-voice",
            'remove_background_noise' : True,
            'description' : '',
            'labels' : ''
        }
        try:
            files = [('files',(input_path,open(input_path,"rb"),'audio/wav'))]
        except Exception as e:
            print(f"Exception: {e}")
            return
        resp = requests.post(url,headers=headers,data=data,files=files)
        
    def apply_voice(self):
        pass