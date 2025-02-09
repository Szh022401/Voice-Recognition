from flask import Flask, request, render_template
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os
import tempfile
from datetime import datetime
from pyngrok import ngrok

# Trained Model
MODEL_PATH = "global_model.pth"

app = Flask(__name__)

UPLOAD_FOLDER = 'audio_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Entire model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.eval()

# encoder
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
encoder_state_dict = checkpoint["encoder_state_dict"]
model.model.encoder.load_state_dict(encoder_state_dict, strict=False)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files.get('audio_file')
    if not audio_file:
        return render_template('index.html', transcription="No file uploaded")

    # save to audio_files
    filename = datetime.now().strftime("%Y%m%d%H%M%S_") + audio_file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(file_path)

    audio_data, sr = librosa.load(file_path, sr=16000)
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return render_template('index.html', transcription=transcription)

# Kill all ngrok channels
ngrok.kill()

# Set ngrok Authtoken
ngrok.set_auth_token("2rGgklJmcKyJ1QlqZh7H5N0lMLn_2W4mrs724RqRZL5LVhqND")

# Set ngrok channel
public_url = ngrok.connect(5000)
print("Public URL:", public_url)

if __name__ == '__main__':
    app.run(port=5000, debug=False)