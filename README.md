# Voice Recognition with Federated Learning

This project implements a **speech-to-text transcription system** using **OpenAI's Whisper model**. It also integrates **federated learning** via **Flower (FL)** to train models across multiple devices without sharing raw audio data.

## 📌 Features
- **Real-time speech-to-text transcription** with Whisper
- **Federated Learning** for privacy-preserving model training
- **Flask API** for processing audio files
- **Ngrok integration** for public access to the Flask app

## 🚀 Getting Started

### 1️⃣ Install Dependencies
Ensure you have Python installed, then run:

```sh
pip install -r requirements.txt
```

### 2️⃣ Run the Flask Server
```sh
python app.py
```
This will start a Flask server and provide a public URL via **ngrok**.

### 3️⃣ Transcribe Audio
- Upload an audio file (WAV, FLAC, MP3, etc.).
- The model will process the audio and return the transcription.

## 🎯 Federated Learning Training

To participate in federated learning, run:

```sh
python client.py --cid <client_id> --server_address <server_ip>
```

To start the server:

```sh
python server.py --num_rounds 10 --server_address <server_ip>
```

## 📂 Project Structure

```
📂 Voice_Recognition
│── app.py                  # Flask API for speech-to-text
│── client.py                # Federated learning client
│── server.py                # Federated learning server
│── utils.py                 # Helper functions
│── global_model.pth         # Pretrained model weights
│── requirements.txt         # Required Python packages
│── audio_files/             # Uploaded audio files
│── client_datasets/         # Processed client datasets
│── README.md                # Documentation
```

## 🛠 Configuration

Edit `.gitignore` to prevent large files from being committed:

```
audio_files/
client_datasets/
*.pth
```

If you've accidentally added large files, remove them:

```sh
git rm --cached <file>
git commit -m "Removed large files"
git push origin main
```

## 🔗 Dependencies
- **Flask** (API)
- **Librosa** (Audio processing)
- **Torch** (Machine learning)
- **Transformers** (Whisper model)
- **Ngrok** (Remote access)
- **Flower (FL)** (Federated learning)

## 📌 Future Work
- Add support for **real-time streaming transcription**
- Improve **federated learning strategies**
- Deploy to **cloud services** for scalability

## 🤝 Contributors
- **[Zehua SUn]** - Initial development

