import argparse
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def main():
    parser = argparse.ArgumentParser(description="Whisper ASR Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to global_model.pth")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file (e.g. .wav, .flac)")
    args = parser.parse_args()

    # Load Whisper processor and model
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.eval()

    # Load our trained model as encoder parameters
    checkpoint = torch.load(args.model_path, map_location="cpu")
    encoder_state_dict = checkpoint["encoder_state_dict"]
    model.model.encoder.load_state_dict(encoder_state_dict, strict=False)

    # Load audio file
    audio_data, sr = librosa.load(args.audio_path, sr=16000)
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")

    # Transcription
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Transcription:", transcription)


if __name__ == "__main__":
    main()