import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa

class AudioVectorizer:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    def extract_features(self, file_path):
        # Load the audio file
        y, sr = librosa.load(file_path, sr=16000)
        # Ensure the waveform data is of the expected shape
        input_values = self.processor(y, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            embeddings = self.model(input_values).last_hidden_state
        # Average the embeddings over the time dimension
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings.numpy()

if __name__ == "__main__":
    vectorizer = AudioVectorizer()
    features = vectorizer.extract_features("example.wav")
    print(features)