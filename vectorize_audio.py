import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

class AudioVectorizer:
    def __init__(self):
        self.model_url = "https://tfhub.dev/google/vggish/1"
        self.model = hub.load(self.model_url)

    def extract_features(self, file_path):
        # Load the audio file
        y, sr = librosa.load(file_path, sr=16000)
        # Convert the waveform to the expected input shape for VGGish
        waveform = y.reshape(1, -1)
        # Extract features using VGGish
        embeddings = self.model(waveform)
        return embeddings.numpy()

if __name__ == "__main__":
    vectorizer = AudioVectorizer()
    features = vectorizer.extract_features("example.wav")
    print(features)