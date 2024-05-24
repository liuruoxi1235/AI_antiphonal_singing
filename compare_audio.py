from pymilvus import connections, Collection
import numpy as np
import torch
from vectorize_audio import AudioVectorizer

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Load the collection
collection = Collection("audio_collection")
collection.load()

# Vectorize the new audio file
vectorizer = AudioVectorizer("path/to/ResNet38_mAP=0.434.pth")
new_vector = vectorizer.vectorize("path/to/new/audio/file.wav").flatten().tolist()

# Search for the best match
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search([new_vector], "vector", search_params, limit=1)

# Print the best match result
for result in results:
    for hit in result:
        print(f"ID: {hit.id}, Distance: {hit.distance}")