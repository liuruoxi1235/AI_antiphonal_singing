from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import numpy as np
import os
import torch
from vectorize_audio import AudioVectorizer

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Define a collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=527)
]
schema = CollectionSchema(fields, description="audio collection")

# Create a collection
collection_name = "audio_collection"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
collection = Collection(name=collection_name, schema=schema)

# Vectorize and insert data into the collection
vectorizer = AudioVectorizer("path/to/ResNet38_mAP=0.434.pth")
data_dir = 'path/to/your/audio/files'
ids = []
vectors = []

for idx, file_name in enumerate(os.listdir(data_dir)):
    file_path = os.path.join(data_dir, file_name)
    mfcc_vector = vectorizer.vectorize(file_path).flatten().tolist()
    ids.append(idx)
    vectors.append(mfcc_vector)

data = [ids, vectors]
collection.insert(data)

# Create an index
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
    "metric_type": "L2"
}
collection.create_index(field_name="vector", index_params=index_params)

# Load the collection into memory
collection.load()

# Count the number of entities in the collection
entity_count = collection.num_entities
print(f"Number of entities in the collection: {entity_count}")