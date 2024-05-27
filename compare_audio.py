import os
from vectorize_audio import AudioVectorizer
from pymilvus import connections, Collection

# Initialize Milvus connection and collections
milvus_host = os.getenv('MILVUS_HOST', 'milvus-standalone')
milvus_port = os.getenv('MILVUS_PORT', '19530')

# Establish connection to Milvus
connections.connect(alias="default", host=milvus_host, port=milvus_port)

# Initialize collections
long_term_collection = Collection("long_term_collection")
short_term_collection = Collection("short_term_collection")

# Initialize the audio vectorizer
vectorizer = AudioVectorizer()

def compare_audio(file_path):
    # Extract feature vector from file
    vector = vectorizer.extract_features(file_path).flatten().tolist()
    # Perform similarity search
    results = long_term_collection.search(
        data=[vector],
        anns_field="vector",  # Ensure this matches your schema
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=1
    )
    # Assuming the best match is the first result
    best_match = results[0]
    return best_match