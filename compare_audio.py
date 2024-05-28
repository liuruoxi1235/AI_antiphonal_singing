from pymilvus import connections, Collection
import os
from vectorize_audio import AudioVectorizer

# Initialize Milvus connection and collections
milvus_host = os.getenv('MILVUS_HOST', 'milvus-standalone')
milvus_port = os.getenv('MILVUS_PORT', '19530')

# Establish connection to Milvus
connections.connect(alias="default", host=milvus_host, port=milvus_port)

# Initialize the audio vectorizer
vectorizer = AudioVectorizer()

def compare_audio(file_path):
    long_term_collection_name = "long_term_collection"
    long_term_collection = Collection(long_term_collection_name)

    # Extract feature vector from file
    vector = vectorizer.extract_features(file_path).flatten().tolist()

    # Perform similarity search
    results = long_term_collection.search(
        data=[vector],
        anns_field="vector",  # Ensure this matches your schema
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=1,
        output_fields=["id"]  # Ensure only valid fields are requested
    )

    # Assuming the best match is the one with the smallest distance
    if results:
        best_match = results[0]  # best_match is of type hits
        if best_match and best_match.ids:
            best_match_id = best_match.ids[0]
            return best_match_id
        else:
            return None
    else:
        return None