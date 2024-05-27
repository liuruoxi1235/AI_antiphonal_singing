import os
from vectorize_audio import AudioVectorizer
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# Initialize Milvus connection and collections
milvus_host = os.getenv('MILVUS_HOST', 'milvus-standalone')
milvus_port = os.getenv('MILVUS_PORT', '19530')

# Establish connection to Milvus
connections.connect(alias="default", host=milvus_host, port=milvus_port)

# Define a collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="audio collection")

# Ensure long_term_collection exists
long_term_collection_name = "long_term_collection"
if not utility.has_collection(long_term_collection_name):
    long_term_collection = Collection(name=long_term_collection_name, schema=schema)
else:
    long_term_collection = Collection(name=long_term_collection_name)

# Ensure short_term_collection exists
short_term_collection_name = "short_term_collection"
if not utility.has_collection(short_term_collection_name):
    short_term_collection = Collection(name=short_term_collection_name, schema=schema)
else:
    short_term_collection = Collection(name=short_term_collection_name)

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