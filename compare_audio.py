from pymilvus import connections, Collection, utility
from vectorize_audio import AudioVectorizer

def compare_audio(file_path, short_term_collection_name, long_term_collection_name):
    connections.connect("default", host="milvus-standalone", port="19530")

    # Ensure the long-term collection exists
    if not utility.has_collection(long_term_collection_name):
        raise ValueError(f"Collection '{long_term_collection_name}' does not exist")

    # Vectorize the new audio file
    vectorizer = AudioVectorizer("path/to/ResNet38_mAP=0.434.pth")
    new_vector = vectorizer.vectorize(file_path).flatten().tolist()

    # Load the long-term collection
    long_term_collection = Collection(name=long_term_collection_name)
    long_term_collection.load()

    # Search for the best match in the long-term collection
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = long_term_collection.search([new_vector], "vector", search_params, limit=1)
    
    best_match = None
    for result in results:
        for hit in result:
            best_match = {"id": hit.id, "distance": hit.distance}
    
    # Insert into the short-term collection
    if not utility.has_collection(short_term_collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=527)
        ]
        schema = CollectionSchema(fields, description=f"{short_term_collection_name} collection")
        short_term_collection = Collection(name=short_term_collection_name, schema=schema)
    else:
        short_term_collection = Collection(name=short_term_collection_name)
    short_term_collection.load()
    
    ids = [short_term_collection.num_entities]
    vectors = [new_vector]
    data = [ids, vectors]
    short_term_collection.insert(data)
    
    # Create an index if not exist
    if not short_term_collection.has_index():
        index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
            "metric_type": "L2"
        }
        short_term_collection.create_index(field_name="vector", index_params=index_params)
    short_term_collection.load()

    return True, best_match