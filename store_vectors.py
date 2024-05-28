from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from vectorize_audio import AudioVectorizer

def store_vector(file_path, collection_name, user_id=None):
    connections.connect("default", host="milvus-standalone", port="19530")

    # Define a collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255)
    ]
    schema = CollectionSchema(fields, description=f"{collection_name} collection")

    # Create or load the collection
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name)
    collection.load()

    # Vectorize and insert data into the collection
    vectorizer = AudioVectorizer()
    mfcc_vector = vectorizer.extract_features(file_path).flatten().tolist()

    # Determine the ID to use
    if user_id is None:
        user_id = collection.num_entities
    else:
        try:
            user_id = int(user_id)
        except ValueError:
            raise ValueError("ID must be an integer")

    ids = [user_id]
    vectors = [mfcc_vector]
    filenames = [os.path.basename(file_path)]

    data = [ids, vectors, filenames]
    collection.insert(data)
    
    # Create an index if not exist
    if not collection.has_index():
        index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
            "metric_type": "L2"
        }
        collection.create_index(field_name="vector", index_params=index_params)
    collection.load()
    
    return True
