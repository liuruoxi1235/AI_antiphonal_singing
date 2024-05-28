import os
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

def clean_collections():
    connections.connect("default", host="milvus-standalone", port="19530")

    collections = ["long_term_collection", "short_term_collection", "wav_collection"]

    for collection_name in collections:
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.drop()

            # Redefine the collection schema
            if collection_name == "wav_collection":
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=2)  # Add a dummy vector field
                ]
                schema = CollectionSchema(fields, description=f"{collection_name} collection")
            else:
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
                    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255)
                ]
                schema = CollectionSchema(fields, description=f"{collection_name} collection")
            
            collection = Collection(name=collection_name, schema=schema)
            
            # Create index if not wav_collection
            if collection_name != "wav_collection":
                index_params = {
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                    "metric_type": "L2"
                }
                collection.create_index(field_name="vector", index_params=index_params)
            print(f"Collection {collection_name} cleaned, recreated, and indexed.")
        else:
            print(f"Collection {collection_name} does not exist.")

    # Clear the uploads directory
    uploads_dir = 'uploads'
    for filename in os.listdir(uploads_dir):
        file_path = os.path.join(uploads_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file {file_path}")

if __name__ == "__main__":
    clean_collections()
