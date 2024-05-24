from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Define a collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, description="example collection")

# Create a collection
collection_name = "example_collection"
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
collection = Collection(name=collection_name, schema=schema)

# Insert data into the collection
import numpy as np

# Example data
ids = [1, 2, 3]
vectors = np.random.rand(3, 128).tolist()  # Generate 3 random vectors with 128 dimensions

data = [
    ids,
    vectors
]

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

# Search for vectors
search_vectors = [vectors[0]]  # Use one of the inserted vectors for the search
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(search_vectors, "vector", search_params, limit=3)

# Print results
for result in results:
    for hit in result:
        print(f"ID: {hit.id}, Distance: {hit.distance}")
