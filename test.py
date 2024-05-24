from pymilvus import connections, utility

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Get Milvus server version
version = utility.get_server_version()
print(f"Milvus server version: {version}")