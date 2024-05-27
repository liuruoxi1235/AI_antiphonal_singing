from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from vectorize_audio import AudioVectorizer
from compare_audio import compare_audio

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Connect to Milvus
milvus_host = os.getenv('MILVUS_HOST', 'milvus-standalone')
milvus_port = os.getenv('MILVUS_PORT', '19530')
connections.connect("default", host=milvus_host, port=milvus_port)

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

# Create an index for long_term_collection
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
    "metric_type": "L2"
}
if not long_term_collection.has_index():
    long_term_collection.create_index(field_name="vector", index_params=index_params)
long_term_collection.load()

# Create an index for short_term_collection
if not short_term_collection.has_index():
    short_term_collection.create_index(field_name="vector", index_params=index_params)
short_term_collection.load()

# Initialize Audio Vectorizer
vectorizer = AudioVectorizer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/upload_long_term', methods=['POST'])
def upload_long_term():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Vectorize and insert into Milvus long-term collection
        vector = vectorizer.extract_features(file_path).flatten().tolist()
        ids = [long_term_collection.num_entities]
        vectors = [vector]
        data = [ids, vectors]
        long_term_collection.insert(data)

        return jsonify({'status': 'success', 'message': 'File uploaded to long-term collection'}), 200
    return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

@app.route('/upload_short_term', methods=['POST'])
def upload_short_term():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Vectorize and insert into Milvus short-term collection
        vector = vectorizer.extract_features(file_path).flatten().tolist()
        ids = [short_term_collection.num_entities]
        vectors = [vector]
        data = [ids, vectors]
        short_term_collection.insert(data)

        # Perform similarity search with long-term collection
        result = compare_audio(file_path)
        return jsonify({'status': 'success', 'message': 'File uploaded to short-term collection', 'best_match': result}), 200
    return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

@app.route('/uploads')
def uploaded_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('uploaded_files.html', files=files)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5001, debug=True)