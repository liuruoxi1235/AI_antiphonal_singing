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
connections.connect("default", host="milvus-standalone", port="19530")  # Revised host to milvus-standalone

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

# Create an index
index_params = {
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
    "metric_type": "L2"
}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()

# Initialize Audio Vectorizer
vectorizer = AudioVectorizer("path/to/ResNet38_mAP=0.434.pth")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Vectorize and insert into Milvus
            vector = vectorizer.vectorize(file_path).flatten().tolist()
            ids = [collection.num_entities]
            vectors = [vector]
            data = [ids, vectors]
            collection.insert(data)

            return redirect(url_for('uploaded_files'))
    return render_template('upload.html')

@app.route('/uploads')
def uploaded_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('uploads.html', files=files)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/compare', methods=['POST'])
def compare_files():
    data = request.get_json()
    file1 = data['file1']
    file2 = data['file2']
    result = compare_audio(file1, file2)
    return jsonify({'similarity': result}), 200

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5001)  # Ensure app runs on port 5001