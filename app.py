from flask import Flask, request, redirect, url_for, send_from_directory, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from vectorize_audio import AudioVectorizer
from compare_audio import compare_audio
from clean import clean_collections
from uploaded_files import uploaded_files_bp
from store_vectors import store_vector

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Register the blueprint
app.register_blueprint(uploaded_files_bp, url_prefix='/uploads')

# Connect to Milvus
milvus_host = os.getenv('MILVUS_HOST', 'milvus-standalone')
milvus_port = os.getenv('MILVUS_PORT', '19530')
connections.connect("default", host=milvus_host, port=milvus_port)

# Define a collection schema
vector_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255)
]
vector_schema = CollectionSchema(vector_fields, description="vector collection")

wav_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="dummy_vector", dtype=DataType.FLOAT_VECTOR, dim=2)  # Add a dummy vector field
]
wav_schema = CollectionSchema(wav_fields, description="wav collection")

# Ensure long_term_collection exists
long_term_collection_name = "long_term_collection"
if not utility.has_collection(long_term_collection_name):
    long_term_collection = Collection(name=long_term_collection_name, schema=vector_schema)
else:
    long_term_collection = Collection(name=long_term_collection_name)

# Ensure short_term_collection exists
short_term_collection_name = "short_term_collection"
if not utility.has_collection(short_term_collection_name):
    short_term_collection = Collection(name=short_term_collection_name, schema=vector_schema)
else:
    short_term_collection = Collection(name=short_term_collection_name)

# Ensure wav_collection exists
wav_collection_name = "wav_collection"
if not utility.has_collection(wav_collection_name):
    wav_collection = Collection(name=wav_collection_name, schema=wav_schema)
else:
    wav_collection = Collection(name=wav_collection_name)

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

        # Get the optional ID from the form
        user_id = request.form.get('id')
        if user_id:
            try:
                user_id = int(user_id)
            except ValueError:
                return jsonify({'status': 'error', 'message': 'ID must be an integer'}), 400
        else:
            user_id = long_term_collection.num_entities

        # Vectorize and insert into Milvus long-term collection
        vector = vectorizer.extract_features(file_path).flatten().tolist()
        vectors = [vector]
        ids = [user_id]
        filenames = [filename]
        data = [ids, vectors, filenames]
        long_term_collection.insert(data)
        print(f"Inserted file {filename} into long_term_collection with ID {user_id}.")

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
        vectors = [vector]
        ids = [short_term_collection.num_entities]
        filenames = [filename]
        data = [ids, vectors, filenames]
        short_term_collection.insert(data)
        print(f"Inserted file {filename} into short_term_collection with ID {short_term_collection.num_entities}.")

        # Perform similarity search with long-term collection
        result = compare_audio(file_path)
        if result is not None:
            # Play the matching wav file from wav_collection
            matching_file = get_wav_file_by_id(result)
            if matching_file:
                return render_template('uploaded_files.html', files=[matching_file])
            else:
                return jsonify({'status': 'success', 'message': 'File uploaded to short-term collection', 'best_match': result, 'note': 'Matching file not found in wav collection'}), 200
        else:
            return jsonify({'status': 'success', 'message': 'File uploaded to short-term collection', 'best_match': 'No match found'}), 200
    return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

@app.route('/upload_wav', methods=['POST'])
def upload_wav():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get the optional ID from the form
        user_id = request.form.get('id')
        if user_id:
            try:
                user_id = int(user_id)
            except ValueError:
                return jsonify({'status': 'error', 'message': 'ID must be an integer'}), 400
        else:
            user_id = wav_collection.num_entities

        # Store the wav file in Milvus wav collection
        store_wav_file(file_path, user_id)
        return jsonify({'status': 'success', 'message': 'Wav file uploaded to wav collection'}), 200
    return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

@app.route('/clean_collections', methods=['POST'])
def clean_collections_route():
    try:
        clean_collections()
        return jsonify({'status': 'success', 'message': 'Collections cleaned successfully'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def store_wav_file(file_path, user_id):
    filenames = [os.path.basename(file_path)]
    ids = [user_id]
    dummy_vectors = [[0.0, 0.0]]  # Dummy vector values
    data = [ids, filenames, dummy_vectors]
    wav_collection.insert(data)
    print(f"Stored wav file {file_path} with ID {user_id} in wav_collection.")

def get_wav_file_by_id(file_id):
    try:
        results = wav_collection.query(expr=f"id == {file_id}", output_fields=["filename"])
        if results:
            return results[0]["filename"]
        else:
            return None
    except Exception as e:
        print(f"Error fetching file with ID {file_id}: {e}")
        return None

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5002, debug=True)
