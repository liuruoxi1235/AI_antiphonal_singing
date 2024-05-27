from flask import Flask, request, jsonify

from store_vectors import store_vector
from compare_audio import compare_audio

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Audio Matching Service!"

@app.route('/upload_long_term', methods=['POST'])
def upload_long_term():
    audio_file = request.files['file']
    filename = audio_file.filename
    filepath = f"/path/to/save/{filename}"
    audio_file.save(filepath)
    success = store_vector(filepath, "long_term_collection")
    return jsonify({"success": success})

@app.route('/upload_short_term', methods=['POST'])
def upload_short_term():
    audio_file = request.files['file']
    filename = audio_file.filename
    filepath = f"/path/to/save/{filename}"
    audio_file.save(filepath)
    success, best_match = compare_audio(filepath, "short_term_collection", "long_term_collection")
    return jsonify({"success": success, "best_match": best_match})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)