from flask import Blueprint, render_template, jsonify, send_file
from pymilvus import utility, Collection
import os

uploaded_files_bp = Blueprint('uploaded_files', __name__)

long_term_collection_name = "long_term_collection"
short_term_collection_name = "short_term_collection"

@uploaded_files_bp.route('/')
def uploaded_files():
    filenames = []

    # Fetch filenames from long-term collection
    if utility.has_collection(long_term_collection_name):
        long_term_collection = Collection(long_term_collection_name)
        try:
            long_term_collection.load()
            long_term_entities = long_term_collection.query(expr="*", output_fields=["filename"])
            filenames += [entity["filename"] for entity in long_term_entities]
            print(f"Fetched {len(long_term_entities)} files from long_term_collection.")
        except Exception as e:
            print(f"Error loading long_term_collection: {e}")

    # Fetch filenames from short-term collection
    if utility.has_collection(short_term_collection_name):
        short_term_collection = Collection(short_term_collection_name)
        try:
            short_term_collection.load()
            short_term_entities = short_term_collection.query(expr="*", output_fields=["filename"])
            filenames += [entity["filename"] for entity in short_term_entities]
            print(f"Fetched {len(short_term_entities)} files from short_term_collection.")
        except Exception as e:
            print(f"Error loading short_term_collection: {e}")

    print(f"Total files fetched: {len(filenames)}")
    return render_template('uploaded_files.html', files=filenames)

@uploaded_files_bp.route('/<filename>')
def uploaded_file(filename):
    # Replace this with logic to fetch the file data from your actual storage
    file_path = os.path.join('uploads', filename)
    if os.path.isfile(file_path):
        return send_file(file_path)
    return jsonify({'status': 'error', 'message': 'File not found'}), 404