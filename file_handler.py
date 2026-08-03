import os
from werkzeug.utils import secure_filename
from flask import send_from_directory

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_uploaded_files(upload_folder):
    if not os.path.exists(upload_folder):
        return []
    files = os.listdir(upload_folder)
    images = [f for f in files if allowed_file(f)]
    images.sort(key=lambda x: os.path.getmtine(os.path.join(upload_folder, x)), reverse=True)
    return images

def save_uploaded_files(files, upload_folder):
    saved = []
    for f in files:
        if f.filename == '':
            continue
        if not allowed_file(f.filename):
            continue
        safe_name = secure_filename(f.filename)
        filepath = os.path.join(upload_folder, safe_name)
        f.save(filepath)
        saved.append(safe_name)
    return saved

def serve_file(filename, upload_folder):
    return send_from_directory(upload_folder, filename)
