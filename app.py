from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from file_handler import *

app = Flask(__name__)
app.secret_key = 'super-duper-secret-key'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    title = "TNQI"
    images = get_uploaded_files(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', title=title, images=images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return serve_file(filename, app.config['UPLOAD_FOLDER'])

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    title = "TNQI"

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('err: no file')
            return refirect(url_for('upload'))
        
        files = request.files.getlist('file')
        saved = save_uploaded_files(files, app.config['UPLOAD_FOLDER'])

        if saved:
            flash('Successfully uploaded')
        else:
            flash('err: no files uploaded')
    
    return render_template('upload.html', title=title)

if __name__ == '__main__':
    app.run(debug=True)
