from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'super-duper-secret-key'

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    title = "TNQI"

    files = os.listdir(app.config['UPLOAD_FOLDER'])
    images = [f for f in files if allowed_file(f)]
    images.sort(key=lambda x: os.path.getatime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True)

    return render_template('index.html', title=title, images=images)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    title = "TNQI"

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('err: no file')
            return redirect(url_for('upload'))
        files = request.files.getlist('file')

        uploaded = []
        for f in files:
            if f.filename == '':
                flash('err: empty file')
                continue
            if not allowed_file(f.filename):
                flash(f'err: inncorect file type: {f.filename}')
                continue
            safe_name = secure_filename(f.filename)

            f.save(os.path.join(app.config['UPLOAD_FOLDER'], safe_name))
            uploaded.append(safe_name)
        if uploaded:
            flash(f'Successfully uploaded file {uploaded}')
        else:
            flash('err: no files uploaded')

        return redirect(url_for('upload'))

    return render_template('upload.html', title = title)

if __name__ == '__main__':
    app.run(debug=True)
