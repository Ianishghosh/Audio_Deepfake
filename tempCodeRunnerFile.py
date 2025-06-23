from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from utils import run_inference

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'flac', 'mp3', 'm4a'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('audio')

    if not file or file.filename == '':
        return render_template("result.html", status="error", message="❌ No audio file selected.")

    if not allowed_file(file.filename):
        return render_template("result.html", status="error", message="❌ Unsupported audio format.")

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    result = run_inference(path)

    # Optional: clean up uploaded file
    try:
        os.remove(path)
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

    if result['status'] == 'error':
        return render_template("result.html", status="error", message=f"❌ Error: {result['message']}")

    prediction = "REAL" if result['prediction'] > 0.5 else "FAKE"
    confidence = f"{result['prediction']:.4f}"

    return render_template("result.html", status="success", prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
