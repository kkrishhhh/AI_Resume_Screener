from flask import Flask, request, render_template, send_from_directory
import os
import pickle
import numpy as np
from werkzeug.utils import secure_filename
from utils.parser import extract_text_from_pdf
from utils.report_generator import generate_pdf
from utils.logger import write_log
import logging

# Setup Flask app
app = Flask(__name__)

# Define folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
LOG_FOLDER = 'logs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Config paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# Setup logging
logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, 'predictions.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Load model & vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return 'No file part'

    file = request.files['resume']
    if file.filename == '':
        return 'No selected file'

    # Save uploaded file
    safe_filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(file_path)

    # Extract text and predict
    extracted_text = extract_text_from_pdf(file_path)
    input_vector = vectorizer.transform([extracted_text])
    probabilities = model.predict_proba(input_vector)[0]

    # Top 3 roles with confidence
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_roles = [(model.classes_[i], round(probabilities[i] * 100, 2)) for i in top_indices]

    # Top prediction
    prediction, confidence = top_roles[0]

    # Generate report
    report_filename = safe_filename.replace('.pdf', '_report.pdf')
    report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
    generate_pdf(prediction, confidence, report_path)

    # Logging
    logging.info(f"{safe_filename} - {prediction} - {confidence:.2f}%")
    write_log(safe_filename, prediction, confidence)

    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        top_roles=top_roles,
        report_filename=report_filename
    )

@app.route('/download/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

