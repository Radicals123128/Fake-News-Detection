from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import pickle
import os
from model import preprocess_text, prepare_data, train_model, load_model
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model and vectorizer
def load_models():
    try:
        if not os.path.exists('model/fake_news_model.pkl') or not os.path.exists('model/vectorizer.pkl'):
            print("Model files not found. Please train the model first.")
            return None, None

        with open('model/fake_news_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        print("Model and vectorizer loaded successfully")
        return model, vectorizer
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

model, vectorizer = load_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    model_status = 'Model Loaded' if model and vectorizer else 'Model Not Loaded'
    return render_template('index.html', model_status=model_status)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            if not model or not vectorizer:
                return jsonify({
                    'status': 'error',
                    'message': 'Model not loaded. Please train the model first.'
                })

            # Get the news text from the request
            news_text = request.form['news_text']

            # Preprocess the text
            processed_text = preprocess_text(news_text)

            # Transform the text using vectorizer
            text_vector = vectorizer.transform([processed_text])

            # Make prediction
            prediction = model.predict(text_vector)[0]

            # Get prediction probability
            prob = model.predict_proba(text_vector)[0]

            # Prepare response
            if prediction == 1:
                result = 'Fake News'
                probability = round(prob[1] * 100, 2)
            else:
                result = 'Real News'
                probability = round(prob[0] * 100, 2)

            # Save prediction to history
            save_prediction_history(news_text, result, probability)

            return jsonify({
                'status': 'success',
                'prediction': result,
                'confidence': probability,
                'text': news_text[:200] + '...' if len(news_text) > 200 else news_text
            })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file uploaded'
        })

    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No selected file'
        })

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Load and preprocess the data
            df = pd.read_csv(filepath)
            print(f"Columns in uploaded file: {df.columns.tolist()}")

            # Clean column names
            df.columns = df.columns.str.lower().str.strip()

            # Define possible column names
            text_cols = ['text', 'content', 'body', 'article']

            # Find the actual text column
            text_col = next((col for col in text_cols if col in df.columns), None)

            if not text_col:
                return jsonify({
                    'status': 'error',
                    'message': f'No valid text column found. Available columns: {df.columns.tolist()}. Looking for any of these: {text_cols}'
                })

            # Create processed content directly from the text column
            df['processed_content'] = df[text_col].fillna('').apply(preprocess_text)

            # Make predictions
            processed_texts = df['processed_content'].tolist()
            text_vectors = vectorizer.transform(processed_texts)
            predictions = model.predict(text_vectors)
            probabilities = model.predict_proba(text_vectors)

            # Prepare results
            results = []
            for i, (text, pred, prob) in enumerate(zip(processed_texts, predictions, probabilities)):
                result = {
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'prediction': 'Fake News' if pred == 1 else 'Real News',
                    'confidence': round(prob[1 if pred == 1 else 0] * 100, 2)
                }
                results.append(result)

            return jsonify({
                'status': 'success',
                'results': results
            })

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            })

    flash('Invalid file type', 'error')
    return redirect(url_for('home'))

def save_prediction_history(text, prediction, confidence):
    history_file = 'prediction_history.json'
    history = []

    # Load existing history
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)

    # Add new prediction
    history.append({
        'text': text[:200] + '...' if len(text) > 200 else text,
        'prediction': prediction,
        'confidence': confidence,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    # Keep only last 100 predictions
    history = history[-100:]

    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f)

@app.route('/history')
def get_history():
    history_file = 'prediction_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        return jsonify(history)
    return jsonify([])

@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        print("\nStarting model retraining process...")

        # Check if all required files exist
        missing_files = []
        for file_path in ['gossipcop_fake.csv', 'gossipcop_real.csv',
                         'politifact_fake.csv', 'politifact_real.csv']:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            return jsonify({
                'status': 'error',
                'message': f'Missing training data files: {", ".join(missing_files)}'
            })

        # Run the training process
        success = train_model()

        if not success:
            return jsonify({
                'status': 'error',
                'message': 'Model training failed. Check server logs for details.'
            })

        # Reload the model
        print("Reloading model...")
        global model, vectorizer
        model, vectorizer = load_models()

        if model is None or vectorizer is None:
            return jsonify({
                'status': 'error',
                'message': 'Failed to load the newly trained model.'
            })

        return jsonify({
            'status': 'success',
            'message': 'Model retrained and loaded successfully!'
        })
    except Exception as e:
        print(f"Error during model retraining: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error retraining model: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)