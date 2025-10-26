# Title: Duplicate Question Detection using Flask API

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib 
from flask import Flask, request, jsonify, render_template
from scipy.sparse import hstack

# --- Global Variables for Model/Vectorizer ---
MODEL_PATH = 'logistic_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
# STEP 1: Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# STEP 2: Text Cleaning Function
def clean_text(text):
    """Clean and preprocess the given text"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# --- NEW FUNCTION: Calculate Numerical Features for a Question Pair ---
def calculate_features(q1, q2):
    """Calculates numerical features based on question text."""
    q1_clean = clean_text(q1)
    q2_clean = clean_text(q2)
    len_diff = abs(len(q1) - len(q2))
    shared_words = len(set(q1_clean.split()) & set(q2_clean.split()))
    return len_diff, shared_words, q1_clean + " " + q2_clean

# STEP 3-7: Modified for Web App
def load_or_train_model():
    """Loads model/vectorizer or trains if files don't exist"""
    try:
        # Attempt to load model and vectorizer
        print("Attempting to load saved model and vectorizer...")
        model = joblib.load(MODEL_PATH)
        tfidf = joblib.load(VECTORIZER_PATH)
        print("✅ Model and vectorizer loaded successfully!")
        return model, tfidf
    except FileNotFoundError:
        print("Saved files not found. Starting full data load and training...")
        file_path =r"D:\python\DuplicateQuestionDetection\train.csv\questions.csv" # Adjusted path based on typical structure
        
        try:
            # Load and sample data
            df = pd.read_csv(file_path).dropna(subset=['question1', 'question2', 'is_duplicate']).sample(n=10000, random_state=42)
        except Exception as e:
            print(f"FATAL: Could not load data from {file_path}. Please check your path and fix the file access issue.")
            print(f"Error: {e}")
            raise 

        # --- FEATURE CALCULATION DURING TRAINING ---
        df['q1_clean'] = df['question1'].apply(clean_text)
        df['q2_clean'] = df['question2'].apply(clean_text)
        
        df['len_diff'] = abs(df['question1'].apply(len) - df['question2'].apply(len))
        df['shared_words'] = df.apply(
            lambda row: len(set(row['q1_clean'].split()) & set(row['q2_clean'].split())), 
            axis=1
        )
        df['combined'] = df['q1_clean'] + " " + df['q2_clean']
        
        # 1. TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        X_text = tfidf.fit_transform(df['combined'])

        # 2. Numerical Features
        X_numeric = df[['len_diff', 'shared_words']].values

        # 3. Combine Features (Horizontal Stack)
        X = hstack([X_text, X_numeric])
        y = df['is_duplicate'].values
        
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = LogisticRegression(solver='liblinear', C=1.0, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        # Save model and vectorizer after training
        joblib.dump(model, MODEL_PATH)
        joblib.dump(tfidf, VECTORIZER_PATH)
        print("✅ Model trained and saved successfully!")
        return model, tfidf

# --- Flask App Setup ---
app = Flask(__name__)
model, tfidf = load_or_train_model() 

@app.route('/')
def index():
    """Renders the HTML interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for prediction."""
    try:
        data = request.get_json(force=True)
        q1 = data['question1']
        q2 = data['question2']

        # --- PREPROCESSING FOR PREDICTION ---
        len_diff, shared_words, combined_text = calculate_features(q1, q2)

        # 1. Text Vectorization (using the fitted TF-IDF)
        X_text = tfidf.transform([combined_text])
        
        # 2. Numerical Features (as a sparse matrix for hstack)
        # Note: Must be 2D array: [[feature1, feature2]]
        X_numeric = np.array([[len_diff, shared_words]])

        # 3. Combine Features
        input_vector = hstack([X_text, X_numeric])
        
        # Predict
        prediction = model.predict(input_vector)[0]
        probability = model.predict_proba(input_vector)[0][1] # Probability of being a duplicate (class 1)
        
        result = {
            'is_duplicate': int(prediction),
            'probability': f"{probability:.4f}"
        }
        
        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':

    app.run(debug=True)
