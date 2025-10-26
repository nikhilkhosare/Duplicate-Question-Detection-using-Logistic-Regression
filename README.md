# Duplicate-Question-Detection-using-Logistic-Regression

##  Project Overview

This project implements a Machine Learning model to determine if two given questions are semantically equivalent (duplicates). The application is built as a Flask web service, allowing real-time prediction via a simple web interface.

The model is trained on a subset of the Quora Question Pairs dataset and uses a **feature-rich Logistic Regression** classifier, combining textual analysis with basic similarity metrics for improved accuracy.

## Key Features

* **Binary Classification:** Predicts if a question pair is a Duplicate (`1`) or Not Duplicate (`0`).
* **Hybrid Feature Engineering:** Combines high-dimensional **TF-IDF vectors** with explicit **similarity features** (e.g., shared word count, length difference).
* **Flask API:** Provides a clean `/predict` endpoint to serve the model in real-time.
* **Model Persistence:** Uses `joblib` to save and load the trained model, avoiding lengthy retraining on every startup.

## Technology Stack

* **Python:** 3.7+
* **Machine Learning:** Scikit-learn, NumPy, SciPy
* **NLP:** NLTK (for stop word removal), TF-IDF
* **Web Framework:** Flask
* **Data:** Pandas

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd Duplicate-Question-Detection-using-Logistic-Regression

## Run the Application
Once the model is trained, the Flask server will automatically start:

# If the previous step didn't launch it, run it again:
python app.py 
Open your browser and navigate to the URL shown in the terminal: http://127.0.0.1:5000

Project Structure
DuplicateQuestionDetection/
├── app.py                  # Flask app, model training/loading, and API logic
├── train.csv               # Quora Question Pairs dataset
├── logistic_model.pkl      # Saved ML model
├── tfidf_vectorizer.pkl    # Saved TF-IDF feature extractor
└── templates/              
    └── index.html          # HTML/JS frontend for the web interface

  Author
[nikhilkhosare]
