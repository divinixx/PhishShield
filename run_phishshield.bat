@echo off
echo 🛡️ PhishShield - Starting Streamlit App
echo ================================

echo 📦 Installing required packages...
pip install streamlit torch scikit-learn nltk pandas numpy joblib

echo.
echo 🔍 Checking for model files...
if not exist "spam_classifier.pth" (
    echo ❌ Model file not found!
    echo Please run: python spam.py
    echo to train and save the model first.
    pause
    exit /b 1
)

if not exist "tfidf_vectorizer.pkl" (
    echo ❌ TF-IDF vectorizer not found!
    echo Please run: python spam.py
    echo to train and save the model first.
    pause
    exit /b 1
)

if not exist "label_encoder.pkl" (
    echo ❌ Label encoder not found!
    echo Please run: python spam.py
    echo to train and save the model first.
    pause
    exit /b 1
)

echo ✅ All model files found!
echo.
echo 🚀 Starting PhishShield app...
echo 📱 The app will open in your browser
echo 🔗 URL: http://localhost:8501
echo.
echo 💡 Press Ctrl+C to stop the app
echo ================================

streamlit run app_simple.py

echo.
echo 👋 PhishShield app stopped.
pause
