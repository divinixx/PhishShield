#!/usr/bin/env python3
"""
PhishShield Setup and Test Script
This script sets up the environment and tests the fraud detection capabilities
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    requirements = [
        'streamlit',
        'torch',
        'scikit-learn',
        'nltk',
        'pandas',
        'numpy',
        'joblib'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def check_model_files():
    """Check if model files exist"""
    required_files = [
        'spam_classifier.pth',
        'tfidf_vectorizer.pkl',
        'label_encoder.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("\n❌ Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📝 Please run 'python spam.py' first to train and save the model.")
        return False
    else:
        print("\n✅ All model files found!")
        return True

def test_fraud_detection():
    """Test the fraud detection with sample messages"""
    print("\n🧪 Testing fraud detection capabilities...")
    
    # Import the enhanced prediction function
    try:
        from app import predict_spam_with_fraud_detection, load_model_components, download_nltk_data
        
        # Load model and components
        stop_words = download_nltk_data()
        model, tfidf, label_encoder, model_loaded = load_model_components()
        
        if not model_loaded:
            print("❌ Failed to load model components")
            return
        
        # Test cases
        test_messages = [
            "Dear customer, your debit card will be blocked within 2 hrs. Call 9876543210 to reactivate.",
            "Get a pre-approved loan of ₹5,00,000 instantly. Apply here: http://instantloan-now.net",
            "You are selected for a free iPhone! Pay ₹99 shipping at: http://applefreegift.com",
            "Hi! Are we still meeting for lunch tomorrow at 12pm?",
            "Your appointment reminder: Doctor visit on Friday at 3pm"
        ]
        
        print("\n📊 Test Results:")
        print("-" * 80)
        
        for i, message in enumerate(test_messages, 1):
            prediction, confidence, fraud_analysis = predict_spam_with_fraud_detection(
                message, model, tfidf, label_encoder, stop_words
            )
            
            print(f"\n{i}. Message: {message[:60]}...")
            print(f"   Prediction: {prediction.upper()}")
            print(f"   Confidence: {confidence:.1%}")
            if fraud_analysis:
                print(f"   Fraud Score: {fraud_analysis['fraud_score']:.1%}")
                if fraud_analysis['financial_keywords']:
                    print(f"   Financial Keywords: {', '.join(fraud_analysis['financial_keywords'])}")
                if fraud_analysis['phone_numbers']:
                    print(f"   Phone Numbers: {', '.join(fraud_analysis['phone_numbers'])}")
        
        print("\n✅ Fraud detection test completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")

def run_streamlit_app():
    """Run the Streamlit app"""
    print("\n🚀 Starting Streamlit app...")
    print("📱 The app will open in your browser automatically")
    print("🔗 Manual URL: http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop the app")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app_simple.py'])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {str(e)}")

def main():
    """Main setup and run function"""
    print("🛡️ PhishShield Setup Script")
    print("=" * 50)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Check model files
    if not check_model_files():
        return
    
    # Step 3: Test fraud detection
    test_fraud_detection()
    
    # Step 4: Ask user if they want to run the app
    print("\n" + "=" * 50)
    response = input("Do you want to run the Streamlit app now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        run_streamlit_app()
    else:
        print("\n📝 To run the app later, use: streamlit run app_simple.py")
        print("🛡️ PhishShield setup completed!")

if __name__ == "__main__":
    main()
