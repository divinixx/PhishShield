#!/usr/bin/env python3
"""
Quick test script for PhishShield fraud detection
Tests the specific fraud messages mentioned by the user
"""

def test_fraud_messages():
    """Test the specific fraud messages"""
    print("🛡️ PhishShield Fraud Detection Test")
    print("=" * 60)
    
    # Your specific fraud examples
    fraud_messages = [
        "Dear customer, your debit card will be blocked within 2 hrs. Call 9876543210 to reactivate.",
        "Get a pre-approved loan of ₹5,00,000 instantly. Apply here: http://instantloan-now.net",
        "You are selected for a free iPhone! Pay ₹99 shipping at: http://applefreegift.com"
    ]
    
    print("Testing fraud messages that should be detected as FRAUD/SPAM:")
    print("-" * 60)
    
    try:
        # Import required functions
        from app import (
            predict_spam_with_fraud_detection, 
            load_model_components, 
            download_nltk_data,
            analyze_fraud_indicators
        )
        
        # Load components
        print("📥 Loading model components...")
        stop_words = download_nltk_data()
        model, tfidf, label_encoder, model_loaded = load_model_components()
        
        if not model_loaded:
            print("❌ Model not loaded. Please run 'python spam.py' first to train the model.")
            return
        
        print("✅ Model loaded successfully!\n")
        
        for i, message in enumerate(fraud_messages, 1):
            print(f"🧪 Test {i}:")
            print(f"📨 Message: {message}")
            print()
            
            # Analyze with rule-based fraud detection
            fraud_analysis = analyze_fraud_indicators(message)
            print(f"🔍 Fraud Analysis:")
            print(f"   📊 Fraud Score: {fraud_analysis['fraud_score']:.1%}")
            
            if fraud_analysis['financial_keywords']:
                print(f"   💳 Financial Keywords: {', '.join(fraud_analysis['financial_keywords'])}")
            
            if fraud_analysis['phishing_indicators']:
                print(f"   🎣 Phishing Indicators: {', '.join(fraud_analysis['phishing_indicators'])}")
            
            if fraud_analysis['phone_numbers']:
                print(f"   📞 Phone Numbers: {', '.join(fraud_analysis['phone_numbers'])}")
            
            if fraud_analysis['suspicious_urls']:
                print(f"   🔗 Suspicious URLs: {', '.join(fraud_analysis['suspicious_urls'])}")
            
            if fraud_analysis['other_indicators']:
                print(f"   ⚠️ Other Indicators: {', '.join(fraud_analysis['other_indicators'])}")
            
            # Get enhanced prediction
            prediction, confidence, _ = predict_spam_with_fraud_detection(
                message, model, tfidf, label_encoder, stop_words
            )
            
            print(f"\n🎯 Final Prediction:")
            print(f"   📋 Classification: {prediction.upper()}")
            print(f"   🎚️ Confidence: {confidence:.1%}")
            
            # Determine if detection was successful
            if prediction.lower() in ['fraud', 'spam', 'suspicious']:
                print("   ✅ SUCCESS: Correctly identified as fraudulent!")
            else:
                print("   ❌ MISSED: Not detected as fraud (this needs improvement)")
            
            print("\n" + "-" * 60)
        
        print("\n📊 Test Summary:")
        print("The enhanced PhishShield system now uses:")
        print("✓ Rule-based fraud keyword detection")
        print("✓ Phone number pattern recognition")
        print("✓ URL analysis for suspicious domains")
        print("✓ Financial terms detection")
        print("✓ Time pressure indicators")
        print("✓ Money amount recognition")
        print("\nThis hybrid approach should catch the fraud messages that")
        print("the basic ML model might miss!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install streamlit torch scikit-learn nltk pandas numpy joblib")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_fraud_messages()
