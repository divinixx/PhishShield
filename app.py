import streamlit as st
import torch
import torch.nn as nn
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return set(stopwords.words('english'))

# Define the same model architecture as training
class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model and preprocessing components
@st.cache_resource
def load_model_components():
    try:
        # Load TF-IDF vectorizer
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        
        # Load label encoder
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Load the trained model
        model = SpamClassifier(input_dim=5000)  # Same as max_features in training
        model.load_state_dict(torch.load('spam_classifier.pth', map_location='cpu'))
        model.eval()
        
        return model, tfidf, label_encoder, True
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        return None, None, None, False

# Enhanced fraud detection keywords and patterns
FINANCIAL_FRAUD_KEYWORDS = [
    'debit card', 'credit card', 'atm card', 'pin number', 'otp', 'cvv', 'account blocked',
    'verify account', 'update kyc', 'urgent action', 'account suspended', 'security alert',
    'bank account', 'net banking', 'upi', 'paytm', 'phonepe', 'googlepay', 'transaction failed',
    'immediate action', 'call back', 'verify identity', 'bank officer', 'reactivate',
    'pre-approved loan', 'instant loan', 'loan approved', 'apply now'
]

PHISHING_INDICATORS = [
    'urgent', 'immediate', 'act now', 'limited time', 'click here', 'verify now',
    'confirm', 'update', 'suspended', 'blocked', 'winner', 'congratulations',
    'prize', 'lottery', 'free', 'offer', 'discount', 'cashback', 'refund',
    'claim', 'selected', 'lucky', 'expires', 'hurry'
]

SUSPICIOUS_URLS = [
    'http://', 'bit.ly', 'tinyurl', '.tk', '.ml', '.ga', '.cf'
]

PHONE_PATTERNS = [
    r'\b\d{10}\b',  # 10-digit phone numbers
    r'\b\d{4}-\d{6}\b',  # 4-6 digit patterns
    r'\b\d{5}-\d{5}\b'  # 5-5 digit patterns
]

# Text preprocessing function
def preprocess_text(text, stop_words):
    """Preprocess text exactly as done during training"""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Enhanced fraud detection function
def analyze_fraud_indicators(message):
    """Analyze message for specific fraud indicators"""
    message_lower = message.lower()
    
    fraud_score = 0
    detected_indicators = []
    
    # Check for financial fraud keywords
    financial_keywords_found = []
    for keyword in FINANCIAL_FRAUD_KEYWORDS:
        if keyword in message_lower:
            financial_keywords_found.append(keyword)
            fraud_score += 0.3
    
    # Check for phishing indicators
    phishing_indicators_found = []
    for indicator in PHISHING_INDICATORS:
        if indicator in message_lower:
            phishing_indicators_found.append(indicator)
            fraud_score += 0.2
    
    # Check for suspicious URLs
    suspicious_urls_found = []
    for url_pattern in SUSPICIOUS_URLS:
        if url_pattern in message_lower:
            suspicious_urls_found.append(url_pattern)
            fraud_score += 0.4
    
    # Check for phone numbers
    phone_numbers_found = []
    for pattern in PHONE_PATTERNS:
        matches = re.findall(pattern, message)
        phone_numbers_found.extend(matches)
        if matches:
            fraud_score += 0.3
    
    # Additional specific patterns
    if re.search(r'₹\s*\d+', message) or re.search(r'\$\s*\d+', message):
        detected_indicators.append("Money amount mentioned")
        fraud_score += 0.2
    
    if re.search(r'within \d+ (hrs|hours|minutes)', message_lower):
        detected_indicators.append("Time pressure")
        fraud_score += 0.3
    
    # Normalize fraud score
    fraud_score = min(fraud_score, 1.0)
    
    return {
        'fraud_score': fraud_score,
        'financial_keywords': financial_keywords_found,
        'phishing_indicators': phishing_indicators_found,
        'suspicious_urls': suspicious_urls_found,
        'phone_numbers': phone_numbers_found,
        'other_indicators': detected_indicators
    }

# Enhanced prediction function
def predict_spam_with_fraud_detection(message, model, tfidf, label_encoder, stop_words):
    """Enhanced prediction with rule-based fraud detection - Binary classification: SPAM or LEGITIMATE"""
    try:
        # First, analyze with rule-based fraud detection
        fraud_analysis = analyze_fraud_indicators(message)
        
        # Preprocess the message
        processed_message = preprocess_text(message, stop_words)
        
        if not processed_message.strip():
            return "Error: Empty message after preprocessing", 0.0, fraud_analysis
        
        # Transform using TF-IDF
        message_tfidf = tfidf.transform([processed_message]).toarray()
        
        # Convert to tensor and predict
        with torch.no_grad():
            inputs = torch.tensor(message_tfidf, dtype=torch.float32)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Get ML prediction and confidence
        ml_predicted_class = label_encoder.inverse_transform([predicted.item()])[0]
        ml_confidence = probabilities[0][predicted.item()].item()
        
        # Binary classification: If fraud indicators are present, classify as SPAM
        if fraud_analysis['fraud_score'] >= 0.3:
            # Any significant fraud score makes it SPAM (includes financial fraud)
            final_prediction = "spam"
            final_confidence = max(0.8, fraud_analysis['fraud_score'] + 0.1)
        else:
            # Low fraud score - trust ML prediction
            final_prediction = ml_predicted_class
            final_confidence = ml_confidence
        
        return final_prediction, final_confidence, fraud_analysis
    except Exception as e:
        return f"Error: {str(e)}", 0.0, None

# Streamlit App Configuration
st.set_page_config(
    page_title="PhishShield - Spam Detection",
    page_icon="🛡️",
    layout="wide"
)

# App Header
st.title("🛡️ PhishShield - Spam Detection System")
st.markdown("### Binary Classification: LEGITIMATE vs SPAM (includes financial fraud)")

# Load resources
stop_words = download_nltk_data()
model, tfidf, label_encoder, model_loaded = load_model_components()

if not model_loaded:
    st.error("⚠️ Model files not found! Please run spam.py first to train and save the model.")
    st.info("Required files: spam_classifier.pth, tfidf_vectorizer.pkl, label_encoder.pkl")
    st.stop()

st.success("✅ Model loaded successfully!")

# Main interface
st.markdown("---")

# Text input
st.markdown("### 📝 Enter a message to analyze:")
user_message = st.text_area(
    "Message:",
    placeholder="Type your message here...\nExample: 'Congratulations! You've won $1000! Call now!'",
    height=120
)

# Prediction button and results
if st.button("🔍 Analyze Message", type="primary"):
    if user_message.strip():
        with st.spinner("Analyzing message..."):
            prediction, confidence, fraud_analysis = predict_spam_with_fraud_detection(
                user_message, model, tfidf, label_encoder, stop_words
            )
            
            if prediction.startswith("Error"):
                st.error(prediction)
            else:
                # Display results
                st.markdown("### 📊 Analysis Results:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction.lower() == 'spam':
                        st.error(f"🚨 **SPAM DETECTED**")
                        st.markdown(f"**Confidence:** {confidence:.1%}")
                        st.markdown("*Includes financial fraud, phishing, and traditional spam*")
                    else:
                        st.success(f"✅ **LEGITIMATE MESSAGE**")
                        st.markdown(f"**Confidence:** {confidence:.1%}")
                
                with col2:
                    # Show confidence meter
                    st.metric("Confidence Score", f"{confidence:.1%}")
                    if fraud_analysis:
                        st.metric("Fraud Risk Score", f"{fraud_analysis['fraud_score']:.1%}")
                
                # Show detailed fraud analysis
                if fraud_analysis and fraud_analysis['fraud_score'] > 0:
                    st.markdown("### 🔍 Detailed Fraud Analysis:")
                    
                    if fraud_analysis['financial_keywords']:
                        st.warning(f"**Financial Keywords Found:** {', '.join(fraud_analysis['financial_keywords'])}")
                    
                    if fraud_analysis['phishing_indicators']:
                        st.warning(f"**Phishing Indicators:** {', '.join(fraud_analysis['phishing_indicators'])}")
                    
                    if fraud_analysis['phone_numbers']:
                        st.warning(f"**Phone Numbers Found:** {', '.join(fraud_analysis['phone_numbers'])}")
                    
                    if fraud_analysis['suspicious_urls']:
                        st.warning(f"**Suspicious URLs:** {', '.join(fraud_analysis['suspicious_urls'])}")
                    
                    if fraud_analysis['other_indicators']:
                        st.warning(f"**Other Risk Factors:** {', '.join(fraud_analysis['other_indicators'])}")
                
                # Additional info
                st.markdown("---")
                if prediction.lower() == 'spam':
                    st.error("🚨 **SPAM MESSAGE DETECTED!**")
                    st.markdown("**⚠️ This message shows signs of spam or fraud. Be extremely cautious:**")
                    st.markdown("""
                    - **NEVER** share personal information (PIN, OTP, passwords)
                    - **DO NOT** click on suspicious links
                    - **VERIFY** the sender through official channels
                    - **REPORT** this message to authorities if it's financial fraud
                    - **BLOCK** the sender immediately
                    """)
                else:
                    st.info("ℹ️ **This message appears legitimate.** However, always:")
                    st.markdown("""
                    - Verify sender identity for important requests
                    - Be cautious with links and downloads
                    - Trust your instincts if something feels off
                    """)
    else:
        st.warning("Please enter a message to analyze.")

# Examples section
st.markdown("---")
st.markdown("### 🎯 Try These Examples:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Spam Examples (Including Financial Fraud):**")
    spam_examples = [
        "Dear customer, your debit card will be blocked within 2 hrs. Call 9876543210 to reactivate.",
        "Get a pre-approved loan of ₹5,00,000 instantly. Apply here: http://instantloan-now.net",
        "You are selected for a free iPhone! Pay ₹99 shipping at: http://applefreegift.com",
        "URGENT! Your bank account will be closed. Call 555-0123 immediately!",
        "Congratulations! You've won $5000! Click here to claim your prize now!",
        "Free money! Send your details to claim your lottery winnings today!"
    ]
    
    for i, example in enumerate(spam_examples):
        if st.button(f"Try Spam Example {i+1}", key=f"spam_{i}"):
            st.session_state.example_message = example

with col2:
    st.markdown("**Legitimate Examples:**")
    ham_examples = [
        "Hi! Are we still meeting for lunch tomorrow at 12pm?",
        "Your appointment reminder: Doctor visit on Friday at 3pm",
        "Thanks for the birthday wishes! Had a great time at the party.",
        "The package will be delivered tomorrow between 10 AM and 2 PM.",
        "Reminder: Team meeting at 3 PM in conference room B.",
        "Happy birthday! Hope you have a wonderful day!"
    ]
    
    for i, example in enumerate(ham_examples):
        if st.button(f"Try Legitimate Example {i+1}", key=f"ham_{i}"):
            st.session_state.example_message = example

# Display selected example
if 'example_message' in st.session_state:
    st.text_area("Selected Example:", value=st.session_state.example_message, height=80, key="example_display")
    if st.button("🔍 Analyze This Example", key="analyze_example"):
        prediction, confidence, fraud_analysis = predict_spam_with_fraud_detection(
            st.session_state.example_message, model, tfidf, label_encoder, stop_words
        )
        
        if prediction.lower() == 'spam':
            st.error(f"🚨 **SPAM DETECTED** - Confidence: {confidence:.1%}")
            if fraud_analysis and fraud_analysis['fraud_score'] > 0.3:
                st.warning(f"Contains fraud indicators (Risk Score: {fraud_analysis['fraud_score']:.1%})")
        else:
            st.success(f"✅ **LEGITIMATE MESSAGE** - Confidence: {confidence:.1%}")

# Footer
st.markdown("---")
st.markdown("### ℹ️ About PhishShield")
st.info("""
**PhishShield** is an AI-powered spam and fraud detection system that helps identify potentially harmful messages.
The system combines machine learning with rule-based fraud detection to catch financial scams, phishing attempts, and spam messages.

**Key Features:**
- ML-based spam detection (~97% accuracy)
- Financial fraud keyword detection
- Phishing indicator analysis
- Phone number and URL pattern recognition
- Real-time risk assessment
""")

# Technical details in expander
with st.expander("🔧 Technical Details"):
    st.markdown("""
    **Hybrid Detection System:**
    - **ML Model:** Neural Network with TF-IDF features (5000 dimensions)
    - **Rule-based:** Financial fraud keywords, phishing indicators
    - **Pattern Recognition:** Phone numbers, suspicious URLs, money amounts
    
    **Model Architecture:**
    - Input Layer: 5000 features (TF-IDF)
    - Hidden Layer 1: 128 neurons (ReLU activation)
    - Hidden Layer 2: 64 neurons (ReLU activation)
    - Output Layer: 2 classes (Spam/Ham)
    
    **Enhanced Detection:**
    - Financial fraud keywords (30+ patterns)
    - Phishing indicators (20+ patterns)
    - Time pressure detection
    - URL analysis
    - Phone number extraction
    
    **Framework:** PyTorch, Scikit-learn, NLTK, Streamlit
    """)

with st.expander("🛡️ Fraud Protection Tips"):
    st.markdown("""
    **Red Flags to Watch For:**
    - Urgent requests for personal information
    - Threats of account closure or suspension
    - Requests for OTP, PIN, or passwords
    - Too-good-to-be-true offers
    - Suspicious links or phone numbers
    
    **Safety Guidelines:**
    - Never share sensitive information via SMS
    - Verify requests through official channels
    - Be skeptical of unsolicited offers
    - Report suspicious messages to authorities
    - Keep your personal information private
    
    **Remember:** Banks never ask for sensitive information via SMS!
    """)
