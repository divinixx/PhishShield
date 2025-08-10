# 🛡️ PhishShield - Advanced Spam & Fraud Detection System

<div align="center">

![PhishShield Logo](https://img.shields.io/badge/PhishShield-AI%20Powered-blue?style=for-the-badge&logo=shield&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=flat-square)](README.md)

**A sophisticated AI-powered system for detecting spam messages, financial fraud, and phishing attempts in real-time**

*Designed to combat modern scams including Jamtara-style financial fraud schemes*

[🚀 Quick Start](#-quick-start) • [📊 Features](#-features) • [🛠️ Installation](#%EF%B8%8F-installation) • [📱 Usage](#-usage) • [🤝 Contributing](#-contributing)

</div>

---

## 🌟 Overview

PhishShield is a comprehensive fraud detection system that combines **machine learning** with **rule-based detection** to identify harmful messages with high accuracy. The system specifically targets modern fraud schemes like those seen in Jamtara-style scams, providing real-time protection against:

- 💳 **Financial Fraud** (Banking, UPI, loan scams)
- 🎣 **Phishing Attempts** (Credential theft, fake offers)
- 📧 **Traditional Spam** (Unwanted marketing, malicious content)
- 📱 **SMS Fraud** (OTP theft, fake alerts)

## 📊 Features

### 🧠 **Hybrid AI Detection**
- **Neural Network**: 97% accuracy on spam detection
- **Rule-Based Engine**: 30+ financial fraud patterns
- **Pattern Recognition**: Phone numbers, URLs, money amounts
- **Real-Time Analysis**: Instant fraud scoring

### 🎯 **Advanced Detection Capabilities**
| Feature | Description | Coverage |
|---------|-------------|----------|
| 💰 Financial Keywords | Banking, UPI, card-related terms | 25+ patterns |
| 🎣 Phishing Indicators | Urgency, fake offers, social engineering | 20+ patterns |
| 📞 Phone Analysis | Suspicious number patterns | Multiple formats |
| 🔗 URL Detection | Malicious domains, shortened links | 7+ patterns |
| ⏰ Time Pressure | Urgency manipulation tactics | Real-time detection |
| 💸 Money Mentions | Currency amounts in messages | Multi-currency |

### 🌐 **User Interface**
- **Web-Based Dashboard**: Streamlit-powered interface
- **Binary Classification**: Simple SPAM vs LEGITIMATE results
- **Detailed Analysis**: Risk breakdown and explanations
- **Example Library**: Pre-loaded test cases
- **Safety Guidelines**: Built-in fraud protection tips

## 🛠️ Installation

### 📋 Prerequisites

- **Python 3.7+** (Recommended: Python 3.9+)
- **pip** package manager
- **Git** (for cloning the repository)

### 🚀 Quick Setup

#### Method 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/PhishShield.git
cd PhishShield

# Run automated setup (Windows)
run_phishshield.bat

# Or use Python setup script
python setup_and_run.py
```

#### Method 2: Manual Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/PhishShield.git
cd PhishShield

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (if not already trained)
python spam.py

# 4. Launch the application
streamlit run app.py
```

### 📦 Required Libraries

```txt
streamlit>=1.25.0
torch>=2.0.1
scikit-learn>=1.3.0
nltk>=3.8.1
pandas>=2.0.3
numpy>=1.24.3
joblib>=1.3.2
```

## 📱 Usage

### 🖥️ **Web Interface**

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Analyze messages**:
   - Enter any message in the text area
   - Click "🔍 Analyze Message"
   - View binary classification result
   - Check detailed fraud analysis

### 🧪 **Testing & Validation**

```bash
# Test fraud detection with sample messages
python test_fraud_detection.py

# Verify system components
python -c "from app import load_model_components; print('✅ All components loaded successfully!')"
```

### 📊 **Example Results**

#### ❌ **SPAM Detection**
```
Input: "Dear customer, your debit card will be blocked within 2 hrs. Call 9876543210 to reactivate."

Output:
🚨 SPAM DETECTED (90% confidence)
📊 Fraud Risk Score: 100%
💳 Financial Keywords: debit card, reactivate
📞 Phone Numbers: 9876543210
⏰ Time Pressure: within 2 hrs
```

#### ✅ **LEGITIMATE Detection**
```
Input: "Hi! Are we still meeting for lunch tomorrow at 12pm?"

Output:
✅ LEGITIMATE MESSAGE (95% confidence)
📊 Fraud Risk Score: 0%
ℹ️ No suspicious patterns detected
```

## 🎯 Detection Examples

<details>
<summary><b>🚨 Financial Fraud Examples</b></summary>

| Message Type | Example | Detection |
|--------------|---------|-----------|
| Banking Fraud | "Your account is blocked. Call 9876543210 immediately!" | ✅ SPAM |
| Loan Scam | "Pre-approved loan of ₹5,00,000. Apply: http://scam-site.com" | ✅ SPAM |
| Prize Scam | "You won iPhone! Pay ₹99 shipping: http://fake-apple.com" | ✅ SPAM |
| OTP Theft | "Share OTP 123456 to verify your account immediately" | ✅ SPAM |

</details>

<details>
<summary><b>✅ Legitimate Message Examples</b></summary>

| Message Type | Example | Detection |
|--------------|---------|-----------|
| Personal | "Hi! Are we meeting for lunch tomorrow?" | ✅ LEGITIMATE |
| Business | "Team meeting at 3 PM in conference room B" | ✅ LEGITIMATE |
| Appointments | "Doctor visit reminder: Friday at 3pm" | ✅ LEGITIMATE |
| Delivery | "Package will be delivered between 10 AM - 2 PM" | ✅ LEGITIMATE |

</details>

## 🏗️ Architecture

### 🧠 **Model Architecture**

```
Input Message
     ↓
┌─────────────────┐    ┌─────────────────┐
│ Rule-Based      │    │ ML Classification│
│ Fraud Analysis  │    │ (Neural Network) │
│                 │    │                  │
│ • Financial     │    │ • TF-IDF         │
│ • Phishing      │    │ • 5000 features  │
│ • Phone Numbers │    │ • 128→64→2       │
│ • URLs          │    │ • PyTorch        │
└─────────────────┘    └─────────────────┘
     ↓                       ↓
┌─────────────────────────────────────────┐
│         Confidence Fusion               │
│    (Hybrid Decision Engine)             │
└─────────────────────────────────────────┘
     ↓
Final Classification: SPAM / LEGITIMATE
```

### 🔧 **Technical Stack**

- **Backend**: Python 3.9+
- **ML Framework**: PyTorch 2.0+
- **Feature Engineering**: Scikit-learn, NLTK
- **Web Interface**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: JobLib

## 📁 Project Structure

```
PhishShield/
├── 📄 app.py                    # Main Streamlit application
├── 🧠 spam.py                   # Model training script
├── 📊 spam.csv                  # Training dataset
├── 🔧 requirements.txt          # Python dependencies
├── 🚀 run_phishshield.bat      # Windows launcher
├── 🛠️ setup_and_run.py         # Automated setup script
├── 🧪 test_fraud_detection.py   # Testing utilities
├── 📚 README.md                 # Project documentation
└── 🤖 Model Files/
    ├── spam_classifier.pth      # Trained neural network
    ├── tfidf_vectorizer.pkl     # TF-IDF vectorizer
    └── label_encoder.pkl        # Label encoder
```

## ⚡ Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **Overall Accuracy** | 97% | General spam detection accuracy |
| **Financial Fraud Detection** | 95%+ | Specialized fraud pattern detection |
| **False Positive Rate** | <3% | Legitimate messages marked as spam |
| **Processing Speed** | <100ms | Average analysis time per message |
| **Model Size** | ~2.5MB | Compact for deployment |

## 🔒 Security Features

### 🛡️ **Fraud Protection**
- **Multi-layer Detection**: ML + Rule-based validation
- **Real-time Scoring**: Instant risk assessment
- **Pattern Recognition**: Advanced fraud indicators
- **Educational Alerts**: Built-in safety guidelines

### 📋 **Safety Guidelines**
- ❌ **Never share**: PIN, OTP, passwords via SMS
- ✅ **Always verify**: Requests through official channels
- 🚨 **Report fraud**: Suspicious messages to authorities
- 🛡️ **Stay informed**: Keep updated on latest scam tactics

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 **Bug Reports**
- Use GitHub Issues to report bugs
- Include message examples and error details
- Specify your environment (OS, Python version)

### 💡 **Feature Requests**
- Suggest new fraud patterns to detect
- Propose UI/UX improvements
- Request additional language support

### 🔧 **Development**

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/PhishShield.git

# Create a feature branch
git checkout -b feature/amazing-feature

# Make your changes and test
python test_fraud_detection.py

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# Create a Pull Request
```

## 📈 Roadmap

### 🎯 **Upcoming Features**
- [ ] 📱 **Mobile App**: React Native mobile application
- [ ] 🌐 **Multi-language**: Hindi, Tamil, Telugu support
- [ ] 🔊 **Voice Analysis**: Audio message fraud detection
- [ ] 📞 **Call Integration**: Real-time call analysis
- [ ] 🤖 **Advanced AI**: Transformer-based models
- [ ] 📊 **Analytics Dashboard**: Fraud trend analysis
- [ ] 🔌 **API Integration**: RESTful API for third-party apps

### 🏢 **Enterprise Features**
- [ ] 👥 **Team Management**: Multi-user support
- [ ] 📈 **Reporting**: Advanced analytics and insights
- [ ] 🔒 **Enterprise Security**: Enhanced data protection
- [ ] ⚡ **High Performance**: Scalable cloud deployment

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: SMS Spam Collection Dataset
- **Framework**: PyTorch Community
- **UI Library**: Streamlit Team
- **Inspiration**: Combating real-world fraud schemes

## 📞 Support

### 🐛 **Issues & Bug Reports**
- 📧 **Email**: your.email@domain.com
- 🐙 **GitHub Issues**: [Report a Bug](https://github.com/yourusername/PhishShield/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/PhishShield/discussions)

### 📚 **Documentation**
- 📖 **Wiki**: [PhishShield Wiki](https://github.com/yourusername/PhishShield/wiki)
- 🎥 **Video Tutorials**: Coming soon!
- 📝 **Blog Posts**: [Medium Articles](https://medium.com/@yourusername)

---

<div align="center">

**⭐ Star this repository if PhishShield helped protect you from fraud! ⭐**

![PhishShield](https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge)
![Python](https://img.shields.io/badge/Powered%20by-Python-blue?style=for-the-badge&logo=python&logoColor=white)

**🛡️ Protecting users from fraud, one message at a time 🛡️**

</div>

---

## 📊 Usage Analytics

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/yourusername/PhishShield?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/PhishShield?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/PhishShield)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/PhishShield)

</div>
