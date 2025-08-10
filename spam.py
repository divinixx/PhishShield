import nltk
nltk.download('stopwords')

import numpy as np
import pandas as pd
import re
import string
import joblib

from textblob import TextBlob
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load Dataset with Correct Encoding
file_path = 'spam.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Step 2: Select Relevant Columns
df = df[['v1', 'v2']]
df.columns = ['class', 'sms']

# Step 3: Handle Missing Values and Remove Duplicates
df = df.dropna()
df = df.drop_duplicates()

# Step 4: Visualize Class Distribution
sns.countplot(x='class', data=df, palette='viridis')
plt.title("Class Distribution (Spam vs Non-Spam)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Step 5: Encode Labels
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Step 6: Preprocess Text Data
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['sms'] = df['sms'].apply(preprocess)

# Step 7: Feature Extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['sms']).toarray()
y = df['class']

# Step 8: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Prepare Data for PyTorch
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Step 10: Define Neural Network
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


# Step 11: Train Model
model = SpamClassifier(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


# Step 12: Evaluate Model
model.eval()
y_pred_nn = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred_nn.extend(preds.tolist())

nn_accuracy = accuracy_score(y_test, y_pred_nn)
print(f"Model Accuracy: {nn_accuracy:.4f}")

# Step 13: Save Model and Preprocessing Components
print("Saving model and preprocessing components...")

# Save the trained model
torch.save(model.state_dict(), 'spam_classifier.pth')

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model and components saved successfully!")
print(f"Files saved:")
print(f"- spam_classifier.pth (PyTorch model)")
print(f"- tfidf_vectorizer.pkl (TF-IDF vectorizer)")
print(f"- label_encoder.pkl (Label encoder)")

# Create a simple test function for the Streamlit app
def preprocess_text(text):
    """Preprocess text for prediction"""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text
