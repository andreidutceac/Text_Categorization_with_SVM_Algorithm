import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

# Sample dataset
texts = [
    "The local team won the championship after a thrilling game.",
    "Olympic athletes prepare for the biggest challenge of their careers.",
    "The election campaign heats up as candidates debate key issues.",
    "New policies introduced aiming to boost economic growth.",
    "Tech giants unveil latest innovations at the annual conference.",
    "Breakthrough in renewable energy technology promises a greener future.",
    "Researchers discover new treatment for a rare disease.",
    "Global health organizations warn about rising levels of obesity."
]

categories = [
    "Sports",
    "Sports",
    "Politics",
    "Politics",
    "Technology",
    "Technology",
    "Health",
    "Health"
]


# Data Preprocessing with TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Splitting dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, categories, test_size=0.3, random_state=42)

# SVM Model
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

# Function to predict category of new text
def predict_category(text):
    text_features = vectorizer.transform([text])
    return clf.predict(text_features)

# Assuming the rest of the code is already executed and the model is trained

# New texts for prediction
new_text1 = "Groundbreaking software revolutionizes the way we interact with our smart devices."
new_text2 = "World Health Organization releases new guidelines for managing pandemic-related stress."

# Predicting categories
print(f"Predicted category for Text 1: {predict_category(new_text1)[0]}")
print(f"Predicted category for Text 2: {predict_category(new_text2)[0]}")
