import pandas as pd
import numpy as np
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load datasets
df1 = pd.read_csv("datasets/Symptom2Disease.csv")
df2 = pd.read_csv("datasets/clean.data.csv")

# Normalize column names
df1.columns = df1.columns.str.lower()
df2.columns = df2.columns.str.lower()

# Merge text columns
df1 = df1[['text', 'label']]
df2 = df2[['text', 'label']]

df = pd.concat([df1, df2], ignore_index=True)

# Map diseases to urgency levels
def map_urgency(disease):
    disease = disease.lower()
    if "heart" in disease or "stroke" in disease:
        return 2
    elif "migraine" in disease or "infection" in disease:
        return 1
    else:
        return 0

df["urgency"] = df["label"].apply(map_urgency)

X = df["text"]
y = df["urgency"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/triage_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))

print("Model training complete.")