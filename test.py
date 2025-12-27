import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("diabetes_prediction_dataset.csv")
print("Dataset loaded!\n")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

le = LabelEncoder()
df["smoking_history"] = le.fit_transform(df["smoking_history"])

X = df.drop(["diabetes", "gender"], axis=1)   # gender removed
y = df["diabetes"]


print("\nFinal training features:", X.columns.tolist())
print("Target shape:", y.shape)