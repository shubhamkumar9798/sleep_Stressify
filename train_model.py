import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the sleep data from CSV
data = pd.read_csv('sleep_data.csv')

# Define features (X) and target variable (y)
X = data.drop(columns=['stress_level'])
y = data['stress_level']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the trained model to model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as 'model.pkl'")
