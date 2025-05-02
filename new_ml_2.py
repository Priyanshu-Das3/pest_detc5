import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Create synthetic training dataset
num_samples = 1000
np.random.seed(42)

df = pd.read_csv("C:/Users/MEM/Downloads/pest_sensor_dataset_selected.csv")

# Drop non-numeric column
df = df.drop(columns=["Timestamp"], errors="ignore")

# Then proceed to split features/labels and train your model
X = df.drop("Pest_Detected", axis=1)
y = df["Pest_Detected"]

df = pd.DataFrame({
    "Moisture_Sensor": np.random.uniform(0, 100, num_samples),
    "Humidity": np.random.normal(60, 10, num_samples).clip(20, 100),
    "Temperature": np.random.normal(30, 5, num_samples).clip(10, 50),
    "Infrared_Sensor": np.random.uniform(0.1, 1.0, num_samples),
    "Motion_Sensor": np.random.randint(0, 2, num_samples),
    "Vibration_Sensor": np.random.uniform(0, 1, num_samples),
    "Gas_Sensor": np.random.normal(200, 50, num_samples).clip(0, 1023),
    "Pest_Detected": np.random.randint(0, 2, num_samples)
})

# Separate features and target
X = df.drop("Pest_Detected", axis=1)
y = df["Pest_Detected"]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Create custom test dataset
test_df = pd.DataFrame({
    "Moisture_Sensor": [45, 70, 12, 88, 33],
    "Humidity": [55, 72, 40, 63, 80],
    "Temperature": [28, 33, 26, 37, 31],
    "Infrared_Sensor": [0.5, 0.7, 0.2, 0.9, 0.4],
    "Motion_Sensor": [0, 1, 1, 0, 1],
    "Vibration_Sensor": [0.3, 0.8, 0.1, 0.6, 0.2],
    "Gas_Sensor": [180, 250, 120, 300, 210]
})

# Predict results
predictions = model.predict(test_df)

# Print readable predictions
print("\n🔍 Prediction Results:")
for i, pred in enumerate(predictions):
    status = "✅ Pest Detected" if pred == 1 else "🌿 No Pest Detected"
    print(f"Sample {i+1}: {status}")

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

import joblib

# Save trained model to a .pkl file
joblib.dump(model, "pest_detector_model.pkl")
