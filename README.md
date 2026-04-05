# PHM_AI-29-36
A hybrid AI framework integrating image-based CNN detection and time-series LSTM prediction for proactive food spoilage monitoring and risk control.
!pip install kagglehub

import kagglehub
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
print("\n========== DATASET LOADING ==========\n")

path = kagglehub.dataset_download("swoyam2609/fresh-and-stale-classification")
train_path = os.path.join(path, "dataset", "Train")

print("Folders:", os.listdir(train_path))
label_map = {}

for folder in os.listdir(train_path):
    if "fresh" in folder:
        label_map[folder] = 0
    elif "rotten" in folder:
        label_map[folder] = 1

print("Label Map:", label_map)
images, labels = [], []

limit = 500
count = 0

for folder in os.listdir(train_path):
    folder_path = os.path.join(train_path, folder)

    for file in os.listdir(folder_path):
        if count >= limit:
            break

        img = cv2.imread(os.path.join(folder_path, file))
        if img is None:
            continue

        img = cv2.resize(img, (224, 224)) / 255.0

        images.append(img)
        labels.append(label_map[folder])
        count += 1

    if count >= limit:
        break

images = np.array(images)
labels = np.array(labels)

print("Image dataset shape:", images.shape)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
print("\n========== CNN TRAINING ==========\n")

cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

cnn_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
print("\n========== CNN OUTPUT ==========\n")

cnn_pred = cnn_model.predict(X_test)

plt.figure()
plt.imshow(X_test[0])
plt.title(f"Predicted: {cnn_pred[0]} | Actual: {y_test[0]}")
plt.axis('off')
plt.show()

loss, acc = cnn_model.evaluate(X_test, y_test)
print("CNN Accuracy:", acc)
print("\n========== SENSOR DATA ==========\n")

np.random.seed(42)

n = 500
time = np.arange(n)

temp = np.random.normal(6, 2, n)
humidity = np.random.normal(70, 8, n)
gas = np.random.normal(0.3, 0.1, n)

trend = time * 0.002

spoilage_score = (
    0.4 * temp +
    0.3 * humidity +
    0.3 * gas +
    trend
)

spoilage_score = (spoilage_score - spoilage_score.min()) / (spoilage_score.max() - spoilage_score.min())

df = pd.DataFrame({
    "temp": temp,
    "humidity": humidity,
    "gas": gas,
    "spoilage_score": spoilage_score
})

print(df.head())
print("Dataset shape:", df.shape)
features = df[["temp", "humidity", "gas"]].values
target = df["spoilage_score"].values

def create_sequences(features, target, steps=10):
    X, y = [], []
    for i in range(len(features) - steps):
        X.append(features[i:i+steps])
        y.append(target[i+steps])
    return np.array(X), np.array(y)

X, y = create_sequences(features, target)
print("\n========== LSTM TRAINING ==========\n")

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.fit(X, y, epochs=10, batch_size=16)
print("\n========== LSTM OUTPUT ==========\n")

lstm_pred = lstm_model.predict(X)

plt.figure()
plt.plot(y[:100], label="Actual")
plt.plot(lstm_pred[:100], label="Predicted")

plt.legend()
plt.title("LSTM Prediction Graph")
plt.show()
print("\n========== FINAL RISK OUTPUT ==========\n")

cnn_pred = cnn_model.predict(X_test)
cnn_score = np.argmax(cnn_pred, axis=1).astype(float)

lstm_pred = lstm_model.predict(X).flatten().astype(float)

min_len = min(len(cnn_score), len(lstm_pred))

cnn_score = cnn_score[:min_len]
lstm_pred = lstm_pred[:min_len]

final_risk = (0.6 * cnn_score) + (0.4 * lstm_pred)

final_risk = (final_risk - final_risk.min()) / (final_risk.max() - final_risk.min() + 1e-7)

def classify(r):
    if r < 0.3:
        return "Low Risk"
    elif r < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

risk_labels = np.array([classify(r) for r in final_risk])

print("Sample Output:", risk_labels[:10])
print("Total:", len(risk_labels))
print("Low:", np.sum(risk_labels=="Low Risk"))
print("Medium:", np.sum(risk_labels=="Medium Risk"))
print("High:", np.sum(risk_labels=="High Risk"))
