import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # Changed from SVC to RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pywt
import time

# List of file paths
file_paths = [
    '/Users/markolchanyi/Desktop/training_data/mark_cheeks_left_right_60k_samples_session_1.xlsx',
    '/Users/markolchanyi/Desktop/training_data/mark_cheeks_left_right_60k_samples_session_2.xlsx',
    '/Users/markolchanyi/Desktop/training_data/mark_cheeks_left_right_60k_samples_session_3.xlsx'
]

# Load and concatenate the datasets
df_list = [pd.read_excel(file) for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

# Parameters
sampling_rate = 500  # Hz
window_size = int(0.2 * sampling_rate)  # 0.2 seconds window
wavelet = 'db4'  # Daubechies order 4, chosen for computational efficiency

def extract_wavelet_features(signal, wavelet_name, level):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff)])
    return features

# Process the data - create windows and extract features
X = []
y = []

for i in range(0, df.shape[0] - window_size, window_size):
    window = df.iloc[i:i + window_size]
    features = []
    for channel in range(1, 8):  # Channels 1 through 7
        signal = window[f'Channel_{channel}']
        features.extend(extract_wavelet_features(signal, wavelet, level=4))
    X.append(features)
    y.append(window['KeyPress'].mode()[0])

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardize the features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest classifier
rf_clf = RandomForestClassifier()  # Using default parameters for simplicity
print("Training Random Forest...")
rf_clf.fit(X_train_scaled, y_train)
print("Training done!")

# Evaluate the classifier
print("Predicting...")
start_time = time.time()
y_pred = rf_clf.predict(X_test_scaled)
prediction_time = time.time() - start_time
print(f"Prediction completed in {prediction_time:.2f} seconds.")

report = classification_report(y_test, y_pred)
print(report)
