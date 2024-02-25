import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pywt
import time
import random
from joblib import dump

# List of file paths
file_paths = [
    '/Users/markolchanyi/Desktop/training_data/mark_tongue_left_right_60k_samples_session_1.xlsx',
    '/Users/markolchanyi/Desktop/training_data/mark_tongue_left_right_60k_samples_session_2.xlsx'
]

# Load and concatenate the datasets
df_list = [pd.read_excel(file) for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

# Parameters
sampling_rate = 500  # Hz
window_size = int(0.2 * sampling_rate)  # 0.2 seconds window
max_shift = int(0.05 * sampling_rate)  # Maximum shift of 0.05 seconds
wavelet = 'db4'  # Daubechies order 4, chosen for computational efficiency

def extract_wavelet_features(signal, wavelet_name, level):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff)])
    return features

# Process the data - create windows and extract features with random shifts
X = []
y = []

for i in range(0, df.shape[0] - window_size, window_size):
    shift = random.randint(-max_shift, max_shift)  # Random shift within the max_shift range
    start = max(0, i + shift)  # Ensure the start index is not negative
    end = min(df.shape[0], start + window_size)  # Ensure the end index does not exceed the DataFrame length
    window = df.iloc[start:end]
    if window.shape[0] < window_size:  # Skip windows that are smaller than the desired window size due to shifting
        continue
    features = []
    for channel in range(1, 8):  # Channels 1 through 7
        signal = window[f'Channel_{channel}']
        features.extend(extract_wavelet_features(signal, wavelet, level=3))
    X.append(features)
    y.append(window['KeyPress'].mode()[0])

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train an SVM classifier
svm_clf = SVC(kernel='linear')
print("Training SVM...")
svm_clf.fit(X_train_scaled, y_train)
print("Training done!")

# Evaluate the classifier
print("Predicting...")
start_time = time.time()
y_pred = svm_clf.predict(X_test_scaled)
prediction_time = time.time() - start_time
print(f"Prediction completed in {prediction_time:.2f} seconds.")

report = classification_report(y_test, y_pred)
print(report)


print("saving SVM...")

model_filename = '/Users/markolchanyi/Desktop/svm_model.joblib'  # Specify your save path here
dump(svm_clf, model_filename)
print(f"Trained SVM model saved to {model_filename}")


# Save the scaler
scaler_filename = '/Users/markolchanyi/Desktop/scaler.joblib'  # Path to save the scaler
dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")
