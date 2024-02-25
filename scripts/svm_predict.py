import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.preprocessing import StandardScaler
import pywt

# Define the function to extract features using wavelet transform
def extract_wavelet_features(signal, wavelet_name, level):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff)])
    return features

# Load the new dataset from the Excel file
new_file_path = '/Users/markolchanyi/Desktop/training_data/mark_tongue_left_right_60k_samples_session_3.xlsx'  # Update this path to your new Excel file
df_new = pd.read_excel(new_file_path)

# Parameters for feature extraction
sampling_rate = 500  # Hz
window_size = int(0.2 * sampling_rate)  # 0.2 seconds window
wavelet = 'db4'  # Daubechies order 4, chosen for computational efficiency
max_level = 3  # Maximum decomposition level

# Process the new data - create windows and extract features
X_new = []
for i in range(0, df_new.shape[0] - window_size, window_size):
    window = df_new.iloc[i:i + window_size]
    features = []
    for channel in range(1, 8):  # Assuming Channels 1 through 7 as before
        signal = window[f'Channel_{channel}']
        features.extend(extract_wavelet_features(signal, wavelet, max_level))
    X_new.append(features)

X_new = np.array(X_new)

# Load the scaler and the trained SVM model
scaler_filename = '/Users/markolchanyi/Desktop/scaler.joblib'
model_filename = '/Users/markolchanyi/Desktop/svm_model.joblib'
scaler = load(scaler_filename)
svm_clf_loaded = load(model_filename)

# Standardize the features of the new dataset using the loaded scaler
X_new_scaled = scaler.transform(X_new)

# Predict with the loaded SVM model
y_new_pred = svm_clf_loaded.predict(X_new_scaled)

# Plotting actual KeyPress column
plt.figure(figsize=(12, 6))
plt.scatter(np.arange(len(df_new['KeyPress'])), df_new['KeyPress'], color='red', label='Actual KeyPress', s=10)
plt.xlabel('Sample Index')
plt.ylabel('KeyPress Value')
plt.title('Actual KeyPress Events')
plt.yticks([0, 1, 2])  # Assuming KeyPress values are in {0, 1, 2}
#plt.legend()
plt.show()

# Plotting SVM predictions
plt.figure(figsize=(12, 6))
plt.scatter(np.arange(len(y_new_pred)), y_new_pred, color='blue', label='SVM Prediction', s=10)
plt.xlabel('Sample Index')
plt.ylabel('Predicted Value')
plt.title('SVM Predictions')
plt.yticks([0, 1, 2])  # Assuming predictions are in {0, 1, 2}
#plt.legend()
plt.show()
