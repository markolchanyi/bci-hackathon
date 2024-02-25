import numpy as np
import pywt
from pylsl import StreamInlet, resolve_stream
from joblib import load
from duckietown.sdk.robots.duckiebot import DB21J
import time

def extract_wavelet_features(signal, wavelet_name, level):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    features = []
    for coeff in coeffs:
        features.extend([np.mean(coeff), np.std(coeff)])
    return features

# Load the trained SVM model and scaler
svm_model_filename = '/Users/markolchanyi/Desktop/svm_model.joblib'
scaler_filename = '/Users/markolchanyi/Desktop/scaler.joblib'
svm_clf_loaded = load(svm_model_filename)
scaler = load(scaler_filename)

# Initialize the Duckiebot
SIMULATED_ROBOT_NAME = "perseverance"  # Change if using a real robot
robot = DB21J(SIMULATED_ROBOT_NAME, simulated=False)  # Set simulated=False for a real robot

# Start the motors
robot.motors.start()

print("Looking for an EEG stream...")
streams = resolve_stream('source_id', '102801-0077 500')
inlet = StreamInlet(streams[0])

sample_count = 0
max_samples = 60000  # Adjust as needed

sampling_rate = 500  # Hz
window_size = int(0.3 * sampling_rate)  # 0.2 seconds window

while sample_count < max_samples:
    eeg_chunk = np.zeros((window_size, 7))  # Assuming 7 channels
    for i in range(window_size):
        sample, timestamp = inlet.pull_sample()
        eeg_chunk[i, :] = sample[0:7]  # Get the first 7 channel values

    # Process the EEG chunk
    features = []
    for channel_idx in range(7):
        signal = eeg_chunk[:, channel_idx]
        features.extend(extract_wavelet_features(signal, 'db4', 3))

    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = svm_clf_loaded.predict(features_scaled)

    # Control the Duckiebot based on the prediction
    if prediction == 2:  # Turn left
        speeds = [0.5,0.1]
        robot.motors.publish(tuple(speeds))
    elif prediction == 1:  # Turn right
        speeds = [0.1,0.5]
        robot.motors.publish(tuple(speeds))
    else:  # Go straight
        speeds = [0.6,0.6]
        robot.motors.publish(tuple(speeds))

    print(f"Prediction for chunk starting at sample {sample_count}: {prediction}")
    sample_count += window_size  # Update the sample count by the chunk size
    #time.sleep(0.05)  # Adjust as needed for your robot's speed

# Stop the motors
robot.motors.stop()
