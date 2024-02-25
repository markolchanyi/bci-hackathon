import pandas as pd
from pylsl import StreamInlet, resolve_stream
from pynput import keyboard


key_press_state = 0

# The event listener for key press
def on_press(key):
    global key_press_state
    try:
        if key == keyboard.Key.left: # 2 is LEFT!!!
            key_press_state = 1
        elif key == keyboard.Key.right:
            key_press_state = 2 # 1 is RIGHT!!!
    except AttributeError:
        pass

# The event listener for key release
def on_release(key):
    global key_press_state
    # Reset the key press state when any key is released
    key_press_state = 0

# Start listening to the keyboard in a non-blocking fashion
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def main():
    print("Looking for an EEG stream...")
    streams = resolve_stream('source_id', '102801-0077 500')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    # Prepare a list to store the data
    eeg_data = []
    sample_count = 0
    max_samples = 60000

    while sample_count < max_samples:
        global key_press_state
        # get a new sample
        sample, timestamp = inlet.pull_sample()
        # Append timestamp, sample data, and key press state to the list
        eeg_data.append([timestamp] + sample + [key_press_state])
        sample_count += 1
        print(f"Collected {sample_count} samples.")

    # Convert the list to a DataFrame
    eeg_df = pd.DataFrame(eeg_data, columns=["Timestamp"] + [f"Channel_{i}" for i in range(1, 12)] + ["KeyPress"])

    # Save the DataFrame to an Excel file
    excel_filename = "/Users/markolchanyi/Desktop/david_cheeks_left_right_60k_samples_session_1.xlsx"
    eeg_df.to_excel(excel_filename, index=False)
    print(f"EEG data saved to {excel_filename}")

if __name__ == '__main__':
    main()
