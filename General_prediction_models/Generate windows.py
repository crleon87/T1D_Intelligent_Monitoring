import pickle
import numpy as np

# Define the file path (Replace with your filepath)
file_path_train = "data/measurements_by_patient_train.pkl"
file_path_test = "data/measurements_by_patient_test.pkl"
file_path_val = "data/measurements_by_patient_val.pkl"

# Open files
with open(file_path_train, 'rb') as file:
    train = pickle.load(file)

with open(file_path_test, 'rb') as file:
    test = pickle.load(file)

with open(file_path_val, 'rb') as file:
    val = pickle.load(file)

### Generating windows Functions ####

# Auxiliary function
def get_windows_one_step_walk_forward(data, lookback_samples, pred_samples):
    """

      :param data:
      :param lookback_samples: Number of samples used to predict
      :param pred_samples: Prediction window (normally 30 or 60 minutes)
      :param backup: Save the progress each 100 patients
      :return: Get windows without missing values from one step sliding windows
      """
    x_per_patient = []
    y_per_patient = []

    for data_patient in data:
        # Creating the one step sliding window
        x = np.lib.stride_tricks.sliding_window_view(data_patient[:-pred_samples], lookback_samples)
        y = np.lib.stride_tricks.sliding_window_view(data_patient[lookback_samples:], pred_samples)

        # Removing rows with missing values
        nan_rows_x = np.isnan(x).any(axis=1)
        nan_rows_y = np.isnan(y).any(axis=1)
        x = x[~(nan_rows_x | nan_rows_y)]
        y = y[~(nan_rows_x | nan_rows_y)]

        x_per_patient.append(x)
        y_per_patient.append(y)

    x_result = np.concatenate(x_per_patient)
    y_result = np.concatenate(y_per_patient)

    return x_result, y_result


# Create windows
history_length = 8  # 120 min
horizons = [2, 4, 6, 8, 12, 16]

for current_horizon in horizons:
    x_train, y_train = get_windows_one_step_walk_forward(train.values(), history_length, current_horizon)
    x_test, y_test = get_windows_one_step_walk_forward(test.values(), history_length, current_horizon)
    x_val, y_val = get_windows_one_step_walk_forward(val.values(), history_length, current_horizon)

    # NOTE: Replace with your filepath
    np.save('windows/x_train_windows_horizon_{0}.npy'.format(current_horizon), x_train)
    np.save('windows/y_train_windows_horizon_{0}.npy'.format(current_horizon), y_train)
    np.save('windows/x_test_windows_horizon_{0}.npy'.format(current_horizon), x_test)
    np.save('windows/y_test_windows_horizon_{0}.npy'.format(current_horizon), y_test)
    np.save('windows/x_val_windows_horizon_{0}.npy'.format(current_horizon), x_val)
    np.save('windows/y_val_windows_horizon_{0}.npy'.format(current_horizon), y_val)