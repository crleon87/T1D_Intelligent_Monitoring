import pandas as pd
import numpy as np
import datetime

def get_patient_ids(df):
    """

    :param df_glucose:
    :return: List of patient ids
    """
    return df['Patient_ID'].unique()

df_glucose = pd.read_csv("Glucose_measurements.csv")
df_glucose["Timestamp"] = pd.to_datetime(df_glucose["Measurement_date"] + ' ' + df_glucose["Measurement_time"])

def resample(df, patients = None, checkpoint = False):
  """
    :param df: Dataframe of BG measurements
    :param patients: Subset of patients. If None, all T1DiabetesGranada patients
    :param checkpoint: If true, data is saved during the process
    :return: Array of resampled datasets per patient
  """

  def get_closest(all_dates_series, expected_date):
    """

    :return: The closest measurement to the expected date, Nan if there is no
    measurements in the range
    """
    allowed_delay = 7
    for delay in range(1, allowed_delay + 1):

      past_value   = expected_date - pd.DateOffset(minutes=delay)
      if not np.isnan(all_dates_series[past_value]):
        return all_dates_series[past_value]

      future_value = expected_date + pd.DateOffset(minutes=delay)
      if not np.isnan(all_dates_series[future_value]):
        return all_dates_series[future_value]

    return None

  if checkpoint:
    data_backup = 50
    iteration = 0

  resampled_data_per_patient = []

  if not patients:
    patients = get_patient_ids(df)
  for patient in patients:
    print(f'Patient {patient}... Iteration: {iteration}')

    df_glucose_patient = df[df['Patient_ID'] == patient]

    start_date = df_glucose_patient['Timestamp'].iloc[0]
    end_date   = df_glucose_patient['Timestamp'].iloc[-1]

    all_dates = pd.date_range(start=start_date, end=end_date, freq='T')
    all_dates_series = pd.Series(dtype='float64', index=all_dates)
    all_dates_series[df_glucose_patient['Timestamp']] = df_glucose_patient['Measurement'].values

    expected_dates = pd.date_range(start=start_date, end=end_date, freq='15T')

    for expected_date in expected_dates:
      if np.isnan(all_dates_series[expected_date]):
        all_dates_series[expected_date] = get_closest(all_dates_series, expected_date)

    resampled_data_per_patient.append(all_dates_series[expected_dates])

    iteration = iteration + 1
    if checkpoint and iteration == data_backup:
      print("Saving...")
      np.save(f'Preprocessed_data_{iteration}.npy', np.asarray(resampled_data_per_patient, dtype=object))
      iteration = 0

  return resampled_data_per_patient

complete_resampled_data = resample(df_glucose, checkpoint=True)
np.save('Resampled_data.npy', complete_resampled_data)

## Removing patients below a treshold (30 days of measurements)
patient_IDs = get_patient_ids(df_glucose)

data_ = []
for patient_ID, series in zip(patient_IDs, resampled_data):
    # Iterate over the values in the series and create tuples with patient ID and measurement
    for measurement in series:
        data_.append((patient_ID, measurement))

df_glucose_resampled = pd.DataFrame(data_, columns=['Patient_ID', 'Measurement'])

# Creating a df with the amount of data per each patient

amount_of_data_per_patient = pd.DataFrame({'Patient_ID': patient_IDs, 'Measurement_count': [len(measurements) for measurements in resampled_data],
                                           'Measurement_no_missing_values_count': [measurements.count() for measurements in resampled_data]})

minutes_in_hour = 60
hours_in_day = 24
samples_separation = 15

amount_of_data_per_patient['Measurement_no_missing_values_count_days'] = amount_of_data_per_patient['Measurement_no_missing_values_count'] * samples_separation / (minutes_in_hour * hours_in_day)

patients_below_30_days = amount_of_data_per_patient.loc[amount_of_data_per_patient['Measurement_no_missing_values_count_days'] < 30, 'Patient_ID'].tolist()

print(f"Number of deleted patients: {len(patients_below_30_days)}")

patients_mask = ~df_glucose_resampled['Patient_ID'].isin(patients_below_30_days)
preprocessed_data = df_glucose_resampled[patients_mask]

preprocessed_data.to_csv('Preprocessed_data.csv')





