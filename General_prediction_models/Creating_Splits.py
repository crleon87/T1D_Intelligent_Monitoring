import pandas as pd
import numpy as np

df_glucose = pd.read_csv("Preprocessed_data.csv")

def get_patient_ids(df):
    """

    :param df_glucose:
    :return: List of patient ids
    """
    return df['Patient_ID'].unique()

patients = get_patient_ids(df=df_glucose)
patients_size = patients.shape[0]

patients_for_testing = int(patients_size * 0.2)
patients_for_validation = int(patients_size * 0.1)

# Stratified (patients are ordered by number of measurements) train/validation/test datasets
# For each ten patients: 7 for training, 2 for testing, 1 for validation

import random

remaining_patients = patients_size % 10

# Create a list to store the selected numbers
train_indexes =      []
test_indexes =       []
validation_indexes = []

# Iterate over the range of numbers from 0 to patients_size
for num in range(0, patients_size - remaining_patients, 10):
    group = list(range(num, num+10))

    # Testing
    random_nums_testing = random.sample(group, 2)
    test_indexes.append(random_nums_testing[0])
    test_indexes.append(random_nums_testing[1])

    group.remove(random_nums_testing[0])
    group.remove(random_nums_testing[1])

    # Validation
    random_num_validation = random.choice(group)
    validation_indexes.append(random_num_validation)

    group.remove(random_num_validation)

    # Training
    train_indexes.extend(group)

# Remaining patients
remaining_patients_indexes = list(range(patients_size - remaining_patients, patients_size))
if remaining_patients < 7:
  train_indexes += remaining_patients_indexes
else:
  train_indexes += remaining_patients_indexes[:7]
  test_indexes += remaining_patients_indexes[7:]

assert not set(train_indexes) & set(test_indexes) & set(validation_indexes)
assert len(test_indexes) == patients_for_testing and len(validation_indexes) == patients_for_validation
assert len(train_indexes) == patients_size - patients_for_testing - patients_for_validation

# Counting only not null measurements
amount_of_data_per_patient_not_null = df_glucose[df_glucose['Measurement'].notnull()].groupby('Patient_ID').size().reset_index(name='Measurement_count_not_null')

amount_of_data_per_patient_not_null_ordered = amount_of_data_per_patient_not_null.sort_values('Measurement_count_not_null', ascending=False)

patients_ordered_by_size_list = amount_of_data_per_patient_not_null_ordered['Patient_ID'].to_list()

train_patients = [patients_ordered_by_size_list[index] for index in train_indexes]
test_patients = [patients_ordered_by_size_list[index] for index in test_indexes]
validation_patients = [patients_ordered_by_size_list[index] for index in validation_indexes]

train_patients = np.array(train_patients)
test_patients = np.array(test_patients)
validation_patients = np.array(validation_patients)

# Saving the random patients to use always the same of them
np.save('train_patients.npy', train_patients)
np.save('validation_patients.npy', validation_patients)
np.save('test_patients.npy', test_patients)

assert not set(train_patients) & set(test_patients) & set(validation_patients)
assert len(test_patients) == patients_for_testing and len(validation_patients) == patients_for_validation
assert len(train_patients) == patients_size - patients_for_testing - patients_for_validation