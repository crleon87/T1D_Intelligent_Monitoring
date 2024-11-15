
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import tensorflow as tf
import random
import os
import sys
import pandas as pd
from keras.models import Model
from keras.layers import Dense, LSTM, Input
import time
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten

import time

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###################### CONFIG EXPERIMENT #############################
batch_size = 4096
patience = 50
max_epoch = 500
windows_path = '../windows/'

history_length = 8  # 2h

horizons = [2, 4, 6, 8, 12, 16]
models = ['Linear', 'LSTM', 'CNN']
repeated_examples = False

###################### END CONFIG EXPERIMENT #############################

import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'

# List of folder names
folders = ["hist", "models", "plots", "test_results"]

# Create each folder if it does not exist
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        for f in self.files:
            if f not in (sys.stdout, sys.stderr):
                f.close()


# Open a log file to write the results
log_file = open("script_log.txt", "a", encoding="utf-8")

# Redirect stdout and stderr
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

try:

    import pandas as pd

    def read_windows_no_duplicated(horizon):
        file_x_train = f"{windows_path}x_train_windows_horizon_{horizon}.npy"
        file_y_train = f"{windows_path}y_train_windows_horizon_{horizon}.npy"
        file_x_val = f"{windows_path}x_val_windows_horizon_{horizon}.npy"
        file_y_val = f"{windows_path}y_val_windows_horizon_{horizon}.npy"
        file_x_test = f"{windows_path}x_test_windows_horizon_{horizon}.npy"
        file_y_test = f"{windows_path}y_test_windows_horizon_{horizon}.npy"

        print(f"Reading training files {file_x_train} and {file_y_train}...")
        print(f"Reading training files {file_x_val} and {file_y_val}...")
        print(f"Reading test files {file_x_test} and {file_y_test}...")

        x_train = np.load(file_x_train)
        y_train = np.load(file_y_train)
        x_val = np.load(file_x_val)
        y_val = np.load(file_y_val)
        x_test = np.load(file_x_test)
        y_test = np.load(file_y_test)

        ####### Remove duplicated ###########

        ### On Train data #############

        # merge arrays
        df_x = pd.DataFrame(x_train)  # All x
        df_y = pd.DataFrame(y_train[:, -1])  # Last column of y
        df = pd.concat([df_x, df_y], axis=1)

        # Find duplicated
        dup = df.duplicated()
        print()
        print(f'Array Train duplicated:')
        print(dup.value_counts())

        # Remove duplicated
        df = df[~dup]
        df.reset_index(inplace=True, drop=True)

        x_train = df.iloc[:, :-1].values  # Get all columns but the last one
        y_train = df.iloc[:, -1:].values  # Get just the last column

        ### On Validation data #############

        # merge arrays
        df_x = pd.DataFrame(x_val)  # All x
        df_y = pd.DataFrame(y_val[:, -1])  # Last column of y
        df = pd.concat([df_x, df_y], axis=1)

        # Find duplicated
        dup = df.duplicated()
        print()
        print(f'Array val duplicated:')
        print(dup.value_counts())

        # Remove duplicated
        df = df[~dup]
        df.reset_index(inplace=True, drop=True)

        x_val = df.iloc[:, :-1].values  # Get all columns but the last one
        y_val = df.iloc[:, -1:].values  # Get just the last column

        ### On Test data #############

        # merge arrays
        df_x = pd.DataFrame(x_test)  # All x
        df_y = pd.DataFrame(y_test[:, -1])  # Last column of y
        df = pd.concat([df_x, df_y], axis=1)

        # Find duplicated
        dup = df.duplicated()
        print()
        print(f'Array test duplicated:')
        print(dup.value_counts())

        # Remove duplicated
        df = df[~dup]
        df.reset_index(inplace=True, drop=True)

        x_test = df.iloc[:, :-1].values  # Get all columns but the last one
        y_test = df.iloc[:, -1:].values  # Get just the last column

        return x_train, y_train, x_val, y_val, x_test, y_test

    def read_windows_with_duplicated(horizon):
        file_x_train = f"{windows_path}x_train_windows_horizon_{horizon}.npy"
        file_y_train = f"{windows_path}y_train_windows_horizon_{horizon}.npy"
        file_x_val = f"{windows_path}x_val_windows_horizon_{horizon}.npy"
        file_y_val = f"{windows_path}y_val_windows_horizon_{horizon}.npy"
        file_x_test = f"{windows_path}x_test_windows_horizon_{horizon}.npy"
        file_y_test = f"{windows_path}y_test_windows_horizon_{horizon}.npy"

        print(f"Reading training files {file_x_train} and {file_y_train}...")
        print(f"Reading training files {file_x_val} and {file_y_val}...")
        print(f"Reading test files {file_x_test} and {file_y_test}...")

        x_train = np.load(file_x_train)
        y_train = np.load(file_y_train)
        x_val = np.load(file_x_val)
        y_val = np.load(file_y_val)
        x_test = np.load(file_x_test)
        y_test = np.load(file_y_test)

        # Using only the horizon measurement as a label
        y_train = y_train[:, -1]
        y_train = y_train.reshape((y_train.shape[0], 1))
        y_val = y_val[:, -1]
        y_val = y_val.reshape((y_val.shape[0], 1))
        y_test = y_test[:, -1]
        y_test = y_test.reshape((y_test.shape[0], 1))

        return x_train, y_train, x_val, y_val, x_test, y_test

    ## Algorithms architectures ################################################
    from keras.models import Model
    from keras.layers import Dense, LSTM, GRU, Lambda, dot, concatenate, Activation, Input

    # LSTM
    class LSTMModel:
        def __init__(self, input_shape, nb_output_units, nb_hidden_units=128):
            self.input_shape = input_shape
            self.nb_output_units = nb_output_units
            self.nb_hidden_units = nb_hidden_units

        def __repr__(self):
            return 'LSTM_{0}_units_{1}_layers_dropout={2}_{3}'.format(self.nb_hidden_units, self.nb_layers, self.dropout,
                                                                      self.recurrent_dropout)

        def build(self):
            # input
            i = Input(shape=self.input_shape)

            # add LSTM layer
            x = LSTM(self.nb_hidden_units)(i)

            x = Dense(self.nb_output_units, activation=None)(x)

            return Model(inputs=[i], outputs=[x])

    # Linear
    class LinearModel:
        def __init__(self, input_shape, nb_output_units):
            self.input_shape = input_shape
            self.nb_output_units = nb_output_units

        def __repr__(self):
            return 'Linear'

        def build(self):
            i = Input(shape=self.input_shape)
            x = Dense(self.nb_output_units, activation=None)(i)

            return Model(inputs=[i], outputs=[x])

    # CNN
    class CNNModel:
        def __init__(self, input_shape, nb_output_units):
            self.input_shape = input_shape
            self.nb_output_units = nb_output_units

        def __repr__(self):
            return 'CNN_{0}_units'.format(self.nb_hidden_units)

        def build(self):
            i = Input(shape=self.input_shape)
            x = Conv1D(filters=16, kernel_size=2, activation='relu')(i)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=32, kernel_size=2, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Flatten()(x)
            x = Dense(units=50, activation='relu')(x)
            x = Dense(units=self.nb_output_units, activation='linear')(x)

            return Model(inputs=[i], outputs=[x])

    ## Model Functions ################################################
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    import keras.backend as K
    from tensorflow import keras
    import tensorflow as tf

    def RMSE(output, target):
        output = tf.cast(output, 'float32')
        target = tf.cast(target, 'float32')

        return tf.sqrt(tf.reduce_mean((output - target) ** 2))

    def build_model(model, weights=''):
        # build & compile model
        m = model.build()

        m.compile(loss=RMSE,
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.MeanSquaredError(), keras.metrics.RootMeanSquaredError()])
        if weights:
            print(f"Weights: {weights}")
            m.load_weights(weights)
        return m

    def callbacks(filepath, early_stopping_patience):
        callbacks = []
        callbacks.append(ModelCheckpoint(filepath=filepath + ".weights.h5",
                                         monitor='loss',
                                         save_best_only=True,
                                         save_weights_only=True))
        callbacks.append(EarlyStopping(monitor='loss', patience=early_stopping_patience))
        return callbacks

    import numpy as np

    def prepare_model_LSTM(history_length, nb_hidden_units=128, weights=''):
        model = LSTMModel(input_shape=(history_length, 1), nb_output_units=1, nb_hidden_units=nb_hidden_units)
        return build_model(model, weights)


    def prepare_model_linear(history_length, weights=''):
        model = LinearModel(input_shape=(history_length,), nb_output_units=1)
        return build_model(model, weights)


    def prepare_model_CNN(history_length, weights=''):
        model = CNNModel(input_shape=(history_length, 1), nb_output_units=1)
        return build_model(model, weights)

    ############## TRAIN FUNCTION #####################################
    def train(x_train, y_train, x_val, y_val, model, horizon, save_filepath="", early_stopping_patience=patience):
        history_length = 8
        x_train = np.reshape(x_train, (x_train.shape[0], history_length, 1))

        x_val = np.reshape(x_val, (x_val.shape[0], history_length, 1))
        validation_data = (x_val, y_val)

        hist = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         validation_data=validation_data,
                         epochs=max_epoch,
                         # shuffle=True,
                         callbacks=callbacks(save_filepath + str(horizon),
                                             early_stopping_patience)
                         )

        return hist, model

    def get_train_plots_loss(hist: keras.callbacks, name: str):
        fig, ax = plt.subplots()

        # data
        x_epoch = hist.epoch
        y_val_loss = hist.history['val_loss']
        y_train_loss = hist.history['loss']

        # Create a line plots
        ax.plot(x_epoch, y_val_loss, label='Validation loss', color='orange', linestyle='-')
        ax.plot(x_epoch, y_train_loss, label='Train loss', color='blue', linestyle='-')

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and validation loss')

        ax.legend()

        fig.savefig(name + '_Loss.pdf', dpi=350, bbox_inches='tight')

    def get_train_plots_RMSE(hist: keras.callbacks, name: str):
        fig, ax = plt.subplots()

        # Sample data
        x_epoch = hist.epoch
        y_val_rmse = hist.history['val_root_mean_squared_error']
        y_train_rmse = hist.history['root_mean_squared_error']

        # Create a line plot
        ax.plot(x_epoch, y_val_rmse, label='Validation RMSE', color='orange', linestyle='-')
        ax.plot(x_epoch, y_train_rmse, label='Train RMSE', color='blue', linestyle='-')

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Training and validation RMSE')

        # Add a legend
        ax.legend()

        # Display the plot
        fig.savefig(name + '_RMSE.pdf', dpi=350, bbox_inches='tight')

    def get_train_plots_MSE(hist: keras.callbacks, name: str):
        fig, ax = plt.subplots()

        # Sample data
        x_epoch = hist.epoch
        y_val_mse = hist.history['val_mean_squared_error']
        y_train_mse = hist.history['mean_squared_error']

        # Create a line plot
        ax.plot(x_epoch, y_val_mse, label='Validation MSE', color='orange', linestyle='-')
        ax.plot(x_epoch, y_train_mse, label='Train MSE', color='blue', linestyle='-')

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Training and validation MSE')

        # Add a legend
        ax.legend()

        # Display the plot
        fig.savefig(name + '_MSE.pdf', dpi=350, bbox_inches='tight')

    # Results    ################################################

    ## Functions ################################################
    def clarke_error_grid(ref_values, pred_values, title_string, show_plot=False):
        """
          This function takes in the reference values and the prediction values as lists and returns a list with each index corresponding to the total number
         of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot
        """
        # Checking to see if the lengths of the reference and prediction arrays are the same
        assert (len(ref_values) == len(
            pred_values)), "Unequal number of values (reference : {0}) (prediction : {1}).".format(len(ref_values),
                                                                                                   len(pred_values))

        # Checks to see if the values are within the normal physiological range, otherwise it gives a warning
        if max(ref_values) > 500 or max(pred_values) > 500:
            print(
                "Input Warning: the maximum reference value {0} or the maximum prediction value {1} exceeds the normal physiological range of glucose (<400 mg/dl).".format(
                    max(ref_values), max(pred_values)))
        if min(ref_values) < 0 or min(pred_values) < 0:
            print(
                "Input Warning: the minimum reference value {0} or the minimum prediction value {1} is less than 0 mg/dl.".format(
                    min(ref_values), min(pred_values)))

        values_out_grid = sum(value > 500 for value in pred_values) + sum(value < 0 for value in pred_values)
        print(f"Number of values outside the grid: {values_out_grid}")

        if show_plot:
            # Clear plot
            plt.clf()

            # Set up plot
            plt.scatter(ref_values, pred_values, marker='o', color='black', s=8)
            plt.title(title_string + " Clarke Error Grid")
            plt.xlabel("Reference Concentration (mg/dl)")
            plt.ylabel("Prediction Concentration (mg/dl)")
            plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
            plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
            plt.gca().set_facecolor('white')

            # Set axes lengths
            plt.gca().set_xlim([0, 500])
            plt.gca().set_ylim([0, 500])
            plt.gca().set_aspect((500) / (500))

            # Plot zone lines
            plt.plot([0, 500], [0, 500], ':', c='black')  # Theoretical 45 regression line
            plt.plot([0, 175 / 3], [70, 70], '-', c='black')
            # plt.plot([175/3, 320], [70, 500], '-', c='black')
            plt.plot([175 / 3, 500 / 1.2], [70, 500], '-',
                     c='black')  # Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
            plt.plot([70, 70], [84, 500], '-', c='black')
            plt.plot([0, 70], [180, 180], '-', c='black')
            plt.plot([70, 290], [180, 500], '-', c='black')
            # plt.plot([70, 70], [0, 175/3], '-', c='black')
            plt.plot([70, 70], [0, 56], '-', c='black')  # Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
            # plt.plot([70, 500],[175/3, 320],'-', c='black')
            plt.plot([70, 500], [56, 320], '-', c='black')
            plt.plot([180, 180], [0, 70], '-', c='black')
            plt.plot([180, 500], [70, 70], '-', c='black')
            plt.plot([240, 240], [70, 180], '-', c='black')
            plt.plot([240, 500], [180, 180], '-', c='black')
            plt.plot([130, 180], [0, 70], '-', c='black')

            # Add zone titles
            plt.text(30, 15, "A", fontsize=15)
            plt.text(370, 260, "B", fontsize=15)
            plt.text(280, 370, "B", fontsize=15)
            plt.text(160, 370, "C", fontsize=15)
            plt.text(160, 15, "C", fontsize=15)
            plt.text(30, 140, "D", fontsize=15)
            plt.text(370, 120, "D", fontsize=15)
            plt.text(30, 370, "E", fontsize=15)
            plt.text(370, 15, "E", fontsize=15)

            # plt.savefig(f'plots/Clarke_Error_Grid{title_string}.pdf')
            plt.savefig(f'plots/Clarke_Error_Grid{title_string}.jpg', dpi=350, bbox_inches='tight')

        # Statistics from the data
        zone = [0] * 5
        for i in range(len(ref_values)):
            if (ref_values[i] <= 70 and pred_values[i] <= 70) or (
                    pred_values[i] <= 1.2 * ref_values[i] and pred_values[i] >= 0.8 * ref_values[i]):
                zone[0] += 1  # Zone A

            elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
                zone[4] += 1  # Zone E

            elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or (
                    (ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7 / 5) * ref_values[i] - 182)):
                zone[2] += 1  # Zone C
            elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (
                    ref_values[i] <= 175 / 3 and pred_values[i] <= 180 and pred_values[i] >= 70) or (
                    (ref_values[i] >= 175 / 3 and ref_values[i] <= 70) and pred_values[i] >= (6 / 5) * ref_values[i]):
                zone[3] += 1  # Zone D
            else:
                zone[1] += 1  # Zone B

        return zone

    def get_values_per_zone(values):
        keys = ['A', 'B', 'C', 'D', 'E']
        return {key: value for key, value in zip(keys, values)}

    def show_results_(model, x_test, y_test, plot_name='', plot_flag=False):
        print("Calculating test score")
        test_score = model.evaluate(x_test, y_test)

        # PREDICTION
        print("########################################################################")
        print("Predicting BG values...")
        BG_predicted_values = model.predict(x_test)
        print("########################################################################")
        print()

        # Clarke error grid
        values = clarke_error_grid(y_test, BG_predicted_values, plot_name, show_plot=plot_flag)

        values_zones = get_values_per_zone(values)
        predicted_values_len = y_test.shape[0]
        perc_values_zones = [value / predicted_values_len * 100 for value in values_zones.values()]

        perc_values_zones = get_values_per_zone(perc_values_zones)

        print('Test score: ', test_score)
        print('Percentage values zones: ', perc_values_zones)

        return test_score, perc_values_zones, BG_predicted_values

    def test_by_range(df_result_vector: pd.DataFrame, grid_name:str)->pd.DataFrame:
        df_result_vector_TBR_2 = df_result_vector[df_result_vector['y_test'] < 54]

        df_result_vector_TBR_1 = df_result_vector[
            (df_result_vector['y_test'] >= 54) & (df_result_vector['y_test'] < 70)]

        df_result_vector_TIR = df_result_vector[(df_result_vector['y_test'] >= 70) & (df_result_vector['y_test'] < 181)]

        df_result_vector_TAR_1 = df_result_vector[
            (df_result_vector['y_test'] >= 181) & (df_result_vector['y_test'] < 251)]

        df_result_vector_TAR_2 = df_result_vector[(df_result_vector['y_test'] >= 251)]

        list_dataframe_by_range = [df_result_vector, df_result_vector_TBR_2, df_result_vector_TBR_1,
                                   df_result_vector_TIR, df_result_vector_TAR_1, df_result_vector_TAR_2]
        list_dataframe_by_range_name = ['ALL', 'TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']

        df_metrics_summary = pd.DataFrame(
            columns=['Model', 'A', 'B', 'C', 'D', 'E', 'A + B', 'RMSE', 'MSE', 'MAE', 'MAPE'])

        for i in range(6):
            reference_values = list_dataframe_by_range[i]['y_test']
            pred_values = list_dataframe_by_range[i]['y_predict']
            print(list_dataframe_by_range_name[i])
            zone = clarke_error_grid(ref_values=reference_values.values, pred_values=pred_values.values,
                                     title_string=grid_name + list_dataframe_by_range_name[i], show_plot=True)

            print(f'Zone Values: {zone}')
            zone_percentajes = round(pd.Series(zone) / reference_values.shape[0] * 100, 2)
            print(f'Zone percentages: {zone_percentajes.tolist()}')
            print()

            # Classic metrics
            mse = mean_squared_error(reference_values, pred_values)
            rmse = np.sqrt(mean_squared_error(reference_values, pred_values))
            mae = mean_absolute_error(reference_values, pred_values)
            mape = mean_absolute_percentage_error(reference_values, pred_values)

            print(f'Root Mean Squared Error (RMSE): {rmse}')
            print(f'Mean absolute Error (MAE): {mae}')
            print(f'Mean absolute percentage Error (MAPE): {mape}')

            new_row = {'Model': list_dataframe_by_range_name[i], 'A': zone_percentajes[0], 'B': zone_percentajes[1],
                       'C': zone_percentajes[2], 'D': zone_percentajes[3], 'E': zone_percentajes[4],
                       'A + B': zone_percentajes[0] + zone_percentajes[1], 'RMSE': rmse, 'MSE': mae, 'MAE': mae,
                       'MAPE': mape}
            new_index = len(df_metrics_summary)
            df_metrics_summary.loc[new_index] = new_row

            print()
            print('-' * 100)
            print()

        return df_metrics_summary

    #################################################### TRAIN AND TEST ##########################################################

    for current_horizon in horizons:  # For all horizons -------------------------------------
        if repeated_examples:
            x_train, y_train, x_val, y_val, x_test, y_test = read_windows_with_duplicated(horizon=current_horizon)
        else:
            x_train, y_train, x_val, y_val, x_test, y_test = read_windows_no_duplicated(horizon=current_horizon)

        for current_model in models:  # For all models ---------------------------------------

            if current_model == 'Linear':  # prepare linear model
                model = prepare_model_linear(history_length)
            elif current_model == 'LSTM': # prepare LSTM model
                model = prepare_model_LSTM(history_length)
            else: # prepare CNN model
                model = prepare_model_CNN(history_length)


            print("######################################################################## - ")
            print(f'START training {current_model} horizon={current_horizon}')
            print()

            # Record the TRAINING start time
            start_time = time.time()

            hist, model_trained = train(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, model=model,
                                        horizon=current_horizon, save_filepath=f"models/{current_model}")

            # Record the TRAINING end time
            end_time = time.time()
            # Calculate the elapsed time of TRAINING
            elapsed_time = end_time - start_time
            print(f"Training time for {current_model} horizon={current_horizon}: {elapsed_time} seconds")

            get_train_plots_loss(hist, f"plots/{current_model}_H{current_horizon}")
            get_train_plots_MSE(hist, f"plots/{current_model}_H{current_horizon}")

            df_hist = pd.DataFrame(data=hist.history)
            df_hist.to_csv(f'hist/df_hist_{current_model}_H{current_horizon}.csv')

            hist_best = df_hist.iloc[df_hist['loss'].idxmin()]
            hist_best['epoch'] = df_hist.shape[0]
            hist_best.to_csv(f'hist/hist_best_{current_model}_H{current_horizon}.csv')

            print()
            print(f'END training {current_model} horizon={current_horizon}')
            print("########################################################################")
            print()

            # ## TEST ################################

            print(f'Test results for {current_model} horizon={current_horizon}')

            # Record the TEST start time
            start_time = time.time()

            if current_model == 'Linear':  # load linear model
                model_load = prepare_model_linear(history_length,
                                                  weights=f"models/{current_model}{current_horizon}.weights.h5")  # Linear4
            else:
                model_load = prepare_model_LSTM(history_length, weights=f"models/{current_model}{current_horizon}.weights.h5")

           
            bg_predict = model_load.predict(x_test)

            # Record the TEST end time
            end_time = time.time()
            # Calculate the elapsed time of TEST
            elapsed_time = end_time - start_time
            print(f"Testing time for {current_model} horizon={current_horizon}: {elapsed_time} seconds")

            # Save predictions and references
            df_test_results = pd.DataFrame({'y_test': y_test.ravel(), 'y_predict': bg_predict.ravel()})
            df_test_results.to_parquet(f'test_results/df_test_results_vectors_{current_model}_H{current_horizon}.parquet')

            df_full_range_metrics = test_by_range(df_test_results, f'_{current_model}_H{current_horizon}_')
            df_full_range_metrics.to_csv(f'test_results/results_by_range_{current_model}_H{current_horizon}.csv')


            print(f'Test results of {current_model} horizon={current_horizon} - DONE')
            print('____________________________________________________________________________')
            print()
finally:
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()