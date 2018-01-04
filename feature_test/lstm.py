import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_run_data(filename, seq_len, normalise_flag):
    f = open(filename, 'rb').read()
    data_temp = f.decode().strip().split('\n')
    data = [ x.split(',') for x in data_temp]

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])    

    # result = np.array(result)
    
    result = normalise_windows(result, normalise_flag)
    result = np.array(result)
    #split data into test and training data sets
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    
    result = np.array(result)
    
    x_train = train[:, :-1]
    y_train = train[:, -1, -1]
    
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1, -1]

    return [x_train, y_train, x_test, y_test]    



def normalise_windows(window_data, normalise_flag):
    normalised_data = []
    for window in window_data:
        base_list = [ float(x) + 1 for x in window[0] ]
        new_window = []
        for e in window:
            if normalise_flag:
                new_window.append([ float(float(i)/j)-1 for i,j in zip(e, base_list)])
            else:
                new_window.append([ float(x) for x in e])
        normalised_data.append(new_window)
    return normalised_data



def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_shape=(layers[1], layers[0]),
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
