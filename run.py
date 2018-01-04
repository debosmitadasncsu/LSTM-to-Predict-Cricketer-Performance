# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:05:52 2017

"""

import lstm
import time
import matplotlib.pyplot as plt

def plot_results(predicted_data, true_data):
    true_data=[x[:-1] for x in true_data]
    true_data = list(map(float, true_data))
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    true_data=[x[:-1] for x in true_data]
    true_data = list(map(int, true_data))
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
 global_start_time = time.time()
 epochs  = 50
 seq_len = 20

 print('> Loading data... ')

 X_train, y_train, X_test, y_test = lstm.load_data('score.csv', seq_len, False)
 X_trainn, y_trainn, X_testn, y_testn = lstm.load_data('score.csv', seq_len, True)
    
 print('> Data Loaded. Compiling...')

 model = lstm.build_model([1, 5, 10, 1])

 model.fit(
	    X_trainn,
	    y_trainn,
	    batch_size=512,
	    nb_epoch=epochs,
	    validation_split=0.05)

 predictions = lstm.predict_sequences_multiple(model, X_testn, seq_len, 10)
 
 for i in range(len(predictions)):
    for j in range(len(predictions[i])):
      predictions[i][j]=int(X_test[i*len(predictions[i])+j][0][0].split("\r")[0])*(predictions[i][j]+1)
      print(i)
 plot_results_multiple(predictions, y_test, 10)  
 
 
 predicted1 = lstm.predict_sequence_full(model, X_testn, seq_len)
 for i in range(len(predicted1)):
      predicted1[i]=int(X_test[i][0][0].split("\r")[0])*(predicted1[i]+1)/5
 plot_results(predicted1, y_test)
 
 
 predicted = lstm.predict_point_by_point(model, X_testn) 
 for i in range(len(predicted)):
      predicted[i]=int(X_test[i][0][0].split("\r")[0])*(predicted[i]+1)      
 plot_results(predicted, y_test)
 
 
 print('Training duration (s) : ', time.time() - global_start_time)
 
