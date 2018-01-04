import lstm
import time
import matplotlib.pyplot as plt
import numpy as np
def plot_results(predicted_data, true_data, name):
    
    #true_data = list(map(float, true_data))    
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig(name + '.png')
    # plt.show()
	
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
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
    
   file_names = ["runs_features_1.csv", "runs_features_2.csv", "runs_features_3.csv", "runs_features_4.csv"]
   
   file_names = ["runs_features_1.csv"]
   #file_names =  ["runrate_England.csv"]          
   for z in range(0, len(file_names)):
        #parameters
        epochs  = 50
        seq_len = 20
        features = z+1
        file_name = file_names[z]

        print('Processing Data...')
        
        X_train, y_train, X_test, y_test = lstm.load_run_data(file_name, seq_len, True)
        X_trainn, y_trainn, X_testn, y_testn = lstm.load_run_data(file_name, seq_len, False)
            
        print('Data Loaded. Building Model...')
    
        model = lstm.build_model([features, seq_len, 100, 1])
    
        model.fit(
                X_train,
                y_train,
                batch_size=512,
                nb_epoch=epochs,
                validation_split=0.05)
    
        print('Training duration (s) : ', time.time() - global_start_time)
        predictions = lstm.predict_point_by_point(model, X_test)  
        name = "pic_features_" + str(z+1)
        plot_results(predictions, y_test, name)
    
	
	
	
        print('Comparison between (test, predicted)')
        for p,y in zip(y_test, predictions):
                print(str(round(p,5)) + "   " + str(round(y,5)))   
    
            #predict trend
        predictions_trend = []
        y_test_trend = []
        for x in range(1, len(predictions)):
              if y_test[x] > y_test[x-1]:                                                                            
                 y_test_trend.append(1)
              else:
                 y_test_trend.append(-1)                         
        for x in range(1, len(y_test)):
          if predictions[x] > predictions[x-1]:
            predictions_trend.append(1)
          else:
            predictions_trend.append(-1)
        name = "pic_features_" + str(z+1) + "_trends"
        plot_results(predictions_trend, y_test_trend, name) 
		
		 
        a=predict_sequence_full(model, X_test, 20) 
        for i in range(len(a)):
            a[i]=float(X_testn[i][0][features-1])*(a[i]+1)  
        plot_results(a, y_testn, name)
		
		
		
        predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 10)  
        for i in range(len(predictions)):
                for j in range(len(predictions[i])):
                    predictions[i][j]=int(X_testn[i*len(predictions[i])+j][0][0])*(predictions[i][j]+1)
                    print(i)
        plot_results_multiple(predictions, y_testn, 10) 
