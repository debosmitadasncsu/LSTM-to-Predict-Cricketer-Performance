# Background

This study mainly aims at developing a deep learning model that can be used for the prediction of performance of a team or player in a cricket match. Not all countries play cricket and in even fewer countries this sport is considered to be one of the most popular ones. The sport is restricted to Commonwealth countries.  Hence, the main challenge was to find a proper dataset with enough number of features. 
The dataset was found on cricsheets.org. It had the ball by ball scores of all games played between 2005 to 2016. This paper looks into the data of the One Day International matches which have occurred in the said duration. This provided a rich source of data through which any relevant features could be calculated with ease.
There are two main problems these papers try to address. The first one, predicting the performance of a player using his past performances. Every team wants to select the best players for international games. This paper looks at the runs scored by a batsman to predict performance.
Predicting whether a player will perform better or worse than his previous game can go a long way in helping build the right team for any game. The second problem attempts to find the performance of a team across the games. This can help predict whether a team is likely to perform better or worse in the upcoming games. This information can be used to tweak the team composition in hopes of a better performance.
This has a large scale monetary impact as a good international ranking of a team at the end of a year makes it eligible for monetary awards. Though, the prestige of performing well at an international stage holds a certain appeal as well.

# Approach

The dataset was obtained from cricsheets.org[21]. It provides a ball by ball detail of the games which occurred between 2005 and 2016. This paper uses data from One Day Internationals. 
The problem is to predict the runs scored by a batsman based on his previous performances. A player who has played extensively for the period of time from which the data was available (ie 2005-16), is to be chosen, in order to generate sufficient data.. As the data has ball by ball details for every game, it is necessary to mine through the relevant games and calculate the runs scored by that particular player. These runs are stored as an array of scores.
As LSTMs is essentially a time series prediction, every element in the array is the ‘y’ column, ie, the column to be predicted. For each ‘y’ value, the previous 20 scores were used as the features. Thus, the array of scores is reduced by 20 scores, as only the 20th score onwards could be used to generate the data. The data would now look like this -


__Dataset after feature generation __

Previous Scores&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Prediction Column

score 1 score 2 ... score 19&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score 20

score 2 score 3 ... score 20&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score 21

                                                       
...

score k-19 score k-18 ... score  k-1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score k


This data is then divided into a test and train set with a 80-20 split. The training set is used to model the network. The LSTM network which has the following number  of variables per layer. [1,5,10,1]. Meaning, 1 input expanding to 5 variables and then 10, before converging to 1 as the output. Thus, there are 2 hidden layers.
Normalization of the data is used to make sure that the data lies on a similar scale across a particular input. This is done by normalizing with respect to the first feature in every observation. The following formula illustrates this point.
	Normalized Value = Vi/V0 - 1 
The predicted values are denormalized using the inverse of the above formula. 
As we are focussing on the trends of the performance of a player, the test data is also used to generate another array where one is set for the element if it is greater than its previous value, and zero otherwise. The same is done for the predicted values. These arrays are then compared on a plot to see if the trends of the predicted performances match those of the actual test performances.
Another aspect consider for prediction purposes, is the number of features used for each innings. The following features were extracted from the data.
Runs
Strike Rate
Fours
Sixes
 4 different models were built, based on the number of features used. The first model used the first feature. The second used the first two, and so on. 
In our developed model, we are using 3 layers - 1 input layer, 1 hidden layer and 1 output layer. For our input layer we are specifying the input dimension as 1 and we are making that layer to give the output in the shape of (*, 20). This output will be feed into the hidden layer as an input array of dimension (*, 20). The hidden layer has been specified in our model to spit out an output of shape (*, 100), which is again the input of our final output layer. Our final output is of the same shape as the input one i.e. of shape (*, 1).
Besides these specifications, we have used the return_sequence parameter of LSTM model. For the 1st layer we wanted to return the output to both the 2nd layer and the layer itself. This means that 1st layer’s output is always fed to the second layer. As a whole regarding time, all its activations can be seen as the sequence of prediction this first layer has made from the input sequence. On the other hand, for the second layer its output is only fed to the next layer at the end of the sequence. As a whole regarding time, it does not output a prediction for the sequence but one only prediction-vector for the whole input sequence. The linear Dense layer of the last layer, is used to aggregate all the information from this prediction-vector into one single value, the predicted 3rd time step of the sequence.
Other than this, in the final layer, we are calculating the loss against the target by using mean squared error and a linear Activation function. Since the Activation function is a linear one, hence ‘rmsprop’ optimizer is used here as it works well with linear data sets.
For implementing the “forget” functionality of LSTM, we have used the Dropout property. A dropout on the input means that for a given probability, the data on the input connection to each LSTM block will be excluded from node activation and weight updates. In our model, we have used a dropout of 20%.
