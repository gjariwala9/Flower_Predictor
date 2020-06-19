# Flower Predictor: Project Overview 
* This is the most famous example in the field of machine learning.
* The aim is to classify iris flowers among three species (setosa, versicolor, or virginica) from measurements of length and width of sepals and petals.
* The iris data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. 
* The central goal here is to design a model that makes good classifications for new flowers or, in other words, one which exhibits good generalization 
* Built a client facing API using flask 

## Code and Resources Used 
**Python Version:** 3.8  
**Packages:** pandas, numpy, matplotlib, sklearn, tensorflow 2, flask, json, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  

## Data Set
The data in in iris.csv. The number of columns is 5 and the number of rows is 150. The columns are:
* sepal_length: Sepal length, in centimeters, used as input.
* sepal_width: Sepal width, in centimeters, used as input.
* petal_length: Petal length, in centimeters, used as input.
* petal_width: Petal width, in centimeters, used as input.
* class: Iris Setosa, Versicolor, or Virginica, used as the target.

## Model Building 

First, I have scaled the data with MinMaxScaler. I also split the data into train and tests sets with a test size of 20%.   

I have applied Artificial Neural Network (ANN) with two layers:
*	**Input Layer** – Units=4, activation=relu 
*	**Output Layer** – Units=2, activation=softmax

I have also applied early stopping.

## Model performance 
*	**Validation loss** : 0.35
*	**Validation Accuracy** : 0.97 or 97%
*	**Confusion matrix for label setosa** :
[[20  0]
 [ 0 10]]
*	**Confusion matrix for label versicolor** :
[[17  1]
 [ 1 11]]
*	**Confusion matrix for label virginica** :
[[22  0]
 [ 1  7]]

## Productionization 
In this step, I built a flask API endpoint that is hosted on Heroku. The API endpoint takes in a request with a list of values of flower measurements and returns the class of the flower.

**Application link:** https://flower-predictor.herokuapp.com/

![alt text](https://github.com/gjariwala9/Flower_Predictor/blob/master/README_IMG/form.png "Flower Measurements Form")

![alt text](https://github.com/gjariwala9/Flower_Predictor/blob/master/README_IMG/prediction.png "Flower Prediction")
