Requirements: 
Libraries- math, random,pickle,sys,re

How to Run:
To create a a hard coded form of the hypothesis use the below command line arguments:
train <examples> <hypothesisOut> <learning-type>

hypothesisOut is the name of the while where the hard-coded form of the decision tree or ensemble will be stored

To do a prediction, use the below command line arguments:
predict <hypothesis> <file>



Training examples are kept In the train.txt file.

A hard coded example of the adaboost model is stored in the file Output_pickle.txt
Please use pickle.load() method to run this file