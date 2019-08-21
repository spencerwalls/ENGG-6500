# This program takes as input the "heart.csv" dataset.
# A decision tree is then implemented with respect to 
# this dataset, and the output is simply the accuracy
# of the algorithm's predictions. The data is first 
# separated into a training set and a testing set,
# which collectively comprise a 70/30 train/test split. 

# Step 1 - import necessary libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 

# Step 2 - import the dataset
data = pd.read_csv("decisiontree_heart_dataset.csv").values

# Step 3 - split the dataset into X and y
X = data[:,0:5]
y = data[:,-1]

# Step 4 - split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Step 5 - create in instance of the decision tree algorithm 
decision_tree = DecisionTreeClassifier()

# Step 6 - fit the algorithm to the dataset 
decision_tree = decision_tree.fit(X_train, y_train)

# Step 7 - make predictions for y_test using the testing set (X_test) as inputs 
predict = decision_tree.predict(X_test)

# Step 8 - store predicted values with ground truth values in a table and print it
table = np.zeros((91,2), dtype=np.int64)
table[:,0] = y_test
table[:,1] = predict
print("\nactual values vs. predicted values\n")
print(table)

# Step 8 - print the accuracy of the decision tree algorithm
print("\nAccuracy of decision tree:", metrics.accuracy_score(y_test, predict))



