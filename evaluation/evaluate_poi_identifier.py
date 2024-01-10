#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list,sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)


### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features,labels,random_state=42,test_size=0.3)

clf = DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print(labels_test)
print(len([i for i in labels_test if i]))
print(len(labels_test))
true_positives = [i for i in range(len(pred)) if (pred[i] == 1.0) & (labels_test[i] == 1.0)]
print(pred, true_positives)

from sklearn.metrics import precision_score, recall_score

precision = precision_score(labels_test, pred) ; print(precision)
recall = recall_score(labels_test, pred) ; print(recall)

# practice on precision and recall
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
print("true positives:", len([i for i in range(len(predictions)) if (predictions[i] == 1) & (true_labels[i] == 1)]))
print("true negatives:", len([i for i in range(len(predictions)) if (predictions[i] == 0) & (true_labels[i] == 0)]))
print("false positives:", len([i for i in range(len(predictions)) if (predictions[i] == 1) & (true_labels[i] == 0)]))
print("false negatives:", len([i for i in range(len(predictions)) if (predictions[i] == 0) & (true_labels[i] == 1)]))
print("precision", len([i for i in range(len(predictions)) if (predictions[i] == 1) & (true_labels[i] == 1)])/(len([i for i in range(len(predictions)) if (predictions[i] == 1) & (true_labels[i] == 1)]) + len([i for i in range(len(predictions)) if (predictions[i] == 1) & (true_labels[i] == 0)])))
print("recall:",len([i for i in range(len(predictions)) if (predictions[i] == 1) & (true_labels[i] == 1)])/(len([i for i in range(len(predictions)) if (predictions[i] == 1) & (true_labels[i] == 1)]) + len([i for i in range(len(predictions)) if (predictions[i] == 0) & (true_labels[i] == 1)])))