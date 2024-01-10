#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

# print(len([i for i in enron_data['PRENTICE JAMES'] if i=='stock']))
dictenron_data = dict(enron_data)
print(dictenron_data)
print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])
print(len([dictenron_data[i]['salary'] for i in dictenron_data if dictenron_data[i]['salary']!='NaN']))
print(len([dictenron_data[i]['email_address'] for i in dictenron_data if dictenron_data[i]['email_address']!='NaN']))