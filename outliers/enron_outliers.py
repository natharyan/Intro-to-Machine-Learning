#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL")
data = featureFormat(data_dict, features)


### your code below
salaries, bonuses = zip(*data)
plt.scatter(salaries, bonuses, color="b")
plt.xlabel("salary")
plt.ylabel("bonus")
salary_range = max(salaries) - min(salaries)
bonus_range = max(bonuses) - min(bonuses)
plt.xlim(min(salaries) - 0.1 * salary_range, max(salaries) + 0.1 * salary_range)
plt.ylim(min(bonuses) - 0.1 * bonus_range, max(bonuses) + 0.1 * bonus_range)
# plt.show()
print([key for key in data_dict if (data_dict[key]["bonus"]!='NaN' and data_dict[key]["bonus"]>=5000000 and data_dict[key]["salary"]!='NaN' and data_dict[key]["salary"]>1000000)])