import json
from operator import length_hint
import pandas as pd
file = r'covid_19_dataset_2020_06_10\yelp_academic_dataset_covid_features.json'
n = 109795
columns = ["highlights","delivery or takeout","Grubhub enabled",
           "Call To Action enabled","Request a Quote Enabled",
           "Covid Banner","Temporary Closed Until","Virtual Services Offered"]
train_data,test_data = [],[]
with open(file,encoding='UTF-8') as f:
    for i in range(n):
        line = json.loads(f.readline())
        res = []
        for j in range(8):
            res.append(1 if line[columns[j]]!= 'FALSE' else 0)
        train_data.append(res)

    lines =[json.loads(i) for i in f.readlines()]
    for line in lines:
        res = []
        for i in range(8):
            res.append(1 if line[columns[i]]!= 'FALSE' else 0)
        test_data.append(res)
df_train_data = pd.DataFrame(train_data,columns=columns)
df_test_data = pd.DataFrame(test_data,columns=columns)
print(df_test_data['delivery or takeout'])
