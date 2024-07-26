#from utils import db_connect
#engine = db_connect()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# your code here
df = pd.read_csv("/workspaces/machine-learning-python-template/data/raw/AB_NYC_2019.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe(include=np.number).T)
print(df.describe(include=["O"]).T) #or "object"
print(df.drop("id", axis = 1).duplicated().sum())

if df.drop("id", axis = 1).duplicated().sum() != 0:
    df.drop_duplicates(inplace=True)

"""Clean the data"""

df.drop(["id", "name", "host_id", "host_name"], axis = 1, inplace = True)
df_num = df.select_dtypes(include=np.number)
df_cat = df.select_dtypes(include=["object"])
print(df.info())

"""Categorical Data"""
fig, axis = plt.subplots(2, 2, figsize = (10, 7))

index = 0
for i in range(2):
    if index > 3:
        break
    for j in range(2):
        c = df_cat.columns[index]
        s = sns.histplot(ax = axis[i,j],data = df_cat, x = c)
        if c in ["neighbourhood", "last_review"]:
            s.set_xticks([])
        index  +=1
# Adjust the layout
plt.xticks(rotation = 90)
plt.tight_layout()

# Show the plot
plt.savefig("df_cat.jpg")

plt.show()

"""Numerical Data"""
fig, axis = plt.subplots(4, 4, figsize = (10, 7))

index = 0
for i in range(2):
    if index > 5:
        break
    for j in range(4):
        c = df_num.columns[index]
        sns.histplot(ax = axis[i,j],data = df_num, x = c)
        sns.boxplot(ax = axis[i+2,j],data = df_num, x = c)
        index  +=1
# Adjust the layout
plt.tight_layout()

# Show the plot
plt.savefig("df_num.jpg")

plt.show()
print("Done")




