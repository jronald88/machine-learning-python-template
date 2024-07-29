#from utils import db_connect
#engine = db_connect()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# your code here
df = pd.read_csv("/workspaces/machine-learning-python-template/data/raw/AB_NYC_2019.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe(include=np.number).T)
print(df.describe(include=["O"]).T) #or "object"
print("There are " + str(df.drop("id", axis = 1).duplicated().sum()) + " duplicates")

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
        if c in ["neighbourhood", "last_review", "room_type"]:
            s.set_xticks([])
        index  +=1
# Adjust the layout
plt.xticks(rotation = 90)
plt.tight_layout()

# Show the plot
plt.savefig("df_cat.jpg")

plt.clf()
sns.countplot(data = df, x="room_type", hue="price")
plt.savefig("corr_price_roomType.jpg")

plt.clf()
sns.regplot(data=df, x="minimum_nights", y="price")
plt.savefig("reg_min_nights_price.jpg")

plt.clf()
sns.regplot(data=df, x="reviews_per_month", y="number_of_reviews")
plt.savefig("reg_reviews.jpg")

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

"""Heatmaps & Analysis of multivariate variables"""

plt.clf()
sns.heatmap(df_num.corr(), annot = True)
plt.tight_layout()
plt.savefig("num_heatmap.jpg")

plt.clf()
sns.pairplot(data = df)
plt.tight_layout()
plt.savefig("pairplot_total.jpg")
print("Day 1 Complete")


"""DAY 2 - Feature Engineering"""
summary  =  df.describe().T
print(summary)

plt.clf()
fig, axis = plt.subplots(3, 3, figsize = (15, 10))

sns.boxplot(ax = axis[0, 0], data = df, y = "latitude")
sns.boxplot(ax = axis[0, 1], data = df, y = "longitude")
sns.boxplot(ax = axis[0, 2], data = df, y = "price")
sns.boxplot(ax = axis[1, 0], data = df, y = "minimum_nights")
sns.boxplot(ax = axis[1, 1], data = df, y = "number_of_reviews")
sns.boxplot(ax = axis[1, 2], data = df, y = "reviews_per_month")
sns.boxplot(ax = axis[2, 0], data = df, y = "calculated_host_listings_count")
sns.boxplot(ax = axis[2, 1], data = df, y = "availability_365")

plt.tight_layout()
plt.savefig("boxplots.jpg")

price_stats = df["price"].describe()
print(price_stats)

price_iqr = price_stats["75%"]-price_stats["25%"]
price_upper_limit = price_stats["75%"] + 1.5 * price_iqr
price_lower_limit = price_stats["25%"] - 1.5 * price_iqr

print(f"The upper and lower limits for finding outliers are {round(price_upper_limit, 2)} and {round(price_lower_limit, 2)}, with an interquartile range of {round(price_iqr, 2)}")

print(df[df["price"]>5000])

print(df.isnull().sum().sort_values(ascending=False)/len(df))
#convert last_review column to pandas datetime type.
df['last_review'] = pd.to_datetime(df['last_review'])

# Fill missing dates with the previous valid date
df['last_review'] = df['last_review'].fillna(method='ffill')
#fill missing data with mean number of reviews per month
df["reviews_per_month"].fillna(df["reviews_per_month"].mean(), inplace = True)
print(df.isnull().sum().sort_values(ascending=False)/len(df))

"""we want to predict the price of the houses. Split train set y var = price. """