#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
houses  = pd.read_csv("AmesHousing.tsv", delimiter = "\t")
houses.head()


# In[2]:


a = houses.isnull().sum().sort_values(ascending = False)
a_index = a[a.values > len(houses)/20].index
houses = houses.drop(a_index, axis = "columns")
houses.head()    


# In[ ]:





# In[3]:


num_null_index = houses.select_dtypes(include=['integer', 'float']).isnull().sum().sort_values(ascending = False)[:9].index
for i in num_null_index:
    houses[i] = houses[i].fillna(houses[i].mode()[0])
houses


# In[4]:


def train_and_test(df):
    train = df[:1460]
    test = df[1460:]
    num_cols = train.select_dtypes(include=['integer', 'float'])
    num_cols = [i for i in num_cols if i!= "SalePrice"]
    lr = LinearRegression()
    lr.fit(train[num_cols], train["SalePrice"])
    predictions = lr.predict(test[num_cols])
    mse = mean_squared_error(predictions,test["SalePrice"])
    rmse = mse**0.5
    return rmse
train_and_test(houses[["Gr Liv Area", "SalePrice"]])


# In[5]:


years_sold = houses['Yr Sold'] - houses['Year Built']
years_since_remod = houses['Yr Sold'] - houses['Year Remod/Add']
houses['Years Before Sale'] = years_sold
houses['Years Since Remod'] = years_since_remod
houses = houses[houses["Years Before Sale"]>=0] 
houses = houses[houses["Years Since Remod"]>=0]
houses


# In[6]:


houses = houses.drop(["PID", "Order","Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)
houses


# In[7]:


# def train_and_test(df):
#     train = df[:1460]
#     test = df[1460:]
#     num_cols = train.select_dtypes(include=['integer', 'float'])
#     num_cols = [i for i in num_cols if i!= "SalePrice"]
#     lr = LinearRegression()
#     lr.fit(train[num_cols], train["SalePrice"])
#     predictions = lr.predict(test[num_cols])
#     mse = mean_squared_error(predictions,test["SalePrice"])
#     rmse = mse**0.5
#     return rmse
train_and_test(houses[["Gr Liv Area", "SalePrice"]])


# In[8]:


get_ipython().magic('matplotlib inline')
numerical_houses = houses.select_dtypes(include=['integer', 'float'])
corr_df = numerical_houses.corr()
sns.heatmap(corr_df)
corr_df


# In[9]:


best_feats = abs(corr_df["SalePrice"][abs(corr_df["SalePrice"]) > 0.4])
best_feats


# In[10]:


# value_count_dic = { }
# for i in numerical_houses.columns:
#     value_count_dic[i] = numerical_houses[i].value_counts()

def select_features(df):
    cols = best_feats.index
    return df[cols]

data = select_features(houses)
train_and_test(data)


# In[11]:


nominal_features = [ "MS SubClass", "MS Zoning", "Street", "Land Contour", "Lot Config", "Neighborhood", 
                    "Condition 1", "Condition 2", "Bldg Type", "House Style", "Roof Style", "Roof Matl", "Exterior 1st", 
                    "Exterior 2nd", "Mas Vnr Type", "Foundation", "Heating", "Central Air"]
dic = {}
for i in nominal_features:
    dic[i] = houses[i].value_counts()
dic


# In[12]:


houses[nominal_features].isnull().sum()
houses["Mas Vnr Type"] = houses["Mas Vnr Type"].fillna("None")
houses[nominal_features].isnull().sum()


# In[13]:


for i in nominal_features:
    houses[i] = houses[i].astype("category")

for col in nominal_features:
    dummy = pd.get_dummies(houses[col])
    houses = pd.concat([houses,dummy], axis = 1)
houses = houses.drop(nominal_features,axis =  "columns")
houses.shape


# In[14]:


from sklearn.model_selection import KFold
def train_and_test_final(df,k):
    lr = LinearRegression()
    num_cols = df.select_dtypes(include=['integer', 'float'])
    num_cols = [i for i in num_cols if i!= "SalePrice"]

    if k==0:
        train = df[:1460]
        test = df[1460:]
        lr.fit(train[num_cols], train["SalePrice"])
        predictions = lr.predict(test[num_cols])
        mse = mean_squared_error(predictions,test["SalePrice"])
        rmse = mse**0.5
    if k==1:
        df = df.sample(frac = 1)
        fold_one = df[:1460]
        fold_two = df[1460:]
        lr.fit(fold_one[num_cols], fold_one["SalePrice"])
        predictions_one = lr.predict(fold_two[num_cols])
        mse_one = mean_squared_error(fold_two["SalePrice"], predictions_one)
        rmse_one = np.sqrt(mse_one)
        
        lr.fit(fold_two[features], fold_two["SalePrice"])
        predictions_two = lr.predict(fold_one[features])        
       
        mse_two = mean_squared_error(fold_one["SalePrice"], predictions_two)
        rmse_two = np.sqrt(mse_two)
        
        avg_rmse = np.mean([rmse_one, rmse_two])
        print(rmse_one)
        print(rmse_two)
        return avg_rmse
    else:
        kf = KFold(n_splits=k, shuffle=True)
        rmse_values = []
        for train_index, test_index, in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            lr.fit(train[num_cols], train["SalePrice"])
            predictions = lr.predict(test[num_cols])
            mse = mean_squared_error(test["SalePrice"], predictions)
            rmse = np.sqrt(mse)
            rmse_values.append(rmse)
        print(rmse_values)
        avg_rmse = np.mean(rmse_values)
        return avg_rmse


# In[15]:


train_and_test_final(houses,4)

