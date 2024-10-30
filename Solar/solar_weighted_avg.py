#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np 
import pandas as pd 
import csv
import pickle
import os
import matplotlib.pyplot as plt
#import seaborn as sns
from pandas.tseries.offsets import DateOffset
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
# models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
#from lightgbm import LGBMRegressor


# # Blob storage config

# In[5]:


from azure.storage.blob import BlobServiceClient

account_name = 'ormatprdstorage1'
account_key = '9n99wAvTcTVBVoANyf8SHJ9cG/VRmA1C2umiyPbHOXb8Bhs578oKQxeK1Sl1DHCVYhTWH+cmNVpPuC1+7EFo8Q==' #Renew in the end of 2024
connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = 'azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec'
blob_storage_path = 'ML_Reasults'


# # Transformers

# # Select Specific Plant

# In[6]:


class SelectPlantTransformer(BaseEstimator, TransformerMixin):
   #######
    def __init__(self,plant):
        self.plant=plant

    def fit(self, X):  #
        return self


    def select_plant(self, df):
        try:
            #df.drop(df[df['Plant']!=self.plant].index, inplace = True)
            df=df[df['Plant']==self.plant]

            #print('Select plant: ' + self.plant)
         
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.select_plant(X)
        return X


# # Add previous n days 

# In[7]:


class PreviousDaysTransformer(BaseEstimator, TransformerMixin):
   #######
    def __init__(self, num_of_days=5):
        self.num_of_days = num_of_days

    def fit(self, X):  #
        return self
    

    def add_days(self, df):
        
        def previous_day(df, i):
            previous = pd.merge(df, df, left_on=f'prev_{i}', right_on=df.index, right_index=True, how='left')
            previous.columns = previous.columns.str.rstrip('_x')
            previous = previous.rename(columns={'Solar_y':f'Solar_{i}'})
            previous = previous.drop([col for col in previous.columns if '_y' in col],axis=1)
            return previous
        
        try:
            df = df.set_index('DateTime')
            for i in range(1,self.num_of_days+1):
                df[f'prev_{i}'] = df.index - pd.DateOffset(days=i)
                
            previous_days = [previous_day(df, 1)]
            
            for i in range(2, self.num_of_days+1):
                previous_days.append(previous_day(previous_days[i-2], i))
                 
            
            df = previous_days[-1].drop([f'prev_{i}' for i in range(1,self.num_of_days+1)],axis=1)

            # back fill NaN values
            df[[f'Solar_{i}' for i in range(1,self.num_of_days+1)]] = df[[f'Solar_{i}' for i in range(1,self.num_of_days+1)]].bfill()
            df = df.reset_index()
            del previous_days
            # print(f'previous {self.num_of_days} days added successfully')
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.add_days(X)
        return X


# # Get last x days

# In[8]:


class LastXDaysTransformer(BaseEstimator, TransformerMixin):
   #######
    def __init__(self,num_of_days=''):
        self.num_of_days=num_of_days
 
    def fit(self, X):  #
        return self
 
 
    def get_last_x_days(self, df):
        try:
            if self.num_of_days!='':    
                if 'DateTime' in df.columns:
                    df = df.set_index('DateTime')
                df = df[df.last_valid_index()-pd.DateOffset(self.num_of_days, 'D'):]
                df = df.reset_index()
        except Exception as ex:
            print(ex)
        return df
 
    def transform(self, X):
        X = self.get_last_x_days(X)
        return X


# # Loading the Dataset

# In[9]:


DAC1 = pd.read_csv('https://ormatprdstorage1.blob.core.windows.net/azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec/Azureml_Generation/Generation.csv?sp=rcw&st=2023-11-28T07:46:25Z&se=2024-12-31T15:46:25Z&spr=https&sv=2022-11-02&sr=c&sig=ZrQv6iSFtEFeKuZs8S1WhGBoQPRjmH7SJI%2BgeRxIu5Q%3D')


# In[10]:


DAC1 = DAC1[['Plant', 'DateTime', 'Temp', 'Solar']]
DAC1['Solar'] = DAC1['Solar'].fillna(0)


# # Date Column to DateTime

# In[11]:


DAC1['DateTime']= pd.to_datetime(DAC1['DateTime'], format='%Y-%m-%d %H:%M:%S')


# # Plants list

# In[28]:


plants = ['Brady','Galena2','SBHR','Tungsten','SB2-3','Wister Solar','Woods Hill']


# # Weights function

# In[13]:


def weight(n):
    return list(range(n, 0, -1))


# In[14]:


weight(5)


# # Fibonacci weights

# In[15]:


def fibonacci(n):
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib[::-1]


# # Calculate weigted avg for each plant

# In[30]:


for plant in plants:
    # Run 2 times- one with fibonacci weights and one without
    for f in range(2):
        if f == 0:
            weights = fibonacci(5)
            n = 'Fibonacci_'
        else:
            weights = weight(5)
            n= ''
            
        # Select plant
        generation = SelectPlantTransformer(plant).transform(DAC1)
        # Add previous n days 
        generation = PreviousDaysTransformer(5).transform(generation)
        # get x last days
        generation = LastXDaysTransformer(90).transform(generation)
        # Calculate weighted average 
        generation.loc[:, 'w_avg'] = np.dot(generation.loc[:, 'Solar_1':'Solar_5'], weights) / sum(weights)

        last_date = generation['DateTime'].max()
        target_date_range = last_date + pd.DateOffset(days=3)
        prediction_days = pd.DataFrame()
        prediction_days['DateTime'] = pd.date_range(last_date+pd.Timedelta(hours=1),target_date_range,freq='H') 

        # Combine original data with prediction data
        all_data = pd.concat([generation, prediction_days], ignore_index=True)

        predict = all_data.tail(72)

        # Get the data for the next day
        current_day = predict.iloc[:24]   
        previous_day = all_data.iloc[current_day.index - 24]
        previous_day = previous_day.reset_index(drop=True)
        ind = current_day.index.values
        current_day = current_day.reset_index(drop=True)
        current_day[current_day.columns[4:9]] = previous_day[previous_day.columns[3:8]]
        current_day['w_avg'] = np.dot(current_day.loc[:, 'Solar_1':'Solar_5'], weights) / sum(weights)
        current_day = current_day.set_index(ind)
        
        # copy data of next day predictions to next 2 days
        for i in range(3):
            all_data['w_avg'].iloc[ind+(i*24)] = current_day['w_avg']
        predictions = all_data[['DateTime', 'Plant', 'Temp', 'w_avg']].rename(columns={'w_avg':'Power'}).tail(72)
        # Convert 00 to 24
        predictions['DateTime'] = predictions['DateTime'].apply(lambda x: x - pd.DateOffset(days=1) if x.hour == 0 else x)
        predictions['DateTime'] = predictions['DateTime'].dt.strftime('%Y-%m-%d %H:00:00').str.replace(' 00:', ' 24:')
        
        predictions['Plant'] = plant
        #predictions = predictions[(predictions['DateTime'].dt.hour >= 8) & (predictions['DateTime'].dt.hour <= 16)]
        
        # save predictions to csv 
        csv_data = predictions.to_csv(index=False)
        # Specify the blob name and upload the CSV data to the container:
        blob_name = f"{blob_storage_path}/{n}w_avg_5_days_solar_{plant}.csv"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(csv_data, overwrite=True)


# # Create RMSE function

# In[35]:


def RMSE(y_pred, y_test):
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    print(f"RMSE = {rmse:.2f}")


# In[ ]:




