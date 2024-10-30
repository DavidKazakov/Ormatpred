#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install cmdstanpy>=1.0.4
#%pip install  prophet


# In[2]:


import numpy as np 
import pandas as pd 
from pandas.tseries.offsets import DateOffset
from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# # Blob storage config

# In[3]:


from azure.storage.blob import BlobServiceClient

account_name = 'ormatprdstorage1'
account_key = '9n99wAvTcTVBVoANyf8SHJ9cG/VRmA1C2umiyPbHOXb8Bhs578oKQxeK1Sl1DHCVYhTWH+cmNVpPuC1+7EFo8Q==' #Renew in the end of 2024
connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = 'azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec'
blob_storage_path = 'ML_Reasults'


# # Transformers

# # Select Specific Plant

# In[4]:


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


# # Get last x days

# In[5]:


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


# # Neural Prophet data preparation

# In[6]:


class NPDataArrangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def data_arrangement(self, df):
        try:
            df = df.sort_values(['DateTime']).reset_index(drop=True)
            df = df[['DateTime','Solar']].rename(columns = {"DateTime": "ds","Solar": "y"})
            df = df.set_index('ds')
            df = df.sort_index()
            df = df.asfreq('h')
            df['y'] = df['y'].bfill().ffill()
            df = df.reset_index()
            df = df.sort_index()

            df["zero"] = df["ds"].apply(lambda x: x.hour not in [7,8,9,10,11,12,13,14,15,16])
            df["value"] = df["ds"].apply(lambda x: x.hour in [7,8,9,10,11,12,13,14,15,16])
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.data_arrangement(X)
        return X


# # Loading the Dataset

# In[7]:


solar_df = pd.read_csv('https://ormatprdstorage1.blob.core.windows.net/azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec/Azureml_Generation/Generation.csv?sp=rcw&st=2023-11-28T07:46:25Z&se=2024-12-31T15:46:25Z&spr=https&sv=2022-11-02&sr=c&sig=ZrQv6iSFtEFeKuZs8S1WhGBoQPRjmH7SJI%2BgeRxIu5Q%3D')


# In[8]:


solar_df['Solar'] = solar_df['Solar'].fillna(0)


# # Date Column to DateTime

# In[9]:


solar_df['DateTime']= pd.to_datetime(solar_df['DateTime'], format='%Y-%m-%d %H:%M:%S')


# # Plants list

# In[10]:


plant_dict_prophet = {'Brady':'BRADY',
              'Galena2':'GALENA 2',
              'SBHR':'SBHills',
              'SB2-3':'STEAMBOAT 2-3',
              'Tungsten':'TUNGSTEN',
              'Woods Hill':'WOODS HILL',
              'Wister Solar':'Wister Solar'}


# # Prophet

# In[11]:


for plant in plant_dict_prophet.keys():

    solar = SelectPlantTransformer(plant).transform(solar_df)
    solar = LastXDaysTransformer(240).transform(solar)
    solar = NPDataArrangeTransformer().transform(solar)
    solar = solar[['ds','y']]
    
    _default_params = {"growth": "flat", "yearly_seasonality": False, "scaling": "minmax"}

    model = Prophet(**_default_params)
    model.fit(solar)
    future = model.make_future_dataframe(periods=48, freq='H')
    pred= model.predict(future)
    pred = pred.tail(48)
    
    dates = pred['ds'].values
    
    results = pd.DataFrame({
        'DateTime': dates,
        'Plant': plant,
        'Temp': 'null',
        'Power': pred['yhat'].values
    })
    
    # Convert 00 to 24
    results['DateTime'] = results['DateTime'].apply(lambda x: x - pd.DateOffset(days=1) if x.hour == 0 else x)
    results['DateTime'] = results['DateTime'].dt.strftime('%Y-%m-%d %H:00:00').str.replace(' 00:', ' 24:')
    
    csv_data = results.to_csv(index=False)
    # Specify the blob name and upload the CSV data to the container:
    blob_name = f"{blob_storage_path}/solar_prophet_{plant}.csv"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(csv_data, overwrite=True)
    print(plant)


# In[ ]:




