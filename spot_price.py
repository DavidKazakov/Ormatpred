#!/usr/bin/env python
# coding: utf-8

# In[29]:


#%pip install neuralprophet


# In[30]:


import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('neuralprophet').disabled = True


# # Blob storage config

# In[31]:


from azure.storage.blob import BlobServiceClient

account_name = 'ormatprdstorage1'
account_key = '9n99wAvTcTVBVoANyf8SHJ9cG/VRmA1C2umiyPbHOXb8Bhs578oKQxeK1Sl1DHCVYhTWH+cmNVpPuC1+7EFo8Q==' #Renew in the end of 2024
connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = 'azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec'
blob_storage_path = 'ML_Reasults'


# # Select Specific Plant

# In[32]:


class SelectPlantTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,plant):
        self.plant=plant

    def fit(self, X):  #
        return self

    def select_plant(self, df):
        try:
            df=df[df['Location']==self.plant]
         
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.select_plant(X)
        return X


# # Data Preparation

# In[33]:


class DataPreparationTransformer(BaseEstimator, TransformerMixin):
   
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def prepare_data(self, df):
        try:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['date'] = pd.to_datetime(df['date'].dt.date)
            df['DateTime'] = df['date'] + pd.to_timedelta(df['hour'].astype(str) + ':00:00')

            df = df.rename(columns = {"DateTime": "ds","Value": "y"})

            df = df[["ds","y"]]

            df = df.dropna()
 
            df = df.set_index('ds')
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df = df.asfreq('h')
            df['y'] = df['y'].bfill().ffill()
            df = df.reset_index()
            df = df.sort_index()
        
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.prepare_data(X)
        return X


# # Get last x years of data

# In[34]:


class HistoryYearsTransformer(BaseEstimator, TransformerMixin):
   #######
    def __init__(self,years='',months=0):
        self.years = years
        self.months = months
 
    def fit(self, X):  #
        return self
 
 
    def get_last_x_years(self, df):
        try:
            if self.years!='':    
                latest_date = df['ds'].max()
                start_date = latest_date - pd.DateOffset(years=self.years,months=self.months)           
                df = df[df['ds'] >= start_date]
        except Exception as ex:
            print(ex)
        return df
 
    def transform(self, X):
        X = self.get_last_x_years(X)
        return X


# # Number Of Hours To Predict

# In[35]:


class HoursToPredictTransformer(BaseEstimator, TransformerMixin):
   
    def __init__(self, pred_hours=''):
        self.pred_hours = pred_hours

    def fit(self, X):
        return self

    def split_data(self, df):
        try:
            if self.pred_hours != '':
                df = df[:-int(self.pred_hours)]
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.split_data(X)
        return X


# # Split Data Per Hour

# In[36]:


class HoursTransformer(BaseEstimator, TransformerMixin):
   
    def __init__(self, i=0):
        self.i = i

    def fit(self, X):
        return self

    def split_data(self, df):
        try:
            if self.i!=0:
                df['hour'] = df['ds'].dt.hour
                df = df[df['hour']==self.i][['ds','y']]
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.split_data(X)
        return X


# # Load Data

# In[37]:


df = pd.read_csv('https://ormatprdstorage1.blob.core.windows.net/azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec/Azureml_Generation/SpotPrice.csv?sp=rcw&st=2023-11-28T07:46:25Z&se=2024-12-31T15:46:25Z&spr=https&sv=2022-11-02&sr=c&sig=ZrQv6iSFtEFeKuZs8S1WhGBoQPRjmH7SJI%2BgeRxIu5Q%3D')


# # Plants List

# In[38]:


plants_dict = {'BRADY':'Brady',
               'DAC 1':'Don Campbell 1',
               'DAC 2':'Don Campbell 2', 
               'TUNGSTEN':'Tungsten',  
               'GALENA 2':'Galena2', 
               'McGINNESS 3':'MGH3',
               'STEAMBOAT':'SBHR', 
               'STEAMBOAT 2-3':'SB2-3'}


# # Make predictions for next 38 hours and save to blob storage

# In[ ]:


import logging
import shutil
import os

# Set the logging level to WARNING or ERROR to reduce output
logging.getLogger("lightning").setLevel(logging.WARNING)  # Suppress INFO logs
logging.getLogger("NP").setLevel(logging.WARNING)  # Suppress Neural Prophet logs

for plant in plants_dict.keys():
    
    spot_price = SelectPlantTransformer(plant).transform(df)
    spot_price = DataPreparationTransformer().transform(spot_price)
    spot_price = HistoryYearsTransformer(years=1,months=0).transform(spot_price)

    model = NeuralProphet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=True,
    n_lags= 6*38,
    ar_layers= [32,32,32],
    learning_rate=0.005,
    n_forecasts=38,
    epochs=94,
    batch_size=64
    )
    
    model.fit(spot_price)
    
    
    future = model.make_future_dataframe(spot_price, periods=38)
    forecast = model.predict(future)
    forecast = model.get_latest_forecast(forecast)
    
    dates = forecast['ds'].values
    
    results = pd.DataFrame({
        'DateTime': dates,
        'Plant': plants_dict[plant],
        'Temp': 'null',
        'Spot_price': forecast['origin-0'].values
    })

    # Convert 00 to 24
    results['DateTime'] = results['DateTime'].apply(lambda x: x - pd.DateOffset(days=1) if x.hour == 0 else x)
    results['DateTime'] = results['DateTime'].dt.strftime('%Y-%m-%d %H:00:00').str.replace(' 00:', ' 24:')
    
    csv_data = results.to_csv(index=False)
    # Specify the blob name and upload the CSV data to the container:
    blob_name = f"{blob_storage_path}/spot_price_{plants_dict[plant]}.csv"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(csv_data, overwrite=True)

    # Clean lighting_logs directory to reduce memory

    def cleanup_logs(log_dir):
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

    cleanup_logs('./lightning_logs/')

