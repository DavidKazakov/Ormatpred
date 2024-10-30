#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import csv
import pickle
import os
from pandas.tseries.offsets import DateOffset
pd.options.mode.chained_assignment = None

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
# models
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


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


# # Fix Tungsten Power values

# In[5]:


class FixTungstenTransformer(BaseEstimator, TransformerMixin):
   #######
    def __init__(self,):
        pass

    def fit(self, X):  #
        return self


    def fix_values(self, df):
        try:
            filtered_df = df[df['DateTime'] >= '2023-09-20 07:00:00']
            filtered_df['Net_Power'] *= 1.5
            filtered_df['Power'] = np.where(filtered_df['Solar'].notna(), filtered_df['Net_Power'] - filtered_df['Solar'], filtered_df['Net_Power'])

            df.update(filtered_df)

        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.fix_values(X)
        return X


# # Add previous n days 

# In[20]:


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
            previous = previous.rename(columns={'Power_y':f'Net_Power_{i}', 'Temp_y':f'Temp_{i}'})
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
            df[[f'Net_Power_{i}' for i in range(1,self.num_of_days+1)]] = df[[f'Net_Power_{i}' for i in range(1,self.num_of_days+1)]].bfill().ffill()
            df[[f'Temp_{i}' for i in range(1,self.num_of_days+1)]] = df[[f'Temp_{i}' for i in range(1,self.num_of_days+1)]].bfill().ffill()
            
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

# In[7]:


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


# # Drop Outliers

# In[8]:


class DropOutliersTransformer(BaseEstimator, TransformerMixin):
   #######
    def __init__(self):        
        pass
 
    def fit(self, X):  #
        return self
 
 
    def drop_outliers(self, df):
        try:
            df['year'] = df['DateTime'].dt.year
            df['month'] = df['DateTime'].dt.month
            monthly_avg = df.groupby(['year', 'month'])['Net_Power'].mean().reset_index()
            monthly_avg = monthly_avg[['year', 'month', 'Net_Power']]
            df = df.merge(monthly_avg, on=['year', 'month'], how='left')
            df = df[df['Net_Power_x']>=(df['Net_Power_y']/3)]

        except Exception as ex:
            print(ex)
        return df
 
    def transform(self, X):
        X = self.drop_outliers(X)
        return X


# # Clean Data
# **Clean data where power >90 or <-90 or ==0  or null values**
# **temp >500 or fare_amount <0**

# In[9]:


class CleaningTransformer(BaseEstimator, TransformerMixin):
   #######
    def __init__(self):
        pass

    def fit(self, X):  #
        return self


    def clean_data(self, df):
        try:
            df.drop(df[df['Temp']<=0].index, inplace = True)
            df.drop(df[df['Temp']>200].index, inplace = True)
            df.drop(df[df['Power']<=1].index, inplace = True)
            df.drop(df[df['Power']>500].index, inplace = True)
            #df = df.drop(['DateTime','Value'], axis = 1)
            df.dropna(subset = ['Power'], inplace=True)
            df.dropna(subset = ['Temp'], inplace=True)

            #print('Cleaned successfully')
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.clean_data(X)
        return X


#  # Add date attributes

# In[10]:


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def add_date_attributes(self, df):
        try:
            df['DateTime']=pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')
            df['year'] = pd.DatetimeIndex(df['DateTime']).year
            df['month'] = pd.DatetimeIndex(df['DateTime']).month
            df['day'] = pd.DatetimeIndex(df['DateTime']).day
            df['hour'] = pd.DatetimeIndex(df['DateTime']).hour
            df['weekday'] = pd.DatetimeIndex(df['DateTime']).weekday
            df['day_name'] = pd.DatetimeIndex(df['DateTime']).day_name()

            df['day_part'] = pd.cut(df['hour'],[-1,6,16,25] , labels=['2','1', '3'] )## 
            df['day_part']=df['day_part'].str.replace('3', '2')
            df['season']=pd.cut(df['month'], [0 ,2,5,9 ,11,12] , labels=['0','1','2','3','4'] )
            df['season']=df['season'].str.replace('0', '4')
            df['weekend']=pd.cut(df['weekday'], [-1,4,6] , labels=['1','2'] )

           # df['retio']= df['Power']/df['Temp']
           # df['std']=df['retio'].std()

            #print('Date attributes add successfully')
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.add_date_attributes(X)
        return X


# # Dummy Replace values to dummies (get_dummies)

# In[11]:


class DummyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
         return self

    def dummy(self, df):
        df=df[['Power','Temp','hour','weekday','day_part','season','weekend', 'Net_Power_1', 'Temp_1', 'Net_Power_2', 'Temp_2', 'Net_Power_3', 'Temp_3', 'Net_Power_4', 'Temp_4', 'Net_Power_5', 'Temp_5']]
        df=pd.get_dummies(df,columns = ['hour','weekday','day_part','season','weekend'])
        return df

    def transform(self, X):
        X = self.dummy(X)
        #print('Get Dummies successfully')
        return X


# # Weather data arrangement

# In[12]:


class WeatherDataArrangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def data_arrangement(self, df):
        try:
            df = df[df['Hour'] != 25]
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            df['DateTime'] = df['Date'] + pd.to_timedelta(df['Hour'].astype(str) + ':00:00')
            df = df[['DateTime', 'Location', 'Value']].rename(columns={'Value':'Temp', 'Location':'Plant'})
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.data_arrangement(X)
        return X


# # Prediction data preparation

# In[13]:


class DaysForPredTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,last_date):
        self.last_date=last_date

    def fit(self, X):
        return self

    def days_for_pred(self, df):
        try:
            # interpolation fill empty values
            df['Temp'] = df['Temp'].interpolate(method='linear', limit_direction='forward')            
            # Take 3 next days data for prediction
            target_date_range = self.last_date + pd.DateOffset(days=3)
            df = df[(df['DateTime'] > last_date) & (df['DateTime'] <= target_date_range)]
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.days_for_pred(X)
        return X


# # Loading the Dataset

# In[14]:


DAC1 = pd.read_csv('https://ormatprdstorage1.blob.core.windows.net/azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec/Azureml_Generation/Generation.csv?sp=rcw&st=2023-11-28T07:46:25Z&se=2024-12-31T15:46:25Z&spr=https&sv=2022-11-02&sr=c&sig=ZrQv6iSFtEFeKuZs8S1WhGBoQPRjmH7SJI%2BgeRxIu5Q%3D')


# In[15]:


DAC1.head()


# # Date Column to DateTime

# In[16]:


DAC1['DateTime']= pd.to_datetime(DAC1['DateTime'], format='%Y-%m-%d %H:%M:%S')


# # Load weather data

# In[17]:


weather_forecast_data = pd.read_csv('https://ormatprdstorage1.blob.core.windows.net/azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec/Azureml_Generation/Forecast.csv?sp=rcw&st=2023-11-28T07:46:25Z&se=2024-12-31T15:46:25Z&spr=https&sv=2022-11-02&sr=c&sig=ZrQv6iSFtEFeKuZs8S1WhGBoQPRjmH7SJI%2BgeRxIu5Q%3D')


# # Plants list

# In[18]:


plant_dict = {'Brady':'BRADY',
 'Don Campbell 1':'DAC',
 'Don Campbell 2':'DAC',
 'Galena2':'GALENA 2',
 'MGH3':'MGH1',
 'SBHR':'SBHills',
 'SB2-3':'SB2',
 'Tungsten':'TUNGSTEN'}


# # Make predictions and save results

# In[24]:


for plant in plant_dict.keys():
    # Select plant
    generation = SelectPlantTransformer(plant).transform(DAC1)
    if plant == 'Tungsten':
        generation = FixTungstenTransformer().transform(generation)
    # Add previous 5 days 
    generation = PreviousDaysTransformer().transform(generation)
    # get x last days
    generation = LastXDaysTransformer(45).transform(generation)
    # Drop Outliers
    generation = DropOutliersTransformer().transform(generation)
    # Clean
    generation = CleaningTransformer().transform(generation)
    
    generation = generation.sort_values(['DateTime']).reindex()
    # Get last date for prediction
    last_date = generation['DateTime'].max()
    # Weather forcast data arrangement
    weather_forecast =  WeatherDataArrangeTransformer().transform(weather_forecast_data)    
    # Select plant in weather data
    weather_forecast = SelectPlantTransformer(plant_dict[plant]).transform(weather_forecast)
    # Prepare data for prediction
    weather_forecast = DaysForPredTransformer(last_date).transform(weather_forecast)
   
    # Combine original data with prediction data
    all_data = pd.concat([generation, weather_forecast], ignore_index=True)
    
    if all_data.empty:
        predictions = all_data[['DateTime','Plant','Temp','Power']]
    else:
        # Date Arrangement
        all_data = DateTransformer().transform(all_data)
        # Save date column for output 
        dates = all_data['DateTime']
        # Convert 00 to 24
        dates = dates.apply(lambda x: x - pd.DateOffset(days=1) if x.hour == 0 else x)
        dates = dates.dt.strftime('%Y-%m-%d %H:00:00').str.replace(' 00:', ' 24:')
        
        # Add Dummy
        all_data = DummyTransformer().transform(all_data)

        train_data = all_data.iloc[:-72]
        test_data = all_data.iloc[-72:]

        # Train test arrangement
        y_train=train_data['Power'] 
        X_train=train_data.drop('Power', axis=1)

        y_test=test_data['Power'] 
        X_test=test_data.drop('Power', axis=1)

        # Linear Regression model
        LinearReg = LinearRegression().fit(X_train, y_train)
        # Loop over 3 days and predict
        for i in range(3):
            # Calculate the start and end indices for the current loop
            start_idx = i * 24
            end_idx = (i + 1) * 24  
            # Get the data for the current loop
            current_day = test_data.iloc[start_idx:end_idx]   
            previous_day = all_data.iloc[current_day.index - 24]
            previous_day = previous_day.reset_index(drop=True)
            ind = current_day.index.values
            current_day = current_day.reset_index(drop=True)
            current_day[current_day.columns[2:12]] = previous_day[previous_day.columns[:10]]
            current_day['Power']  = LinearReg.predict(current_day.drop('Power', axis=1))
            current_day = current_day.set_index(ind)
            all_data.iloc[ind] = current_day
        # Organize predictions columns
        all_data_copy = all_data.copy()
        all_data_copy['DateTime'] = dates
        
        all_data_copy['Plant'] = plant
        predictions = all_data_copy[['DateTime','Plant','Temp','Power']].tail(72).reset_index(drop=True)
    
    # save predictions to csv 
    csv_data = predictions.to_csv(index=False)
    # Specify the blob name and upload the CSV data to the container:
    blob_name = f"{blob_storage_path}/LR_results_{plant}.csv"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(csv_data, overwrite=True)
    print("saved successfuly!")
    
    if not all_data.empty:
        # KNN model
        Neigh = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
        # Loop over 3 days and predict
        for i in range(3):
            # Calculate the start and end indices for the current loop
            start_idx = i * 24
            end_idx = (i + 1) * 24  
            # Get the data for the current loop
            current_day = test_data.iloc[start_idx:end_idx]   
            previous_day = all_data.iloc[current_day.index - 24]
            previous_day = previous_day.reset_index(drop=True)
            ind = current_day.index.values
            current_day = current_day.reset_index(drop=True)
            current_day[current_day.columns[2:12]] = previous_day[previous_day.columns[:10]]
            current_day['Power']  = Neigh.predict(current_day.drop('Power', axis=1))
            current_day = current_day.set_index(ind)
            all_data.iloc[ind] = current_day

        # Organize predictions columns
        all_data['DateTime'] = dates
        all_data['Plant'] = plant
        predictions = all_data[['DateTime','Plant','Temp','Power']].tail(72).reset_index(drop=True)

    # save predictions to csv 
    csv_data = predictions.to_csv(index=False)
    # Specify the blob name and upload the CSV data to the container:
    blob_name = f"{blob_storage_path}/KNN5_results_{plant}.csv"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(csv_data, overwrite=True)
    print("saved successfuly!")
 


# In[ ]:




