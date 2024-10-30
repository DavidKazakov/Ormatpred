#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np 
import pandas as pd 
from pandas.tseries.offsets import DateOffset
pd.options.mode.chained_assignment = None
from neuralprophet import NeuralProphet
#from prophet import Prophet
import logging
logging.getLogger('neuralprophet').disabled = True
import warnings
warnings.simplefilter("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
# models
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor


# # Blob storage config

# In[48]:


from azure.storage.blob import BlobServiceClient

account_name = 'ormatprdstorage1'
account_key = '9n99wAvTcTVBVoANyf8SHJ9cG/VRmA1C2umiyPbHOXb8Bhs578oKQxeK1Sl1DHCVYhTWH+cmNVpPuC1+7EFo8Q==' #Renew in the end of 2024
connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = 'azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec'
blob_storage_path = 'ML_Reasults'


# # Transformers

# # Select Specific Plant

# In[49]:


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

# In[50]:


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

# In[51]:


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


#  # Add date attributes

# In[52]:


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


# # Clean Data

# In[53]:


class CleaningTransformer(BaseEstimator, TransformerMixin):
   #######
    def __init__(self):
        pass

    def fit(self, X):  #
        return self


    def clean_data(self, df):
        try:
            df.drop(df[df['Temp']<0].index, inplace = True)
            df.drop(df[df['Temp']>200].index, inplace = True)
            df.dropna(subset = ['Temp'], inplace=True)

            #print('Cleaned successfully')
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.clean_data(X)
        return X


# # Dummy Replace values to dummies (get_dummies)

# In[54]:


class DummyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
         return self

    def dummy(self, df):
        df=df[['Temp','Solar','hour','weekday','day_part','season','weekend', 'Solar_1', 'Solar_2', 'Solar_3', 'Solar_4', 'Solar_5']]
        df=pd.get_dummies(df,columns = ['hour','weekday','day_part','season','weekend'])
        return df

    def transform(self, X):
        X = self.dummy(X)
        #print('Get Dummies successfully')
        return X


# # Weather data arrangement

# In[55]:


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


# # Neural Prophet data preparation

# In[56]:


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


# # Prediction data preparation

# In[57]:


class DaysForPredTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,last_date,days):
        self.last_date=last_date
        self.days = days

    def fit(self, X):
        return self

    def days_for_pred(self, df):
        try:
            # interpolation fill empty values
            df['Temp'] = df['Temp'].interpolate(method='linear', limit_direction='forward')            
            # Take x next days data for prediction
            target_date_range = self.last_date + pd.DateOffset(days=self.days)
            
            df = df[(df['DateTime'] > last_date) & (df['DateTime'] <= target_date_range)]
            #df = df[(df['DateTime'].dt.hour >= 5) & (df['DateTime'].dt.hour <= 19)]
        except Exception as ex:
            print(ex)

        return df

    def transform(self, X):
        X = self.days_for_pred(X)
        return X


# # Loading the Dataset

# In[58]:


solar_df = pd.read_csv('https://ormatprdstorage1.blob.core.windows.net/azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec/Azureml_Generation/Generation.csv?sp=rcw&st=2023-11-28T07:46:25Z&se=2024-12-31T15:46:25Z&spr=https&sv=2022-11-02&sr=c&sig=ZrQv6iSFtEFeKuZs8S1WhGBoQPRjmH7SJI%2BgeRxIu5Q%3D')


# In[59]:


solar_df['Solar'] = solar_df['Solar'].fillna(0)


# # Date Column to DateTime

# In[60]:


solar_df['DateTime']= pd.to_datetime(solar_df['DateTime'], format='%Y-%m-%d %H:%M:%S')


# # Load weather data

# In[61]:


weather_forecast_data = pd.read_csv('https://ormatprdstorage1.blob.core.windows.net/azureml-blobstore-5930a3cf-9d2a-4c61-8e7c-3e55e29484ec/Azureml_Generation/Forecast.csv?sp=rcw&st=2023-11-28T07:46:25Z&se=2024-12-31T15:46:25Z&spr=https&sv=2022-11-02&sr=c&sig=ZrQv6iSFtEFeKuZs8S1WhGBoQPRjmH7SJI%2BgeRxIu5Q%3D')


# # Plants list

# In[63]:


plant_dict = {'Brady':'BRADY',
              'Galena2':'GALENA 2',
              'SBHR':'SBHills',
              'SB2-3':'STEAMBOAT 2-3',
              'Tungsten':'TUNGSTEN'}


# # Neural Prophet

# In[ ]:


for plant in plant_dict.keys():

    solar = SelectPlantTransformer(plant).transform(solar_df)
    solar = LastXDaysTransformer(90).transform(solar)
    solar = NPDataArrangeTransformer().transform(solar)

    # Initialize NeuralProphet model
    model = NeuralProphet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=True,
            n_lags=5 * 24,
            ar_layers=[32, 32, 32],
            learning_rate=0.005,
            n_forecasts=48)

    model.add_seasonality(name="zero", period=24, fourier_order=3, condition_name="zero")
    model.add_seasonality(name="value", period=24, fourier_order=3, condition_name="value")

    model.fit(solar)
    
    future = model.make_future_dataframe(solar, periods=48)
    future["zero"] = future["ds"].apply(lambda x: x.hour not in [7,8,9,10,11,12,13,14,15,16])
    future["value"] = future["ds"].apply(lambda x: x.hour in [7,8,9,10,11,12,13,14,15,16])
    forecast = model.predict(future)
    forecast = model.get_latest_forecast(forecast)
    # Set value to zero at hours without solar generation
    forecast['origin-0'] = np.where(forecast['origin-0'] < 0.7, 0, forecast['origin-0'])
    
    dates = forecast['ds'].values
    
    results = pd.DataFrame({
        'DateTime': dates,
        'Plant': plant,
        'Temp': 'null',
        'Power': forecast['origin-0'].values
    })
    
    # Convert 00 to 24
    results['DateTime'] = results['DateTime'].apply(lambda x: x - pd.DateOffset(days=1) if x.hour == 0 else x)
    results['DateTime'] = results['DateTime'].dt.strftime('%Y-%m-%d %H:00:00').str.replace(' 00:', ' 24:')
    
    csv_data = results.to_csv(index=False)
    # Specify the blob name and upload the CSV data to the container:
    blob_name = f"{blob_storage_path}/solar_np_{plant}.csv"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(csv_data, overwrite=True)


# # ExtraTreesRegressor & RandomForestRegressor

# In[65]:


for plant in plant_dict.keys():
    # Select plant
    solar = SelectPlantTransformer(plant).transform(solar_df)
    # Add previous days 
    solar = PreviousDaysTransformer(5).transform(solar)
    # get x last days
    solar = LastXDaysTransformer(45).transform(solar)
    # Clean
    solar = CleaningTransformer().transform(solar)
    solar = solar.sort_values(['DateTime']).reset_index(drop=True)
    # Get last date for prediction
    last_date = solar['DateTime'].max()
    # Weather forcast data arrangement
    weather_forecast =  WeatherDataArrangeTransformer().transform(weather_forecast_data)    
    # Select plant in weather data
    weather_forecast = SelectPlantTransformer(plant_dict[plant]).transform(weather_forecast)
    # Prepare data for prediction
    weather_forecast = DaysForPredTransformer(last_date,days=2).transform(weather_forecast)
   
    # Combine original data with prediction data
    all_data = pd.concat([solar, weather_forecast], ignore_index=True)
    
    # Date Arrangement
    all_data = DateTransformer().transform(all_data)
    # Save date column for output 
    dates = all_data['DateTime']

    # Add Dummy
    all_data = DummyTransformer().transform(all_data)
    
    train_data = all_data.iloc[:-48]
    test_data = all_data.iloc[-48:]

    # Train test arrangement
    y_train=train_data['Solar'] 
    X_train=train_data.drop('Solar', axis=1)

    y_test=test_data['Solar'] 
    X_test=test_data.drop('Solar', axis=1)
    
    models = [ExtraTreesRegressor(n_jobs=-1, random_state=159),RandomForestRegressor(n_jobs=-1, random_state=159)]
    m_names = ['ExtraTrees', 'RandomForest']
    for m_name, model in enumerate (models): 
        model = model.fit(X_train, y_train)
        # Loop over 2 days and predict
        for i in range(2):
            # Calculate the start and end indices for the current loop
            start_idx = i * 24
            end_idx = (i + 1) * 24  
            # Get the data for the current loop
            current_day = test_data.iloc[start_idx:end_idx]   
            previous_day = all_data.iloc[current_day.index - 24]
            previous_day = previous_day.reset_index(drop=True)
            ind = current_day.index.values
            current_day = current_day.reset_index(drop=True)
            current_day[current_day.columns[2:7]] = previous_day[previous_day.columns[1:6]]
            current_day['Solar']  = model.predict(current_day.drop('Solar', axis=1))
            current_day = current_day.set_index(ind)
            all_data.iloc[ind] = current_day
        # Organize predictions columns
        all_data_copy = all_data.copy()
        all_data_copy['DateTime'] = dates
        all_data_copy['Plant'] = plant
        
        predictions = all_data_copy[['DateTime','Plant','Temp','Solar']].rename(columns={'Solar':'Power'}).tail(48).reset_index(drop=True)
        predictions['Power'] = predictions.apply(lambda x: x['Power'] if 7 <= x['DateTime'].hour <= 16 else 0, axis=1)
        # Convert 00 to 24
        predictions['DateTime'] = predictions['DateTime'].apply(lambda x: x - pd.DateOffset(days=1) if x.hour == 0 else x)
        predictions['DateTime'] = predictions['DateTime'].dt.strftime('%Y-%m-%d %H:00:00').str.replace(' 00:', ' 24:')
        
        # save predictions to csv 
        csv_data = predictions.to_csv(index=False)
        
        # Specify the blob name and upload the CSV data to the container:
        blob_name = f"{blob_storage_path}/solar_{m_names[m_name]}_{plant}.csv"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(csv_data, overwrite=True)
        print("saved successfuly!")


# In[ ]:




