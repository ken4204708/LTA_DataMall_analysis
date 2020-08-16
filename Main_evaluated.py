# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:19:44 2020

@author: ken78
"""


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.cluster import hierarchy

def get_union_sets_np(namelist):
    union_result = np.array([])
    for i in range(len(namelist)):
        union_result = np.union1d(union_result, namelist[i])
    return(union_result)
   
def plotSubplot(axes, df_data, xlabel, ylabel, title_str):
    df_data.plot(legend = False, ax = axes, title = title_str)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    
def data_preprocessing(df_data):
    x = df_data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def kmeans_algorithm(n_clusters, df_data_normalized, df_data, title_str):
    kmeans.fit(df_data_normalized.transpose())
    for num_cluster in range(num_clusters):
        fig = plt.figure()
        fig = plt.plot(pd_temp_weekday_in_raw.index, df_data.to_numpy()[:, np.where(kmeans.labels_ == num_cluster)][:, 0, :])
        plt.title('%s (Cluster #%s)' %(title_str, num_cluster + 1))


if __name__=="__main__":
    pd_train_PV = []
    monthes = [6]
    num_clusters = 4
    time_period = np.array([5,23])
    day_types = ['WEEKDAY', 'WEEKENDS/HOLIDAY']
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs=-1)
    
    for idx, month in enumerate(monthes):
         pd_temp = pd.read_csv('transport_node_train_20200' + str(month) +'.csv')
         pd_temp = pd_temp[(pd_temp['TIME_PER_HOUR'] >= time_period.min()) & (pd_temp['TIME_PER_HOUR'] <= time_period.max())] # remove the data outside the range
         pd_temp_weekday_in_raw = pd_temp[pd_temp['DAY_TYPE']=='WEEKDAY'].pivot_table(index=['TIME_PER_HOUR'], columns=['PT_CODE', 'DAY_TYPE'], values='TOTAL_TAP_IN_VOLUME').dropna(axis = 1)
         pd_temp_weekday_in = data_preprocessing(pd_temp_weekday_in_raw)
         pd_temp_weekday_out_raw = pd_temp[pd_temp['DAY_TYPE']=='WEEKDAY'].pivot_table(index=['TIME_PER_HOUR'], columns=['PT_CODE', 'DAY_TYPE'], values='TOTAL_TAP_OUT_VOLUME').dropna(axis = 1)
         pd_temp_weekday_out = data_preprocessing(pd_temp_weekday_out_raw)
         pd_temp_weekend_in_raw = pd_temp[pd_temp['DAY_TYPE']=='WEEKENDS/HOLIDAY'].pivot_table(index=['TIME_PER_HOUR'], columns=['PT_CODE', 'DAY_TYPE'], values='TOTAL_TAP_IN_VOLUME').dropna(axis = 1)
         pd_temp_weekend_in = data_preprocessing(pd_temp_weekend_in_raw)
         pd_temp_weekend_out_raw = pd_temp[pd_temp['DAY_TYPE']=='WEEKENDS/HOLIDAY'].pivot_table(index=['TIME_PER_HOUR'], columns=['PT_CODE', 'DAY_TYPE'], values='TOTAL_TAP_OUT_VOLUME').dropna(axis = 1)
         pd_temp_weekend_out = data_preprocessing(pd_temp_weekend_out_raw)
         obj_temp = {'Year': 2020,
                     'Month': month, 
                     'Data_weekday_in': pd_temp_weekday_in, 
                     'Data_weekday_out': pd_temp_weekday_out, 
                     'Data_weekend_in': pd_temp_weekend_in, 
                     'Data_weekend_out': pd_temp_weekend_out}
         Num_station = len(pd_temp['PT_CODE'].unique())
         print('Number of station in %s/%s: %s' %(2020, month, Num_station))
         
         fig, axs = plt.subplots(2, 2, figsize=(15,10))
         x_label = 'Time (hour)'
         y_label = 'Normalized passanger volume'
         plotSubplot(axs[0, 0], pd_temp_weekday_in_raw, x_label, y_label, 'PV_weekday (IN)')
         plotSubplot(axs[1, 0], pd_temp_weekend_in_raw, x_label, y_label, 'PV_weekend (IN)')
         plotSubplot(axs[0, 1], pd_temp_weekday_out_raw, x_label, y_label, 'PV_weekday (OUT)')
         plotSubplot(axs[1, 1], pd_temp_weekend_out_raw, x_label, y_label, 'PV_weekend (OUT)')
         
         kmeans_algorithm(kmeans, pd_temp_weekday_in, pd_temp_weekday_in_raw, 'Data_weekday_in')
         kmeans_algorithm(kmeans, pd_temp_weekday_out, pd_temp_weekday_out_raw, 'Data_weekday_out')
         kmeans_algorithm(kmeans, pd_temp_weekend_in, pd_temp_weekend_in_raw, 'Data_weekend_in')
         kmeans_algorithm(kmeans, pd_temp_weekend_out, pd_temp_weekend_out_raw, 'Data_weekend_out')
         
         # kmeans.fit(pd_temp_weekday_in.transpose())
         # for num_cluster in range(num_clusters):
         #     fig = plt.figure()
         #     fig = plt.plot(pd_temp_weekday_in.to_numpy()[:, np.where(kmeans.labels_ == num_cluster)][:, 0, :])
         #     plt.title('Cluster #%s' %num_cluster)
         
         pd_train_PV.append(obj_temp)
    # Num_station_per_month = [len(x['Data']['PT_CODE'].unique()) for x in pd_train_PV]
    # station_name_per_month = [x['Data']['PT_CODE'].unique() for x in pd_train_PV]
    # station_names = get_union_sets_np(station_name_per_month)
    
    # for idx, month in enumerate(monthes):
    #     pd_month = pd_train_PV[idx]
    #     for station_name in station_names:
    #         bool_selected_station = pd_month['Data']['PT_CODE'] == station_name
    #         pd_temp = pd_month['Data'][bool_selected_station]
    #         PV_day = []
    #         for day_type in day_types:
    #             bool_selected_day_type = pd_temp['DAY_TYPE'] == day_type
    #             pd_temp_selected_daytype = pd_temp[bool_selected_day_type]
    #             for time in time_period:
    #                 print(pd_temp_selected_daytype[pd_temp_selected_daytype['TIME_PER_HOUR'] == time])
                    
                    
    
    
    