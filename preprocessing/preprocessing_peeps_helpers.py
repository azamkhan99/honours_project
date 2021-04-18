# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
from basic_imports import *
import pandas as pd
from download_data import *
from project_specific.philap_helpers import *
from project_specific.daphne_helpers import *
from project_specific.peeps_helpers import *
from math import radians, cos, sin, asin, sqrt
from pixelgrams import *
import datetime
from constants import *
from load_files import *
import numpy as np
from scipy import stats

from pandarallel import pandarallel

pandarallel.initialize()

################################################***Helper Functions***#####################################################
def pmt(timestamp):
    #timestamp = row['timestamp']
    unix = timestamp.timestamp()
    minus = unix-120
    plus = unix+120
    c_minus = pd.Timestamp(minus, unit='s', tz='Asia/Kolkata')
    c_plus = pd.Timestamp(plus, unit='s', tz='Asia/Kolkata')
    first = str(c_minus).split()[1].split('-')[0].split('+')[0]
    last = str(c_plus).split()[1].split('-')[0].split('+')[0]

    return first, last


def drop_numerical_outliers(df, z_thresh=3.5):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)


def parse_dow_hod(dataset):
    #filDataset = dataset.filter(items = ['timestamp'])
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df['day_of_week'] = dataset['timestamp'].dt.day_name()
    df2['hour_of_day'] = dataset['timestamp'].dt.hour
    dataset = pd.concat([df2, df, dataset], axis =1)
    dataset = dataset.filter(items=['timestamp','pm2_5','hour_of_day','day_of_week', 'temperature', 'humidity' ,'gpsLongitude', 'gpsLatitude', 'UUID'])
    
    dataset.loc[(dataset.day_of_week == 'Monday'),'day_of_week']=int(0)
    dataset.loc[(dataset.day_of_week == 'Tuesday'),'day_of_week']=int(1)
    dataset.loc[(dataset.day_of_week == 'Wednesday'),'day_of_week']=int(2)
    dataset.loc[(dataset.day_of_week == 'Thursday'),'day_of_week']=int(3)
    dataset.loc[(dataset.day_of_week == 'Friday'),'day_of_week']=int(4)
    dataset.loc[(dataset.day_of_week == 'Saturday'),'day_of_week']=int(5)
    dataset.loc[(dataset.day_of_week == 'Sunday'),'day_of_week']=int(6)
    
    return dataset

def get_sensor_uuids(row, k_closest):

    all_stations = pd.read_csv("peeps_sensors.csv")
    all_stations = all_stations.set_index("UUID")
    
    def havesine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r
    
    
    lon1 = row['gpsLongitude']
    lat1 = row['gpsLatitude']
    
    haves = {}
    for idx, rows in all_stations.iterrows():
        haves[idx] = havesine(lon1,lat1,rows[0],rows[1])
    
    
    x = sorted(((v,k) for k,v in haves.items()))
    uuid = x[k_closest][1]
    dist = x[k_closest][0]

    
    return pd.Series([uuid, dist])

def get_sensors(dataset):
    inter = dataset
    inter[['closest_UUID', 'dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,0), axis=1)
    inter[['2_closest_UUID', '2_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,1), axis=1)
    inter[['3_closest_UUID', '3_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,2), axis=1)
    inter[['4_closest_UUID', '4_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,3), axis=1)
    inter[['5_closest_UUID', '5_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,4), axis=1)
    inter[['6_closest_UUID', '6_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,5), axis=1)
    inter[['7_closest_UUID', '7_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,6), axis=1)
    inter[['8_closest_UUID', '8_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,7), axis=1)
    inter[['9_closest_UUID', '9_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,8), axis=1)
    inter[['10_closest_UUID', '10_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,9), axis=1)
    inter[['11_closest_UUID', '11_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,10), axis=1)
    inter[['12_closest_UUID', '12_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,11), axis=1)
    inter[['13_closest_UUID', '13_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,12), axis=1)
#     inter[['14_closest_UUID', '14_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,13), axis=1)
#     inter[['15_closest_UUID', '15_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,14), axis=1)
#     inter[['16_closest_UUID', '16_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,15), axis=1)
#     inter[['17_closest_UUID', '17_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,16), axis=1)
#     inter[['18_closest_UUID', '18_dist_closest_s']] = inter.parallel_apply(lambda row: get_sensor_uuids(row,17), axis=1)
    inter = inter.reset_index()

    return inter

def closest_pm_value(row, dataframe):
    uuid1 = row['closest_UUID']#.item()
    dist1 = row['dist_closest_s']
    uuid2 = row['2_closest_UUID']
    dist2 = row['2_dist_closest_s']    
    uuid3 = row['3_closest_UUID']    
    dist3 = row['3_dist_closest_s']
    uuid4 = row['4_closest_UUID']    
    dist4 = row['4_dist_closest_s']
    uuid5 = row['5_closest_UUID']    
    dist5 = row['5_dist_closest_s']
    uuid6 = row['6_closest_UUID']    
    dist6 = row['6_dist_closest_s']
    
    uuid7 = row['7_closest_UUID']
    dist7 = row['7_dist_closest_s']
    uuid8 = row['8_closest_UUID']
    dist8 = row['8_dist_closest_s']    
    uuid9 = row['9_closest_UUID']    
    dist9 = row['9_dist_closest_s']
    uuid10 = row['10_closest_UUID']    
    dist10 = row['10_dist_closest_s']
    uuid11 = row['11_closest_UUID']    
    dist11 = row['11_dist_closest_s']
    uuid12 = row['12_closest_UUID']    
    dist12 = row['12_dist_closest_s']
    
    uuid13 = row['13_closest_UUID']
    dist13 = row['13_dist_closest_s']
 
    
    date = row['date']    
    first, last = np.vectorize(pmt)(row['timestamp'])

   
    
    try:
        pm, humidity = dataframe[dataframe.UUID == uuid1].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
        return pd.Series([pm, humidity, dist1, uuid1]) 
    except:
        try:
            pm, humidity = dataframe[dataframe.UUID == uuid2].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
            return pd.Series([pm,humidity, dist2, uuid2])
        except:
            try:
                pm, humidity = dataframe[dataframe.UUID == uuid3].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                return pd.Series([pm,humidity, dist3, uuid3]) 
            except:
                try:
                    pm, humidity = dataframe[dataframe.UUID == uuid4].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                    return pd.Series([pm,humidity, dist4, uuid4]) 
                except:
                    try:
                        pm, humidity = dataframe[dataframe.UUID == uuid5].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                        return pd.Series([pm,humidity, dist5, uuid5]) 
                    except:
                        try:
                            pm, humidity = dataframe[dataframe.UUID == uuid6].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                            return pd.Series([pm,humidity, dist6, uuid6]) 
                        except:
                            try:
                                pm, humidity = dataframe[dataframe.UUID == uuid7].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                return pd.Series([pm,humidity, dist7, uuid7]) 
                            except:
                                try:
                                    pm, humidity = dataframe[dataframe.UUID == uuid8].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                    return pd.Series([pm,humidity, dist8, uuid8])
                                except:
                                    try:
                                        pm, humidity = dataframe[dataframe.UUID == uuid9].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                        return pd.Series([pm,humidity, dist9, uuid9]) 
                                    except:
                                        try:
                                            pm, humidity = dataframe[dataframe.UUID == uuid10].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                            return pd.Series([pm,humidity, dist10, uuid10]) 
                                        except:
                                            try:
                                                pm, humidity = dataframe[dataframe.UUID == uuid11].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                                return pd.Series([pm,humidity, dist11, uuid11]) 
                                            except:
                                                try:
                                                    pm, humidity = dataframe[dataframe.UUID == uuid12].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                                    return pd.Series([pm,humidity, dist12, uuid12])
                                                except:
                                                    try:
                                                        pm, humidity = dataframe[dataframe.UUID == uuid13].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                                        return pd.Series([pm,humidity, dist13, uuid13]) 
                                                    except:
                                                        try:
                                                            pm, humidity = dataframe[dataframe.UUID == uuid14].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                                            return pd.Series([pm,humidity, dist14, uuid14])
                                                        except:
                                                            try:
                                                                pm, humidity = dataframe[dataframe.UUID == uuid15].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                                                return pd.Series([pm,humidity, dist15, uuid15]) 
                                                            except:
                                                                try:
                                                                    pm, humidity = dataframe[dataframe.UUID == uuid16].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                                                    return pd.Series([pm,humidity, dist16, uuid16]) 
                                                                except:
                                                                    try:
                                                                        pm, humidity = dataframe[dataframe.UUID == uuid17].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                                                        return pd.Series([pm,humidity, dist17, uuid17]) 
                                                                    except:
                                                                        try:
                                                                            pm, humidity = dataframe[dataframe.UUID == uuid18].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
                                                                            return pd.Series([pm,humidity, dist18, uuid18]) 
                                                                        except:
                                                                            pm = 0
                                                                            humidity = 0
                                                                            return pd.Series([pm,humidity, 'no sensor'])  


def parse_date_time(dataframe):
    
    def parse_date(row):
        date = str(row['timestamp']).split()[0]
        return date
    
    def parse_time(row):
        timestamp = str(row['timestamp']).split()[1]
        time = timestamp.split('-')[0]
        return time


    
    split_ind = dataframe.copy()
    split_ind['date'] = split_ind.apply(lambda row: parse_date(row), axis=1)
    split_ind['time'] = split_ind.apply(lambda row: parse_time(row), axis=1)
    #split_ind = split_ind.filter(items=['timestamp','date','time','pm2_5','hour_of_day','day_of_week' ,'temperature', 'humidity','gpsLongitude', 'gpsLatitude', 'UUID_closest_s', 'dist_closest_S'])
    return split_ind

def clean_df(df):
    df = df.filter(items=['timestamp','pm2_5','hour_of_day','day_of_week' , 'humidity','gpsLongitude', 'gpsLatitude', 'closest_pm', 'dist_to_closest_pm','closest_pm_id', 'walk'])
    return df

def outlier_removal(dataset):

    dataset = dataset.set_index('timestamp')
    dataset = dataset[dataset['gpsLongitude'] > 76]
    #dataset = dataset[dataset.temperature.gt(0)]
    dataset = dataset[dataset.humidity.gt(0)]
    #cleaned_dataset = drop_numerical_outliers(dataset)
    return cleaned_dataset

################################################***Static Sensor Data***###################################################
def loading_peeps_static():
    peeps_logs = load_peeps_participant_details()
    pps_static_airspeck = pd.DataFrame()
    for idx, row in peeps_logs.iterrows():
        subj_id = row.name
    
        frame = load_static_airspeck_file(subj_id, 'peeps',suffix_filename="_home", upload_type ='automatic', subject_visit_number=None)
        frame1 = load_static_airspeck_file(subj_id, 'peeps',suffix_filename="_work", upload_type ='automatic', subject_visit_number=None)
    
        if frame is not None:
            frame['UUID'] = row[5]
            frame['type'] = 'home'
        
        if frame1 is not None:
            frame1['UUID'] = row[4]
            frame1['type'] = 'work'
    
        pps_static_airspeck = pps_static_airspeck.append(frame)
        pps_static_airspeck = pps_static_airspeck.append(frame1)

    pps_static_airspeck['timestamp'] = pps_static_airspeck.timestamp.dt.round('S', 'NaT')
    pps_static_airspeck = pps_static_airspeck.set_index('timestamp')

    return pps_static_airspeck

def preprocessing_peeps_static(unprocessed_dataset):
    pps_static_df = unprocessed_dataset.filter(items=['timestamp','pm2_5', 'temperature', 'humidity' ,'gpsLongitude', 'gpsLatitude', 'UUID'])
    pps_static_df = pps_static_df.reset_index()
    static_peeps = parse_dow_hod(pps_static_df)

    

    static_peeps = static_peeps.set_index("timestamp")

    return static_peeps



################################################***Personal Sensor Data***###################################################

def loading_peeps_personal():#Function loads airspeck-p files and removes rows with erroneous values

    peeps_logs = load_peeps_participant_details()
    pps_pers_airspeck = pd.DataFrame()
    for idx, row in peeps_logs.iterrows():
        subj_id = row.name

    
        frame = load_personal_airspeck_file(subj_id, 'peeps', upload_type='manual', subject_visit_number=None, is_minute_averaged=False)
    
        if frame is not None:
            frame['walk'] = int(subj_id[3:])
        pps_pers_airspeck = pps_pers_airspeck.append(frame)
        
    pps_pers_airspeck['timestamp'] = pps_pers_airspeck.timestamp.dt.round('S', 'NaT')
    pps_pers_airspeck = pps_pers_airspeck.set_index('timestamp')
    pps_pers_airspeck = pps_pers_airspeck.resample('min').mean()
    pps_pers_airspeck = pps_pers_airspeck[pps_pers_airspeck['temperature'].notna()]
    pps_pers_airspeck = pps_pers_airspeck[pps_pers_airspeck['gpsLongitude'] != 0]
    pps_pers_airspeck = pps_pers_airspeck[pps_pers_airspeck['gpsLatitude'] != 0]

    return pps_pers_airspeck
    
#Function performs full preprocessing and outlier removal  
def preprocessing_peeps_personal(unprocessed_dataset, static_data):
    pps_pers_df = unprocessed_dataset.filter(items=['timestamp','pm2_5','gpsLongitude', 'gpsLatitude', 'walk'])
    pps_pers_df = pps_pers_df.sort_index()
    pps_pers_df = pps_pers_df.reset_index()

    peeps_p = parse_dow_hod(pps_pers_df)

    peeps_p = peeps_p.set_index('timestamp')

    inter = peeps_p

    inter = get_sensors(inter)

    dtp_inter = parse_date_time(inter)

    dtp_inter[['closest_pm','humidity', 'dist_to_closest_pm', 'closest_pm_id']] = dtp_inter.parallel_apply(lambda row: closest_pm_value(row, static_data), axis=1)

    personal_peeps = clean_df(dtp_inter)
    personal_peeps = personal_peeps[personal_peeps.dist_to_closest_pm != 'no sensor']

    personal_peeps = outlier_removal(personal_peeps)

    return personal_peeps