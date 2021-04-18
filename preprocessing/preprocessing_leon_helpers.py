# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
from basic_imports import *

from download_data import *
from project_specific.philap_helpers import *
from project_specific.daphne_helpers import *
from project_specific.peeps_helpers import *
from math import radians, cos, sin, asin, sqrt
from pixelgrams import *
import folium
import datetime
from constants import *
from load_files import *
from scipy.stats import *
import numpy as np
from pandarallel import pandarallel
pandarallel.initialize()

##################################################**Helper Functions**###########################################################
def pmt(row):
    timestamp = row['timestamp']
    unix = timestamp.timestamp()
    minus = unix-120
    plus = unix+180
    c_minus = pd.Timestamp(minus, unit='s', tz='America/Mexico_City')
    c_plus = pd.Timestamp(plus, unit='s', tz='America/Mexico_City')
    first = str(c_minus).split()[1].split('-')[0]
    last = str(c_plus).split()[1].split('-')[0]
    
    return first, last
    
def parse_date(row):
    date = str(row['timestamp']).split()[0]
    return date
def parse_time(row):
    timestamp = str(row['timestamp']).split()[1]
    time = timestamp.split('-')[0]
    return time
def parse_date_time(dataframe):
    split_ind = dataframe.copy()
    split_ind['date'] = split_ind.apply(lambda row: parse_date(row), axis=1)
    split_ind['time'] = split_ind.apply(lambda row: parse_time(row), axis=1)
    return split_ind
def clean_df(df):
    df = df.filter(items=['walk','timestamp','pm2_5','hour_of_day','day_of_week', 'humidity','gpsLongitude', 'gpsLatitude', 'closest_pm', 'dist_to_closest_pm', 'closest_UUID'])
    return df

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
    
    date = row['date']#.item()
    first, last = pmt(row)
    
    #print("a")
    try:
        pm, humidity = dataframe[dataframe.UUID == uuid1].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
     #   print("b")
        return pd.Series([pm, humidity, dist1, uuid1]) 
    except:
        try:
            pm, humidity = dataframe[dataframe.UUID == uuid2].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
      #      print("c")
            return pd.Series([pm,humidity, dist2, uuid2])
        except:
            try:
                pm, humidity = dataframe[dataframe.UUID == uuid3].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
       #         print("d")
                return pd.Series([pm,humidity, dist3, uuid3]) 
            except:
                try:
                    pm, humidity = dataframe[dataframe.UUID == uuid4].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
        #            print("e")
                    return pd.Series([pm,humidity, dist4, uuid4]) 
                except:
                    try:
                        pm, humidity = dataframe[dataframe.UUID == uuid5].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
         #               print("f")
                        return pd.Series([pm,humidity, dist5, uuid5]) 
                    except:
                        try:
                            pm, humidity = dataframe[dataframe.UUID == uuid6].loc[date].between_time(first, last)[['pm2_5', 'humidity']].iloc[0,:]
          #                  print("g")
                            return pd.Series([pm,humidity, dist6, uuid6]) 
                        except:
                            pm = 0
                            humidity = 0
           #                 print("h")
                            return pd.Series([pm, humidity,'no sensor'])



def get_sensor_uuids(row, k_closest,distorid):
    
    median_sensors = pd.read_csv("leon_sensors.csv")
    median_sensors = median_sensors.set_index("UUID")
    
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
    for idx, rows in median_sensors.iterrows():
        haves[idx] = havesine(lon1,lat1,rows[0],rows[1])
    
    
    x = sorted(((v,k) for k,v in haves.items()))
    uuid = x[k_closest][1]
    dist = x[k_closest][0]
    #print(dist)
    #uuid = median_sensors['UUID'].iloc[[indic]].values
    
    if (distorid == 'id'):
        return uuid
    if (distorid == 'dist'):
        return dist
    


def get_sensors(dataset):
    
    inter = dataset
    inter['closest_UUID'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,0, 'id'), axis=1)
    inter['dist_closest_s'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,0, 'dist'), axis=1)
    inter['2_closest_UUID'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,1, 'id'), axis=1)
    inter['2_dist_closest_s'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,1, 'dist'), axis=1)
    inter['3_closest_UUID'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,2, 'id'), axis=1)
    inter['3_dist_closest_s'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,2, 'dist'), axis=1)
    inter['4_closest_UUID'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,3, 'id'), axis=1)
    inter['4_dist_closest_s'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,3, 'dist'), axis=1)
    inter['5_closest_UUID'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,4, 'id'), axis=1)
    inter['5_dist_closest_s'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,4, 'dist'), axis=1)
    inter['6_closest_UUID'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,5, 'id'), axis=1)
    inter['6_dist_closest_s'] = inter.parallel_apply(lambda row: get_sensor_uuids(row,5, 'dist'), axis=1)
    inter = inter.reset_index()
    
    return inter

def parse_dow_hod(dataset):
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    df['day_of_week'] = dataset['timestamp'].dt.day_name()
    df2['hour_of_day'] = dataset['timestamp'].dt.hour
    dataset = pd.concat([df2, df, dataset], axis =1)
    dataset = dataset.filter(items=['timestamp','pm2_5','hour_of_day','day_of_week', 'temperature', 'humidity' ,'gpsLongitude', 'gpsLatitude', 'walk', 'UUID'])

    dataset.loc[(dataset.day_of_week == 'Monday'),'day_of_week']=int(0)
    dataset.loc[(dataset.day_of_week == 'Tuesday'),'day_of_week']=int(1)
    dataset.loc[(dataset.day_of_week == 'Wednesday'),'day_of_week']=int(2)
    dataset.loc[(dataset.day_of_week == 'Thursday'),'day_of_week']=int(3)
    dataset.loc[(dataset.day_of_week == 'Friday'),'day_of_week']=int(4)
    dataset.loc[(dataset.day_of_week == 'Saturday'),'day_of_week']=int(5)
    dataset.loc[(dataset.day_of_week == 'Sunday'),'day_of_week']=int(6)
    
    return dataset





def load_mexico_airrespeck_file(filepath, project_name, timestamp_column_name='timestamp'):
    try:
        data = pd.read_excel(filepath, engine='openpyxl')
        data['timestamp'] = pd.to_datetime(data[timestamp_column_name]).dt.tz_localize('UTC').dt.tz_convert(
            project_mapping[project_name][1])
        data = data.set_index('timestamp')
        data['timestamp'] = data.index
        data = data.replace('None', np.nan)
        return data
    except EmptyDataError:
        print("Skipped file because it is empty")
        return
    except IOError:
        print("Skipped file because it doesn't exist")
        return
    
def load_mexico_personal_airspeck_file(subject_id, project_name=None, upload_type='automatic', is_minute_averaged=True,
                                subject_visit_number=None, suffix_filename="",
                                calibrate_pm_and_gas=False, use_all_features_for_pm_calibration=False,
                                use_all_features_for_gas_calibration=False, suppress_output=False,
                                set_below_zero_to=np.nan, return_calibration_flag=False, calibration_id=None,
                                filter_pm=True, country_name=None):

    
    if subject_visit_number is None:
        label_files = subject_id
    else:
        label_files = "{}({:.0f})".format(subject_id, int(subject_visit_number))

    if project_name is None:
        project_name = get_project_for_subject(subject_id)

    if is_minute_averaged:
        filename = "{}_{}_calibrated.xlsx".format(label_files, suffix_filename)
    else:
        filename = "{}_{}_calibrated_raw.xlsx".format(label_files, suffix_filename)

    print("Loading file: {}".format(project_mapping[project_name][2] + filename))
    data = load_mexico_airrespeck_file(project_mapping[project_name][2] + filename,
                                project_name)

    if calibrate_pm_and_gas:
        result_date, was_calibrated_pm, was_calibrated_no2, was_calibrated_ox, data = calibrate_airspeck(
            subject_id, data, project_name=project_name, calibrate_pm=True, calibrate_no2=False, 
                           calibrate_ox=False, calibration_id=calibration_id, 
                           use_all_features_pm=use_all_features_for_pm_calibration,
                           use_all_features_gas=use_all_features_for_gas_calibration, country_name=country_name)

    if filter_pm and data is not None and len(data) > 0:
        below_zero_mask = data['pm2_5'] <= 0

        if np.count_nonzero(below_zero_mask):
            if not suppress_output:
                print("Setting {} values equal to or below 0 to {}".format(np.count_nonzero(below_zero_mask),
                                                                           set_below_zero_to))
            data.loc[below_zero_mask, 'pm2_5'] = set_below_zero_to

        # Fix humidity values. Sometimes valid readings of humidity pass 100. Above 105, they are definitely invalid
        

    if calibrate_pm_and_gas and return_calibration_flag:
        return result_date, was_calibrated_pm, data
    else:
        return data
    
    
def load_mexico_static_airspeck_file(sid_or_uuid, project_name=None, sensor_label=None, suffix_filename="",
                              upload_type='automatic',
                              subject_visit_number=None, calibrate_pm=False, calibrate_ox=False, calibrate_no2=False,
                              use_all_features_for_pm_calibration=False,
                              use_all_features_for_gas_calibration=True,
                              return_calibration_flag=False, calibration_id=None, filename=None,
                              country_name=None):
    assert upload_type in ['automatic', 'sd_card'], "upload_type has to be either 'automatic' or 'sd_card'"

    if project_name is None and len(sid_or_uuid) == 6:
        project_name = get_project_for_subject(sid_or_uuid)

    if sensor_label is None:
        if subject_visit_number is None:
            sensor_label = "{}".format(sid_or_uuid)
        else:
            sensor_label = "{}({})".format(sid_or_uuid, subject_visit_number)

    if filename is None:
        filename = "{}_{}_calibrated.xlsx".format(sensor_label, suffix_filename)

    print("Loading file: {}".format(project_mapping[project_name][2] + filename))
    data = load_mexico_airrespeck_file(project_mapping[project_name][2] + filename, project_name)


    if calibrate_pm or calibrate_ox or calibrate_no2:
        result_date, was_calibrated_pm, was_calibrated_no2, was_calibrated_ox,  data = calibrate_airspeck(
            sid_or_uuid, data, calibrate_pm=calibrate_pm, calibrate_no2=calibrate_no2, 
            calibrate_ox=calibrate_ox,project_name=project_name, calibration_id=calibration_id,
            use_all_features_pm=use_all_features_for_pm_calibration,
            use_all_features_gas=use_all_features_for_gas_calibration, country_name=country_name)

        if return_calibration_flag:
            return result_date, was_calibrated_pm,  was_calibrated_no2, was_calibrated_ox, data

    return data

def load_mexico_participant_details(phase=1):
    details = pd.read_excel(leon_logs_filepath, engine='openpyxl')
    if (phase == 1):
        details = details.iloc[1:20]
    details = details.set_index('Sensor UUID')
    details['Sensor UUID'] = details.index
    return details 



##################################################**Static Files**##############################################################

def loading_leon_static():
    
    
    gdl_pd = pd.read_excel(leon_logs_filepath, sheet_name="Leon Walking", engine='openpyxl')


    #Download data (and want to write one file per day)
    for idx, row in gdl_pd.iterrows():
        #print(row['Date'])

        subject_id = row["Subject ID"]    
        from_time = row['Start time']
        from_date = row['Date'].replace(hour=from_time.hour, minute=from_time.minute,
                                                              second=from_time.second)
        to_time = row['End time']
        to_date = row['Date'].replace(hour=to_time.hour, minute=to_time.minute,
                                                          second=to_time.second)
        timeframe = [from_date, to_date]

        suffix_filename = "_" + str(row["Date"])[:10] + "_leon"
        
        
    CALIBRATE = False

    leon_logs = load_mexico_participant_details()
    leon_static_airspeck = pd.DataFrame()

    for idx, row in leon_logs.iterrows():
        uuid = row['Sensor UUID']    


        frame = load_mexico_static_airspeck_file(uuid, project_name="leon", suffix_filename="leon",
                                  upload_type='sd_card',
                                    calibrate_pm=CALIBRATE, calibrate_ox=CALIBRATE, calibrate_no2=CALIBRATE,
                                  use_all_features_for_pm_calibration=False,
                                  use_all_features_for_gas_calibration=True,
                                  return_calibration_flag=False, calibration_id=None, country_name="Mexico")

        if frame is not None:
            frame['UUID'] = uuid
        leon_static_airspeck = leon_static_airspeck.append(frame)
        
    leon_static_airspeck['timestamp'] = leon_static_airspeck.timestamp.dt.round('S', 'NaT')
    leon_static_airspeck = leon_static_airspeck.set_index('timestamp')
    uncalibrated_leon = pd.read_csv("uncalibrated_leon.csv")
    leon_static_airspeck['humidity'] = uncalibrated_leon['humidity'].values
    leon_static_airspeck['temperature'] = uncalibrated_leon['temperature'].values
        
        
    return leon_static_airspeck


def preprocessing_leon_static(unprocessed_dataset):
    
    leon_static_df = unprocessed_dataset.filter(items=['timestamp','pm2_5','temperature','humidity','gpsLongitude', 'gpsLatitude', 'UUID'])
    leon_static_df = leon_static_df.reset_index()
    static_leon = parse_dow_hod(leon_static_df)
    
    return static_leon




    
    
    
    
    
#################################################**Personal Files**############################################################## 


def loading_leon_personal():
    
    gdl_pd = pd.read_excel(leon_logs_filepath, sheet_name="Leon Walking", engine='openpyxl')


    #Download data (and want to write one file per day)
    for idx, row in gdl_pd.iterrows():
        #print(row['Date'])

        subject_id = row["Subject ID"]    
        from_time = row['Start time']
        from_date = row['Date'].replace(hour=from_time.hour, minute=from_time.minute,
                                                              second=from_time.second)
        to_time = row['End time']
        to_date = row['Date'].replace(hour=to_time.hour, minute=to_time.minute,
                                                          second=to_time.second)
        timeframe = [from_date, to_date]

        suffix_filename = "_" + str(row["Date"])[:10] + "_leon"
    
    leon_pers_airspeck = pd.DataFrame()

    for idx, row in gdl_pd.iterrows():
        sid = row["Subject ID"]
        date = str(row["Date"])[:10]

        suffix_filename = "leon_" + str(row["Date"])[:10]

        frame = load_mexico_personal_airspeck_file(sid, project_name="leon", suffix_filename=suffix_filename,
                                  upload_type='manual',is_minute_averaged=False)
        if frame is not None:
            frame['walk'] = sid+ "_" +date
        leon_pers_airspeck = leon_pers_airspeck.append(frame)
        
    leon_pers_airspeck['timestamp'] = leon_pers_airspeck.timestamp.dt.round('S', 'NaT')
    leon_pers_airspeck = leon_pers_airspeck.set_index('timestamp')
    
    return leon_pers_airspeck

def preprocessing_leon_personal(unprocessed_dataset, static_data):
    
    leon_pers_df = unprocessed_dataset.filter(items=['timestamp','pm2_5','gpsLongitude', 'gpsLatitude', 'walk'])
    leon_pers_df = leon_pers_df.sort_index()
    leon_pers_df = leon_pers_df.reset_index()
    leon_p = parse_dow_hod(leon_pers_df)
    leon_p = leon_p.set_index('timestamp')
    
    inter = leon_p
    inter = get_sensors(inter)
    dtp_inter = parse_date_time(inter)
    
    static_data = static_data.set_index("timestamp")
    dtp_inter[['closest_pm','humidity','dist_to_closest_pm', 'closest_pm_id']] = dtp_inter.parallel_apply(lambda row: closest_pm_value(row, static_data), axis=1)
    leon = clean_df(dtp_inter)
    leon = leon[leon['dist_to_closest_pm'] != 'no sensor']
    
    return leon
    
    #inter['timestamp'] = inter['timestamp'].parallel_apply(lambda x: pd.to_datetime(x).tz_convert('America/Mexico_City'))
    
    
    
    
    