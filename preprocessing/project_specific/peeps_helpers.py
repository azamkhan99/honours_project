import glob
#import datetime #import datetime

import numpy as np
import pandas as pd
from pytz import timezone

from constants import peeps_participant_details_filepath
from download_data import download_static_airspeck, download_respeck_data, download_personal_airspeck_data
from load_files import load_static_airspeck_file
# Until all cells are filled in, assume that the start and end times are the same for personal and static sensors
from misc_utils import distance_of_coords, gps_conversion_to_decimal

static_sensor_ids_peeps = ["9FB02899E28FF607", "B61241EF668DBC2C", "E786F1568F65C296", "33B45C90B13731DE",
                           "200A7CED9D597407", "E1EFA8FCA05B3FF9"]

peeps_office_locations = {"WHO country office (WCO)": (28.5593250, 77.1882889, 'B61241EF668DBC2C', 'WCO'),
                          "WHO SEARO": (28.6315639, 77.2079944, 'E786F1568F65C296', 'WHOSEARO'),
                          "UN house": (28.5926333, 77.2226278, 'E1EFA8FCA05B3FF9', 'UNHOUSE'),  # Lodhi road
                          "UNESCO": (28.595760345458984, 77.17916107177734, '200A7CED9D597407', 'UNESCO')}

##"WHO SEARO": (28.631584, 77.2079944

peeps_work_id_to_gps_phase1 = {'B61241EF668DBC2C': {'gpsLatitude': 28.5593250, 'gpsLongitude': 77.1882889},
                        'E786F1568F65C296': {'gpsLatitude': 28.6315639, 'gpsLongitude': 77.2079944},
                        'E1EFA8FCA05B3FF9': {'gpsLatitude': 28.5926333, 'gpsLongitude': 77.2226278}}

#phase 2 location
peeps_work_id_to_gps = {'xxx': {'gpsLatitude': 28.5593250, 'gpsLongitude': 77.1882889},
                        'B61241EF668DBC2C': {'gpsLatitude': 28.6315639, 'gpsLongitude': 77.2079944},
                        'xxx1': {'gpsLatitude': 28.5926333, 'gpsLongitude': 77.2226278},
                        '33B45C90B13731DE': {'gpsLatitude':  28.6105, 'gpsLongitude': 77.2156},
                        '67884227A72B71D1': {'gpsLatitude': 28.5600, 'gpsLongitude': 77.1886},
                       'E864778321F55A8F' : {'gpsLatitude': 28.5925105,  'gpsLongitude':77.2229058}}


def download_all_peeps_data(overwrite_if_already_exists=False, download_raw_airspeck_data=True, phase=1):
    project_name = 'peeps'
    logs = load_peeps_participant_details(phase=phase)

    for idx, row in logs.iterrows():
        subject_id = row['Subject ID']

        if len(subject_id) != 6:
            continue

        from_time = row['From time personal sensors']
        start_date = row['From date all sensors'].replace(hour=from_time.hour, minute=from_time.minute,
                                                          second=from_time.second)#.to_pydatetime()

        to_time = row['To time personal sensors']
        #print(row['To date all sensors'])
        #print(row)
        end_date = row['To date all sensors'].replace(hour=to_time.hour, minute=to_time.minute,
                                                      second=to_time.second)#.to_pydatetime()

        timeframe = [start_date, end_date]
        # Download personal data if not yet present
        #Changed respeck to is_minute-averaged 
        download_respeck_data(subject_id, upload_type='manual', project_name=project_name, timeframe=timeframe,
                              overwrite_if_already_exists=overwrite_if_already_exists, is_minute_averaged=True)
        download_personal_airspeck_data(subject_id, upload_type='manual', project_name=project_name,
                                        is_minute_averaged=not download_raw_airspeck_data,
                                        timeframe=timeframe,
                                        overwrite_if_already_exists=overwrite_if_already_exists)

        # Download data from stationary sensor at home
        if not pd.isnull(row['Home static sensor ID']):
            # Was a personal sensor used as static one?
            if len(row['Home static sensor ID']) == 6:
                # We download the data from the personal sensor into a file which looks just like the other static
                # sensors so that it's easier to work with it.
                download_personal_airspeck_data(row['Home static sensor ID'], upload_type='manual',
                                                timeframe=timeframe,
                                                overwrite_if_already_exists=overwrite_if_already_exists,
                                                filename="{}({})_static_airspeck_automatic_home.csv")
            else:
                download_static_airspeck(row['Home static sensor ID'], sensor_label=subject_id,
                                         project_name=project_name, upload_type='automatic', suffix_filename='_home',
                                         timeframe=timeframe, overwrite_if_already_exists=overwrite_if_already_exists)

        # Download data from stationary sensor at work
        if not pd.isnull(row['Work static sensor ID']):
            download_static_airspeck(row['Work static sensor ID'], sensor_label=subject_id,
                                     project_name=project_name, upload_type='automatic', suffix_filename='_work',
                                     timeframe=timeframe, overwrite_if_already_exists=overwrite_if_already_exists)

    print("Finished!")


def load_peeps_participant_details_pilot():
    logs = pd.read_excel(peeps_participant_details_filepath).drop(3)
    return logs


def load_peeps_participant_details(phase=1):
    details = pd.read_excel(peeps_participant_details_filepath)
    if (phase == 1):
        details = details.iloc[1:75]
    if (phase == 2):
        details = details.iloc[77:]
    details = details.set_index('Subject ID')
    details['Subject ID'] = details.index
    #if (phase == 1):
    #    details = details.drop('PEV015')
    return details 


def download_closest_office_sensor_peeps(work_gps, timeframe, subject_id, overwrite_if_already_exists=False):
    distances = []
    ids = []
    for name, coords in peeps_office_locations.iteritems():
        distances.append(distance_of_coords(work_gps['gpsLatitude'], work_gps['gpsLongitude'], coords[0], coords[1]))
        ids.append(coords[2])
    min_idx = np.argmin(distances)

    # Download data for sensor which is closest
    print("Closest office: {} ({} km)".format(peeps_office_locations.keys()[min_idx], distances[min_idx]))
    download_static_airspeck(ids[min_idx], sensor_label=subject_id,
                             project_name='peeps', upload_type='automatic', suffix_filename='_work',
                             timeframe=timeframe, overwrite_if_already_exists=overwrite_if_already_exists)


def get_peeps_office_airspeck(office_name, timeframe, calibrated=False, use_all_features_for_calibration=False):
    uuid = peeps_office_locations[office_name][2]
    filename = "{}_{}_{}_{}_static_airspeck.csv".format(office_name, uuid, datetime.strftime(timeframe[0], "%Y%m%d"),
                                                        datetime.strftime(timeframe[1], "%Y%m%d"))

    download_static_airspeck(uuid, timeframe=timeframe, filename=filename)
    airspeck = load_static_airspeck_file(uuid, calibrate_pm_and_gas=calibrated,
                                         use_all_features_for_pm_calibration=use_all_features_for_calibration)
    return airspeck


def peeps_get_recording_timeframe_for_subject(subj_id, phase=1):
    peeps_logs = load_peeps_participant_details(phase)
    
    row = peeps_logs.loc[subj_id]

    tz = timezone('Asia/Kolkata')

    from_time = row['From time personal sensors']
    start_date = tz.localize(row['From date all sensors'].replace(hour=from_time.hour, minute=from_time.minute,
                                                                  second=from_time.second))#.to_pydatetime())

    to_time = row['To time personal sensors']
    end_date = tz.localize(row['To date all sensors'].replace(hour=to_time.hour, minute=to_time.minute,
                                                              second=to_time.second))#.to_pydatetime())

    return [start_date, end_date]


def get_nearest_reference_sensor_delhi(location, recording_timeframe, data_available_percentage=0.4):
    cpcb_basefolder = '/Users/zoepetard/Documents/Speckled/projects/daphne/data/From CADTIME/'
    cpcb_locations = pd.read_excel(cpcb_basefolder + "site_locations_040619.xlsx")

    distances = []
    for idx, row in cpcb_locations.iterrows():
        distances.append(distance_of_coords(location['gpsLatitude'], location['gpsLongitude'],
                                            float(row['LAT']), float(row['LNG'])))
    distances = np.asarray(distances)

    sort_idxs = np.argsort(distances)
    station_names_sorted = cpcb_locations['File_Suffix'][sort_idxs]

    # We don't have any recordings which cross new years. Therefore, they are either in 2018 or 2019
    file_format = 'csv'
    if recording_timeframe[0].year == 2019:
        dir_data = cpcb_basefolder + "CPCB Data 2019/"
        month = recording_timeframe[0].month
        if month == 11 or month == 12:
            dir_data = cpcb_basefolder + "CPCB_Nov-Dec_2019/"
            file_format = 'excel'
    else:
        dir_data = cpcb_basefolder + "CPCB Data until 2018/"

    for s_idx, station in enumerate(station_names_sorted):
        file_path = glob.glob(dir_data + station + "*")
        if len(file_path) == 0: ## No data for that station
            continue
        if  not file_path: ## No data for that station
            continue

        # Load next-nearest CPCB station
        if file_format == 'csv':
            cpcb_station = pd.read_csv(file_path[0])
        if file_format == 'excel':
            cpcb_station = pd.read_excel(file_path[0], skiprows=16)
            cpcb_station['PM2.5'] = cpcb_station['PM2.5'].apply(pd.to_numeric, errors='coerce') #Replace string 'None' with NaN. Format as float
            cpcb_station = cpcb_station.rename(columns={'From Date': 'From.Date'})
        new_index = pd.to_datetime(cpcb_station['From.Date'], format="%d-%m-%Y %H:%M").dt.tz_localize('Asia/Kolkata')
        cpcb_station = cpcb_station.set_index(new_index).sort_index()

        num_hours_should = int((recording_timeframe[1] - recording_timeframe[0]).total_seconds() / (60 * 60.))
        cpcb_in_time = cpcb_station[recording_timeframe[0]:recording_timeframe[1]]

        if (cpcb_in_time['PM2.5'].dropna().size / num_hours_should >= data_available_percentage):
            break
        
        # Add empty PM10 column if it's not there
        if 'PM10' not in cpcb_in_time:
            cpcb_in_time.loc[:, 'PM10'] = np.nan

        # Do we have at least 40% of data in this timeframe for this station? If not, look at next closest station
        if (cpcb_in_time['PM2.5'].dropna().size / num_hours_should >= data_available_percentage and
                cpcb_in_time['PM10'].dropna().size / num_hours_should >= data_available_percentage):
            break

    # We have the closest station. Return data and distance in km
    return cpcb_in_time, distances[sort_idxs][s_idx]


def get_peeps_subject_ids():
    logs = load_peeps_participant_details()

    ids = []
    for idx, row in logs.iterrows():
        subj_id = row['Subject ID']

        if len(subj_id) != 6:
            continue

        ids.append(subj_id)

    return ids
