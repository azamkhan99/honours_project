# Load Airspeck, Respeck and log files from the local file system. Change the timestamps from UTC into the
# project_specific timzone. Apply basic data filters. If no filenames are specified during download, the downloaded
# files should follow the same format as the loading methods expect.
import os
from datetime import timedelta

import pandas as pd
from pandas.errors import EmptyDataError
from pytz import timezone

from constants import project_mapping, calibration_columns_pm, calibration_columns_ox, calibration_columns_no2
from decrypt_file import *
from misc_utils import get_datastore_client, get_from_entity_if_present, get_project_for_subject, filter_out_outliers_gas


def load_log_file(subj_id):
    filename = project_mapping[subj_id[:2]][2] + '{}_logs.csv'.format(subj_id)

    if not os.path.isfile(filename):
        print("No file exists for subject {}".format(subj_id))

    data = pd.read_csv(filename, names=['timestamp', 'message'])
    return data


def load_personal_airspeck_file(subject_id, project_name=None, upload_type='automatic', is_minute_averaged=True,
                                subject_visit_number=None, suffix_filename="",
                                calibrate_pm_and_gas=False, use_all_features_for_pm_calibration=False,
                                use_all_features_for_gas_calibration=False, suppress_output=False,
                                set_below_zero_to=np.nan, return_calibration_flag=False, calibration_id=None,
                                filter_pm=True, country_name=None):
    '''
    Load an Airspeck personal csv file to a pandas dataframe in the correct timezone
    :param subject_id: 6-character subject ID
    :param project_name: For some projects, this is the actual name "daphne", for others, it's the project ID.
    See constants.py for a list of all project names
    :param filename: the filename to load. If None, load default filename "[Subject ID]_airspeck_personal.csv"
    :param is_minute_averaged: If the raw file was downloaded instead of minute averages. This only affects the default filename.
    :param subject_visit_number: Which of several recordings of a subject should be loaded.
    Only relevant for some projects like Daphne
    :param calibrate_pm_and_gas: Whether to calibrated the PM2.5 data, if calibration factors are available.
    :param use_all_features_for_pm_calibration: Whether to only use the uncalibrated PM2.5 data (recommended), or all
     features, including the bin counts and temperature/humidity. The latter often looks better during the actual
      calibration, but gives worse results later.
    :param suppress_output: Whether to print out if bad values were filtered, i.e. set to zero.
    :param set_below_zero_to: Set values below zero to a desired value (default np.nan)
    :param return_calibration_flag: Instead of just returning the dataframe, prepend whether the data was calibrated:
    return is_calibrated, data. This is useful to see if data from a subject loaded from disk was calibrated.
    :return:
    '''
    if subject_visit_number is None:
        label_files = subject_id
    else:
        label_files = "{}({:.0f})".format(subject_id, int(subject_visit_number))

    if project_name is None:
        project_name = get_project_for_subject(subject_id)

    if is_minute_averaged:
        filename = "{}_airspeck_personal_{}{}.csv".format(label_files, upload_type, suffix_filename)
    else:
        filename = "{}_airspeck_personal_{}_raw{}.csv".format(label_files, upload_type, suffix_filename)

    print("Loading file: {}".format(project_mapping[project_name][2] + filename))
    data = load_airrespeck_file(project_mapping[project_name][2] + filename,
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
        data.loc[data['humidity'] > 105, 'humidity'] = np.nan

    if calibrate_pm_and_gas and return_calibration_flag:
        return result_date, was_calibrated_pm, data
    else:
        return data


def load_respeck_file(subject_id, project_name=None, filter_out_not_worn=True, subject_visit_number=None,
                      upload_type='automatic', suffix_filename="", raw_file=False):
    '''
    Load a Respeck csv file to a pandas dataframe in the correct timezone
    :param subject_id: 6-character subject ID
    :param project_name: For some projects, this is the actual name "daphne", for others, it's the project ID.
    See constants.py for a list of all project names
    :param filter_out_not_worn: Whether to filter out those periods where the Respeck was most likely not worn. These
    are the periods where the activity level is below a threshold for some time.
    :param subject_visit_number: Which of several recordings of a subject should be loaded.
    Only relevant for some projects like Daphne
    :return: Respeck data as pandas dataframe.
    '''

    if subject_visit_number is None:
        label_files = "{}".format(subject_id)
    else:
        label_files = "{}({})".format(subject_id, int(subject_visit_number))

    if project_name is None:
        project_name = get_project_for_subject(subject_id)

    if raw_file:
        filename = "{}_respeck_{}_raw{}.csv".format(label_files, upload_type, suffix_filename)
    else:
        filename = "{}_respeck_{}{}.csv".format(label_files, upload_type, suffix_filename)

    print("Loading file: {}".format(project_mapping[project_name][2] + filename))
    respeck_data = load_airrespeck_file(project_mapping[project_name][2] + filename, project_name)

    if respeck_data is not None and filter_out_not_worn and len(respeck_data) > 0:
        set_breathing_rate_nan_when_lying_on_stomach(respeck_data)
        set_breathing_rate_nan_when_not_worn(respeck_data)

    return respeck_data


def load_diary_file(subject_id, project_name, subject_visit_number=None):
    if subject_visit_number is None:
        label_files = "{}".format(subject_id)
    else:
        label_files = "{}({})".format(subject_id, subject_visit_number)

    diary_data = load_airrespeck_file(project_mapping[project_name][2] + "{}_diary.csv".format(label_files),
                                      project_name)
    return diary_data


def load_static_airspeck_file(sid_or_uuid, project_name=None, sensor_label=None, suffix_filename="",
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
        filename = "{}_static_airspeck_{}{}.csv".format(sensor_label, upload_type, suffix_filename)

    print("Loading file: {}".format(project_mapping[project_name][2] + filename))
    data = load_airrespeck_file(project_mapping[project_name][2] + filename, project_name)
    
    #Sdata = filter_out_outliers_gas(data)

    if calibrate_pm or calibrate_ox or calibrate_no2:
        result_date, was_calibrated_pm, was_calibrated_no2, was_calibrated_ox,  data = calibrate_airspeck(
            sid_or_uuid, data, calibrate_pm=calibrate_pm, calibrate_no2=calibrate_no2, 
            calibrate_ox=calibrate_ox,project_name=project_name, calibration_id=calibration_id,
            use_all_features_pm=use_all_features_for_pm_calibration,
            use_all_features_gas=use_all_features_for_gas_calibration, country_name=country_name)

        if return_calibration_flag:
            return result_date, was_calibrated_pm,  was_calibrated_no2, was_calibrated_ox, data

    return data


def load_merged_personal_data(subject_id, project_name=None, upload_type='automatic', subject_visit_number=None,
                              calibrate_pm=False, use_all_features_for_pm_calibration=False):
    # Load both personal airspeck and respeck data, and join them on the timestamp index.
    airspeck = load_personal_airspeck_file(subject_id, project_name, upload_type=upload_type,
                                           subject_visit_number=subject_visit_number,
                                           calibrate_pm_and_gas=calibrate_pm,
                                           use_all_features_for_pm_calibration=use_all_features_for_pm_calibration)
    respeck = load_respeck_file(subject_id, project_name, upload_type=upload_type,
                                subject_visit_number=subject_visit_number)

    if airspeck is None or respeck is None:
        return None
    else:
        data = airspeck.join(respeck, how='outer', lsuffix='_airspeck', rsuffix='_respeck')
        return data


def load_gps_file(subject_id, project_name, subject_visit_number=1):
    label_files = "{}({})".format(subject_id, subject_visit_number)
    return load_airrespeck_file(project_mapping[project_name][2] + "{}_gps.csv".format(label_files), project_name)


# This is a helper function which doesn't have to be called directly.
def load_airrespeck_file(filepath, project_name, timestamp_column_name='timestamp'):
    try:
        data = pd.read_csv(filepath)
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


# Load files created with the Respeck activity logging screen in the AirRespeck app.
def load_activity_logs_periods(path):
    periods = []
    for dp, dn, filenames in os.walk(path):
        for f in filenames:
            if "Activity RESpeck Logs" in f:
                periods.extend(load_periods_from_respeck_activity_log(dp + "/" + f, cut_edges=37))
    return periods


# cut_edges defines amount of data cut off at beginnig and end of period
def load_periods_from_respeck_activity_log(path, cut_edges=0):
    logs = pd.read_csv(path)
    logs = logs.set_index('timestamp')

    periods = []
    recording = []
    for idx, row in logs.iterrows():
        if row['subjectName'] == 'end of record':
            df_recording = pd.DataFrame(recording, columns=list(logs))
            periods.append((recording[0][5], df_recording.iloc[cut_edges:-cut_edges]))
            recording = []
        else:
            recording.append(row.values)
    return periods


def set_breathing_rate_nan_when_lying_on_stomach(respeck_data):
    # When activity type is lying on stomach, we can't trust the breathing rate!
    respeck_data.loc[respeck_data['activity_type'] == 8, 'breathing_rate'] = np.nan


def set_breathing_rate_nan_when_not_worn(respeck_data):
    # Add nan rows for missing data
    respeck_data_resampled = respeck_data.resample('1Min').mean()

    # 5 minute averages after filling in missing 1 min periods
    resampled_data_mean = respeck_data_resampled.resample('5Min').mean()

    # 5 minute periods where the sensor is likely not worn
    not_worn_pred = ((resampled_data_mean['activity_level'] < 0.013) & (resampled_data_mean['breathing_rate'] > 24.5))

    # 5 minute periods which are surrounded by not_worn periods, are probably also not worn
    not_worn_interpolated = (not_worn_pred.shift(-1) & not_worn_pred.shift(1)) | not_worn_pred

    # Upsample to 1 minute again and set breathing rate for not worn periods to nan
    back_sampled = not_worn_interpolated.resample('1Min').pad()
    # Add 4 minutes in the end which might be lost due to downsampling
    back_sampled = back_sampled.append(pd.DataFrame(data=[back_sampled.iloc[-1]] * 4,
                                                    index=pd.date_range(
                                                        start=back_sampled.index[-1] + pd.DateOffset(minutes=1),
                                                        periods=4, freq="1Min")))

    # Cut off beginning and end as in original data
    not_worn_final = back_sampled[respeck_data.index[0]:respeck_data.index[-1]]

    print("{} minutes of breathing rate set to nan because the sensor was likely not worn.".format(
        np.count_nonzero(not_worn_final[0])))

    # Set breathing rate to nan when sensor was not worn
    respeck_data.loc[not_worn_final[0], 'breathing_rate'] = np.nan
    respeck_data.loc[not_worn_final[0], 'activity_type'] = 3


def get_uuid_for_subj_id_airspeck_personal(subject_id, timestamp=None):
    # Look in Google storage for Airspeck file with that subject ID.
    # Extract the UUID from the first one which comes up.
    # Import here as it can clash with datastore imports

    # There was no file for this subject. Check the automatic upload
    client = get_datastore_client()
    filters = [('subject_id', '=', subject_id)]
    if timestamp is not None:
        filters.extend([('timestamp', '>=', timestamp - timedelta(hours=1)),
                        ('timestamp', '<', timestamp + timedelta(hours=1))])

    # Try manual upload first
    kind = 'MobileAirspeckManualUpload'
    query = client.query(kind=kind, filters=filters, order=['timestamp'])
    result = list(query.fetch(1))

    if len(result) == 0:
        # If there was no data there, try automatic upload
        kind = 'MobileAirspeck'
        query = client.query(kind=kind, filters=filters, order=['timestamp'])
        result = list(query.fetch(1))

    if len(result) > 0:
        uuid = result[0]['airspeck_uuid'].replace(':', '')
        if len(uuid) == 21:
            print("Extracted UUID {} (Length {})".format(uuid, len(uuid)))
            return uuid[5:]
        else:
            print("Extracted UUID {} (Length {})".format(uuid, len(uuid)))
            return uuid


###################
# The below calibration functions cannot be moved to sensor_calibration.py due to resulting circular imports.
###################

def calibrate_airspeck(subj_id_or_uuid, airspeck_data, project_name, calibrate_pm=False, calibrate_no2=False, calibrate_ox=False,
                       calibration_id=None, use_all_features_pm=False, use_all_features_gas=True,
                       country_name=None):
    if airspeck_data is None or len(airspeck_data) == 0:
        print("No data passed, so couldn't calibrate.")
        return False, False, False, False, airspeck_data

    if len(subj_id_or_uuid) == 6:
        uuid = get_uuid_for_subj_id_airspeck_personal(subj_id_or_uuid, airspeck_data.index[0])
    else:
        uuid = subj_id_or_uuid

    if uuid is None:
        print("Couldn't be calibrated, as matching UUID was not found in datastore")
        return False, False, False, False, airspeck_data
    
    result_date = False
    was_calibrated_pm = False
    was_calibrated_no2 = False
    was_calibrated_ox = False

    if airspeck_data is not None and len(airspeck_data) > 0:
        # When getting calibration data for a whole project, the same uuids are reused. Store these to
        # not have to load them again every time.
        result_date, factors = get_calibration_factors_airspeck(uuid,
                                                   to_be_calibrated_timestamp=airspeck_data.iloc[-1][
                                                       'timestamp'].to_pydatetime(),
                                                   project_name=project_name, country_name=country_name, 
                                                    calibrate_pm=calibrate_pm, calibrate_no2=calibrate_no2,
                                                    calibrate_ox=calibrate_ox, use_all_features_pm=use_all_features_pm,
                                                   use_all_features_gas=use_all_features_gas,
                                                   calibration_id=calibration_id)

        was_calibrated_pm, was_calibrated_no2, was_calibrated_ox, airspeck_data = calibrate_with_factors(airspeck_data, factors)

    return result_date, was_calibrated_pm, was_calibrated_no2, was_calibrated_ox, airspeck_data



def get_calibration_factors_airspeck(subj_or_uuid, to_be_calibrated_timestamp=None, project_name=None, calibrate_pm=False,
                                     calibrate_no2=False, calibrate_ox=False,
                                     country_name=None, use_all_features_pm=False,
                                     use_all_features_gas=True, calibration_id=None):
    if len(subj_or_uuid) == 6:
        uuid = get_uuid_for_subj_id_airspeck_personal(subj_or_uuid, timestamp=to_be_calibrated_timestamp)
    else:
        uuid = subj_or_uuid

    # Convert to_be_calibrated_timestamp into UTC
    tz = timezone(str(to_be_calibrated_timestamp.tzinfo))
    results_pm = []
    result_date = False

    if calibrate_pm:
    ##Query for PM
        filters = [('uuid', '=', uuid)]
        if country_name is not None:
            filters.append(('country_name', '=', country_name))
        if calibration_id is not None:
            filters.append(('calibration_id', '=', calibration_id))

        client = get_datastore_client()
        results_pm = list(client.query(kind='AirspeckCalibrationFactors', filters=filters,
                                   order=['-time_of_calibration']).fetch())
    
        
    if len(results_pm) > 0 and not (len(results_pm) == 1 and results_pm[0]['calibration_id'] == '2019-11 EBAM IITD'):
        #if len(results_pm) == 1 and results_pm[0]['calibration_id'] != '2019-11 EBAM IITD':
        #    result_pm = results_pm[0]
        #else:
            # Choose calibration factors which time of calibration was nearest to to_be_calibrated_timestamp
        result_to_use_idx = 0
        time_difference = results_pm[0]['time_of_calibration'].astimezone(tz) - to_be_calibrated_timestamp
            
        if len(results_pm) > 1:
            for idx, result in enumerate(results_pm[1:]):
                if result['calibration_id'] == '2019-11 EBAM IITD':
                    continue
                    
                   
                new_time_difference = result['time_of_calibration'].astimezone(tz) - to_be_calibrated_timestamp
                if abs(new_time_difference.days) < abs(time_difference.days):
                    time_difference = new_time_difference
                    result_to_use_idx = idx + 1

        result_pm = results_pm[result_to_use_idx]
        result_date = result_pm['time_of_calibration'].astimezone(tz).date()

            
        print("--> Chose calibration factors (time difference {}): {}".format(
            result_pm['time_of_calibration'].astimezone(tz) - to_be_calibrated_timestamp, result_pm['calibration_id']))

        low_humidity_factors_pm = get_from_entity_if_present(result_pm, 'low_humidity_factors_simple')

        if use_all_features_pm or low_humidity_factors_pm == None:
            print("Using calibration factors for all {} features".format(len(calibration_columns_pm)))
            low_humidity_factors_pm = get_from_entity_if_present(result_pm, 'low_humidity_factors_all')
            high_humidity_factors_pm = get_from_entity_if_present(result_pm, 'high_humidity_factors_all')
        else:
            print("Using simple calibration factors")
            low_humidity_factors_pm = get_from_entity_if_present(result_pm, 'low_humidity_factors_simple')
            high_humidity_factors_pm = get_from_entity_if_present(result_pm, 'high_humidity_factors_simple')
        humidity_threshold_pm = get_from_entity_if_present(result_pm, 'humidity_threshold')
    else:
        print("No PM calibration data available for sensor {}".format(uuid))
        # Return "identity" factors
        if use_all_features_pm:
            low_humidity_factors_pm = np.append([1.] * len(calibration_columns_pm), [0.])
            high_humidity_factors_pm = np.append([1.] * len(calibration_columns_pm), [0.])
        else:
            low_humidity_factors_pm = [1., 0.]
            high_humidity_factors_pm = [1., 0.]
        humidity_threshold_pm = 1000.
        
    result_both_gases = []
    
    if calibrate_ox or calibrate_no2:
        filters_both = [('uuid', '=', uuid)]
        filters_both.append(('calibrated_data_type', '=', 'both'))
        if country_name is not None:
            filters_both.append(('country_name', '=', country_name))
        if calibration_id is not None:
            filters_both.append(('calibration_id', '=', calibration_id))
            
        result_both_gases = list(client.query(kind='AirspeckCalibrationFactorsGas', filters=filters_both,
                                   order=['-time_of_calibration']).fetch())

    results_no2 = []
    if calibrate_no2:
        ## Query for NO2
        filters_no2 = [('uuid', '=', uuid)]
        filters_no2.append(('calibrated_data_type', '=', 'no2'))
        #filters_no2.append(('calibrated_data_type', '=', 'both'))
        if country_name is not None:
            filters_no2.append(('country_name', '=', country_name))
        if calibration_id is not None:
            filters_no2.append(('calibration_id', '=', calibration_id))
        
        results_no2 = list(client.query(kind='AirspeckCalibrationFactorsGas', filters=filters_no2,
                                   order=['-time_of_calibration']).fetch())
        
        results_no2 = results_no2 + result_both_gases

    
    if len(results_no2) > 0:
        
        if len(results_no2) == 1:
            result_no2 = results_no2[0]
        else:
            # Choose calibration factors which time of calibration was nearest to to_be_calibrated_timestamp
            result_to_use_idx = 0
            time_difference = results_no2[0]['time_of_calibration'].astimezone(tz) - to_be_calibrated_timestamp
            
            if len(results_no2) > 1:
                for idx, result in enumerate(results_no2[1:]):
                   
                    new_time_difference = result['time_of_calibration'].astimezone(tz) - to_be_calibrated_timestamp
                    if abs(new_time_difference.days) < abs(time_difference.days):
                        time_difference = new_time_difference
                        result_to_use_idx = idx + 1

            result_no2 = results_no2[result_to_use_idx]
            result_date = result_no2['time_of_calibration'].astimezone(tz).date()

        print("--> Chose calibration factors (time difference {}): {}".format(
            result_no2['time_of_calibration'].astimezone(tz) - to_be_calibrated_timestamp, result_no2['calibration_id']))
        
        if use_all_features_gas:
            print("Using gas calibration factors for all {} features for no2.".format(len(calibration_columns_no2)))
            low_humidty_factors_no2 = get_from_entity_if_present(result_no2, 'low_humidity_factors_all_no2')
            high_humidity_factors_no2 = get_from_entity_if_present(result_no2, 'high_humidity_factors_all_no2')
        else:
            print("Using simple gas calibration factors for no2")
            low_humidty_factors_no2 = get_from_entity_if_present(result_no2, 'low_humidity_factors_simple_no2')
            high_humidity_factors_no2 = get_from_entity_if_present(result_no2, 'high_humidity_factors_simple_no2')
        humidity_threshold_gas = get_from_entity_if_present(result_no2, 'humidity_threshold')

    else:
        print("No NO2 calibration data available for sensor {}".format(uuid))
        # Return np.nan, as the uncalibrated gas data isn't usable
        # 1000 is set as humidity threshold which is never reached,
        # i.e. all will be calibrated with the low humidity factors
        low_humidty_factors_no2 = []
        high_humidity_factors_no2 = []
        humidity_threshold_gas = 1000.
    
    results_ox = []
    if calibrate_ox:
    ## Query for OX
        filters_ox = [('uuid', '=', uuid)]
        filters_ox.append(('calibrated_data_type', '=', 'ox'))
        if country_name is not None:
            filters_ox.append(('country_name', '=', country_name))
        if calibration_id is not None:
            filters_ox.append(('calibration_id', '=', calibration_id))
        
        results_ox = list(client.query(kind='AirspeckCalibrationFactorsGas', filters=filters_ox,
                                   order=['-time_of_calibration']).fetch())
        
        results_ox = results_ox + result_both_gases
        
    if len(results_ox) > 0:
        
        if len(results_ox) == 1:
            result_ox = results_ox[0]
        else:
            # Choose calibration factors which time of calibration was nearest to to_be_calibrated_timestamp
            result_to_use_idx = 0
            time_difference = results_ox[0]['time_of_calibration'].astimezone(tz) - to_be_calibrated_timestamp
            
            if len(results_ox) > 1:
                for idx, result in enumerate(results_ox[1:]):
                   
                    new_time_difference = result['time_of_calibration'].astimezone(tz) - to_be_calibrated_timestamp
                    if abs(new_time_difference.days) < abs(time_difference.days):
                        time_difference = new_time_difference
                        result_to_use_idx = idx + 1

            result_ox = results_ox[result_to_use_idx]
            result_date = result_ox['time_of_calibration'].astimezone(tz).date()

        print("--> Chose calibration factors (time difference {}): {}".format(
            result_ox['time_of_calibration'].astimezone(tz) - to_be_calibrated_timestamp, result_ox['calibration_id']))
        
        if use_all_features_gas:
            print("Using gas calibration factors for all {} features for ox.".format(len(calibration_columns_ox)))
            low_humidity_factors_ox = get_from_entity_if_present(result_ox, 'low_humidity_factors_all_ox')
            high_humidity_factors_ox = get_from_entity_if_present(result_ox, 'high_humidity_factors_all_ox')
        else:
            print("Using simple gas calibration factors for ox")
            low_humidity_factors_ox = get_from_entity_if_present(result_ox, 'low_humidity_factors_simple_ox')
            high_humidity_factors_ox = get_from_entity_if_present(result_ox, 'high_humidity_factors_simple_ox')
        humidity_threshold_gas = get_from_entity_if_present(result_ox, 'humidity_threshold')

    else:
        print("No OX calibration data available for sensor {}".format(uuid))
        # Return np.nan, as the uncalibrated gas data isn't usable
        # 1000 is set as humidity threshold which is never reached,
        # i.e. all will be calibrated with the low humidity factors
        low_humidity_factors_ox = []
        high_humidity_factors_ox = []
        humidity_threshold_gas = 1000.

    return result_date, [low_humidity_factors_pm, high_humidity_factors_pm, humidity_threshold_pm, low_humidty_factors_no2, \
           high_humidity_factors_no2, low_humidity_factors_ox, high_humidity_factors_ox, humidity_threshold_gas]


def has_been_calibrated(subj_id, project_name=None, country_name=None, calibration_id=None):
    factors = get_calibration_factors_airspeck(subj_id, project_name, country_name, calibration_id)
    was_calibrated_pm = ~(factors[0][0] == 1.0 and factors[0][-1] == 0.0)
    was_calibrated_gas = ~np.isnan(factors[3])
    return was_calibrated_pm, was_calibrated_gas


def calibrate_with_factors(data, factors):
    print("Factors: ", factors)
    low_humidity_factors_pm, high_humidity_factors_pm, humidity_threshold_pm, low_humidity_factors_no2, \
    high_humidity_factors_no2, low_humidity_factors_ox, high_humidity_factors_ox, humidity_threshold_gas = factors

    high_humidity_mask_pm = data['humidity'] > humidity_threshold_pm

    print("Calibrating PM2.5 with factors (low hum): {}".format(low_humidity_factors_pm))

    # Do we have high humidity factors for this sensor?
    if len(high_humidity_factors_pm) == 0: #If not, we will calibrate with low huidity factors only.
        # Are there high humidity periods?
        if np.count_nonzero(high_humidity_mask_pm) > 0:
            # We have at least one case with a humidity above the threshold, but no high humidity calibration factors
            # Print a warning.
            print("{}% of data points are above the humidity threshold of {}, but no calibration factors "
                  "for high humidity exist for this sensor. Using low humidity calibration for all samples.".format(
                round(np.count_nonzero(high_humidity_mask_pm) / float(len(high_humidity_mask_pm)) * 100, 0),
                humidity_threshold_pm
            ))
        if len(low_humidity_factors_pm) > 2:
            # We use all features for calibration
            data.loc[:, 'pm2_5'] = np.dot(data.loc[:, calibration_columns_pm], low_humidity_factors_pm[:-1]) + \
                                   low_humidity_factors_pm[-1]
        else:
            data.loc[:, 'pm2_5'] = data.loc[:, 'pm2_5'] * low_humidity_factors_pm[0] + low_humidity_factors_pm[1]
    else:
        # Calibrate low and high humidity separately
        # Are we using all features for calibration?
        if len(low_humidity_factors_pm) > 2:
            # We use all features for calibration
            data.loc[~high_humidity_mask_pm, 'pm2_5'] = np.dot(
                data.loc[~high_humidity_mask_pm, calibration_columns_pm],
                low_humidity_factors_pm[:-1]) + low_humidity_factors_pm[-1]
            data.loc[high_humidity_mask_pm, 'pm2_5'] = np.dot(
                data.loc[high_humidity_mask_pm, calibration_columns_pm],
                high_humidity_factors_pm[:-1]) + high_humidity_factors_pm[-1]
        else:
            # Simple calibration
            data.loc[~high_humidity_mask_pm, 'pm2_5'] = data.loc[~high_humidity_mask_pm, 'pm2_5'] * \
                                                        low_humidity_factors_pm[0] + low_humidity_factors_pm[1]
            data.loc[high_humidity_mask_pm, 'pm2_5'] = data.loc[high_humidity_mask_pm, 'pm2_5'] * \
                                                       high_humidity_factors_pm[0] + high_humidity_factors_pm[1]

    ## Calibrate NO2
    was_calibrated_no2 = False
    if len(low_humidity_factors_no2) > 0:
        was_calibrated_no2 = True
        
        #data = filter_out_outliers_gas(data)
        print("Calibrating NO2 with factors (low hum): {}".format(low_humidity_factors_no2))
        high_humidity_mask_gas = data['humidity'] > humidity_threshold_gas

        # Do we have high humidity factors for this sensor?
        if len(high_humidity_factors_no2) == 0:
            # Are there high humidity periods?
            if np.count_nonzero(high_humidity_mask_gas) > 0:
                # We have at least one case with a humidity above the threshold, but no high humidity calibration factors
                # Print a warning.
                print("{}% of data points are above the humidity threshold of {}, but no calibration factors "
                      "for high humidity exist for this sensor. Using low humidity calibration for all samples.".format(
                    round(np.count_nonzero(high_humidity_mask_gas) / float(len(high_humidity_mask_gas)) * 100, 0),
                    humidity_threshold_gas
                ))
            # Do we use all features or just the WE?
            if len(low_humidity_factors_no2) > 2:
                # We use all features for calibration
                data.loc[:, 'no2'] = np.dot(data.loc[:, calibration_columns_no2], low_humidity_factors_no2[:-1]) + \
                                     low_humidity_factors_no2[-1]

            else:
                data.loc[:, 'no2'] = data.loc[:, 'no2_we'] * low_humidity_factors_no2[0] + low_humidity_factors_no2[1]
        else:
            # Calibrate low and high humidity separately
            # Do we use all features or just the WE?
            if len(low_humidity_factors_no2) > 2:
                # We use all features for calibration
                data.loc[~high_humidity_mask_gas, 'no2'] = np.dot(
                    data.loc[~high_humidity_mask_gas, calibration_columns_no2],
                    low_humidity_factors_no2[:-1]) + low_humidity_factors_no2[-1]
                data.loc[high_humidity_mask_gas, 'no2'] = np.dot(
                    data.loc[high_humidity_mask_gas, calibration_columns_no2],
                    high_humidity_factors_no2[:-1]) + high_humidity_factors_no2[-1]

            else:
                data.loc[~high_humidity_mask_gas, 'no2'] = data.loc[~high_humidity_mask_gas, 'no2_we'] * \
                                                           low_humidity_factors_no2[0] + low_humidity_factors_no2[1]

                data.loc[high_humidity_mask_gas, 'no2'] = data.loc[high_humidity_mask_gas, 'no2_we'] * \
                                                          high_humidity_factors_no2[0] + high_humidity_factors_no2[1]
                
        # Filter out negative values
        data.loc[data['no2'] < 0, 'no2'] = np.nan
        
    ##Calibrate OX
    was_calibrated_ox = False
    if len(low_humidity_factors_ox) > 0:
        was_calibrated_ox = True
        
        #data = filter_out_outliers_gas(data)
        print("Calibrating OX with factors (low hum) {}".format(low_humidity_factors_ox))
        high_humidity_mask_gas = data['humidity'] > humidity_threshold_gas

        # Do we have high humidity factors for this sensor?
        if len(high_humidity_factors_ox) == 0:
            # Are there high humidity periods?
            if np.count_nonzero(high_humidity_mask_gas) > 0:
                # We have at least one case with a humidity above the threshold, but no high humidity calibration factors
                # Print a warning.
                print("{}% of data points are above the humidity threshold of {}, but no calibration factors "
                      "for high humidity exist for this sensor. Using low humidity calibration for all samples.".format(
                    round(np.count_nonzero(high_humidity_mask_gas) / float(len(high_humidity_mask_gas)) * 100, 0),
                    humidity_threshold_gas
                ))
            # Do we use all features or just the WE?
            if len(low_humidity_factors_ox) > 2:
                # We use all features for calibration
                data.loc[:, 'ox'] = np.dot(data.loc[:, calibration_columns_ox], low_humidity_factors_ox[:-1]) + \
                                    low_humidity_factors_ox[-1]
            else:
                data.loc[:, 'ox'] = data.loc[:, 'ox_we'] * low_humidity_factors_ox[0] + low_humidity_factors_ox[1]
        else:
            # Calibrate low and high humidity separately
            # Do we use all features or just the WE?
            if len(low_humidity_factors_ox) > 2:
                # We use all features for calibration
                data.loc[~high_humidity_mask_gas, 'ox'] = np.dot(
                    data.loc[~high_humidity_mask_gas, calibration_columns_ox],
                    low_humidity_factors_ox[:-1]) + low_humidity_factors_ox[-1]

                data.loc[high_humidity_mask_gas, 'ox'] = np.dot(
                    data.loc[high_humidity_mask_gas, calibration_columns_ox],
                    high_humidity_factors_ox[:-1]) + high_humidity_factors_ox[
                                                             -1]
            else:
              
                data.loc[~high_humidity_mask_gas, 'ox'] = data.loc[~high_humidity_mask_gas, 'ox_we'] * \
                                                          low_humidity_factors_ox[0] + low_humidity_factors_ox[1]

                data.loc[high_humidity_mask_gas, 'ox'] = data.loc[high_humidity_mask_gas, 'ox_we'] * \
                                                         high_humidity_factors_ox[0] + high_humidity_factors_ox[1]

        # Filter out negative values
        data.loc[data['ox'] < 0, 'ox'] = np.nan

    was_calibrated_pm = not (low_humidity_factors_pm[0] == 1.0 and low_humidity_factors_pm[-1] == 0.0)

    return was_calibrated_pm, was_calibrated_no2, was_calibrated_ox, data
