'''
These functions are from Meeke Roet's thesis project from 2018.
They are used to generate correlation response graphs for Dublin and don't need to be used for anything else.
'''
import copy
import re
from math import radians

import numpy as np
import pandas as pd

import project_specific.meeke_helper_functions_dublin as hf


def prepare_minute_respeck(raw_df, subj_id, project='apcaps', partial_outliers=None):
    """ Prepare a raw dataframe of minute RESpeck data by setting the timestamp as the index and creating rows for missing minutes.
    """

    # Convert timestamp to datetime type.
    if project == 'apcaps':
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp']) + pd.DateOffset(hours=5, minutes=30)
    elif project == 'dublin':
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp']) + pd.DateOffset(hours=1, minutes=0)

    # Set timestamp as the index to be able to select specific time periods (e.g. night).
    raw_df.set_index('timestamp', inplace=True)

    # Select only data from this year to circumvent issues with wrong dates.
    raw_df = raw_df.query("'2018-01-01' <= index <= '2020-01-01'")

    # Add to outlier list if the recording is shorter than two hours.
    if len(raw_df) < 120:
        print("Subject {} has less than 120 minutes of recording".format(subj_id))
        # hf.add_outlier_to_list(subj_id, 'Recording less than two hours', project='apcaps')

    # Remove parts of the recording that were bad if any.
    if isinstance(partial_outliers, dict):
        if subj_id in partial_outliers.keys():
            if any(isinstance(el, list) for el in partial_outliers[subj_id]) == True:
                for stretch in partial_outliers[subj_id]:
                    raw_df.loc[stretch[0]:stretch[1]] = np.nan
            else:
                raw_df.loc[partial_outliers[subj_id][0]:partial_outliers[subj_id][1]] = np.nan

    # Add indices for missing minutes.
    raw_df = raw_df.reindex(pd.date_range(raw_df.index[0], raw_df.index[-1], freq='min'), fill_value=np.nan)
    mins_without_data = raw_df.isnull().sum(axis=1) == len(raw_df.columns)
    raw_df['data_recorded'] = 1
    raw_df.loc[mins_without_data, 'data_recorded'] = 0

    # Create column with the simple activity types (undetermined, sitting/standing, walking, lying down, wrong orientation, movement).
    raw_df['activity_type'].loc[raw_df['activity_type'].isnull()] = -1
    raw_df['activity_type_extended'] = raw_df['activity_type'].values
    activity_mapping_dict = {-1: -1, 0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 0, 6: 2, 7: 2, 8: 2, 9: 4}
    raw_df['activity_type'] = raw_df['activity_type_extended'].map(activity_mapping_dict)

    # Add subject ID.
    raw_df['subj_id'] = subj_id

    # if project == 'dublin':
    #     raw_df = create_lagged_and_ahead_vars(raw_df, n_lags=30, n_ahead=30)

    return raw_df


def prepare_minute_airspeck(raw_df, subj_id, project='apcaps'):
    """ Prepare a raw dataframe of minute RESpeck data by setting the timestamp as the index and creating rows for missing minutes.
    """

    # Convert timestamp to datetime type.
    if project == 'apcaps':
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp']) + pd.DateOffset(hours=5, minutes=30)
    elif project == 'dublin':
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp']) + pd.DateOffset(hours=1, minutes=0)

    # Set timestamp as the index to be able to select specific time periods (e.g. night).
    raw_df.set_index('timestamp', inplace=True, drop=False)

    # Get minutely averages.
    df = raw_df.resample(rule='Min', on='timestamp').mean()

    # Add indices for missing minutes.
    df = df.query("'2018-01-01' <= index <= '2020-01-01'")
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='min'), fill_value=np.nan)

    # Add subject ID.
    df['subj_id'] = subj_id

    return df


def create_lagged_and_ahead_vars(subj_data, n_lags=30, n_ahead=10, variables=['breathing_rate', 'activity_level']):
    # Create lagged variables.
    for t in range(n_lags):
        t += 1

        if 'breathing_rate' in variables:
            subj_data['br_lag{}'.format(t)] = subj_data['breathing_rate'].shift(t)
        if 'activity_level' in variables:
            subj_data['al_lag{}'.format(t)] = subj_data['activity_level'].shift(t)
        if 'activity_type' in variables:
            subj_data['at_lag{}'.format(t)] = subj_data['activity_type'].shift(t)
        if 'step_count' in variables:
            subj_data['sc_lag{}'.format(t)] = subj_data['step_count'].shift(t)
        if 'pm1' in variables:
            subj_data['pm1_lag{}'.format(t)] = subj_data['pm1'].shift(t)
        if 'pm2_5' in variables:
            subj_data['pm2_5_lag{}'.format(t)] = subj_data['pm2_5'].shift(t)
        if 'pm10' in variables:
            subj_data['pm10_lag{}'.format(t)] = subj_data['pm10'].shift(t)

    # Create breathing rate ahead.
    for t in range(n_ahead):
        t += 1

        if 'breathing_rate' in variables:
            subj_data['br_ahead{}'.format(t)] = subj_data['breathing_rate'].shift(-t)
        if 'activity_level' in variables:
            subj_data['al_ahead{}'.format(t)] = subj_data['activity_level'].shift(-t)
        if 'activity_type' in variables:
            subj_data['at_ahead{}'.format(t)] = subj_data['activity_type'].shift(-t)
        if 'step_count' in variables:
            subj_data['sc_ahead{}'.format(t)] = subj_data['step_count'].shift(-t)
        if 'pm1' in variables:
            subj_data['pm1_ahead{}'.format(t)] = subj_data['pm1'].shift(-t)
        if 'pm2_5' in variables:
            subj_data['pm2_5_ahead{}'.format(t)] = subj_data['pm2_5'].shift(-t)
        if 'pm10' in variables:
            subj_data['pm10_ahead{}'.format(t)] = subj_data['pm10'].shift(-t)

        # subj_data['br_ahead{}'.format(t)] = subj_data['breathing_rate'].shift(-t)

    return subj_data


def add_laboured_breathing_to_subj_data(subj_data, window_size=5, laboured_breathing_cutoff=30):
    # Apply mean and median convolution filter.
    subj_data.loc[:, 'convoluted_mean_br'] = subj_data['breathing_rate'].rolling(center=False,
                                                                                 window=window_size).mean()
    subj_data.loc[:, 'convoluted_median_br'] = subj_data['breathing_rate'].rolling(center=False,
                                                                                   window=window_size).median()

    # Find laboured breathing episodes.
    subj_data['laboured_breathing'] = subj_data.loc[:, 'convoluted_mean_br'] > laboured_breathing_cutoff

    return subj_data


def add_movement_to_subj_data(subj_data):
    subj_data['movement'] = hf.element_wise_haversine(subj_data['lat'], subj_data['long'],
                                                      subj_data['lat'].shift(1), subj_data['long'].shift(1))

    return subj_data


def add_loc_type_to_subj_data(subj_data, loc_list):
    subj_data['loc_type'] = loc_list['category'][subj_data['closest_loc']].values

    return subj_data


def replace_invalid_walking_with_movement(subj_data):
    """ Replace false walking classifications by 'Movement', where it's considered false if:

        - activity type = 1 ('walking') and movement is >200
        or
        - activity type = 1 ('walking') and the subject moved too far relative to the step count.
    """

    idx_false_walking = ((subj_data['activity_type'] == 1) & (
                subj_data['movement'] > 1.2 * subj_data['step_count'])) | (
                                (subj_data['activity_type'] == 1) & (subj_data['movement'] > 200))

    # Set activity to wrong orientation
    subj_data.loc[idx_false_walking, 'activity_type'] = 3

    return subj_data


def add_turning_while_lying_dummy_to_subj_data(subj_data):
    # Get the minutes where the person turned by finding the moments where they were lying down for two consecutive minutes, but in a different lying down type.
    turning_minutes = (np.isin(subj_data['activity_type_extended'], [2, 6, 7, 8])) & (
        np.isin(subj_data['activity_type_extended'].shift(1), [2, 6, 7, 8])) & (
                                  subj_data['activity_type_extended'] != subj_data['activity_type_extended'].shift(1))

    # Create a dummy indicating whether or not the subject turned in each minute.
    subj_data['dummy_turn'] = False
    subj_data.loc[turning_minutes, 'dummy_turn'] = True

    return subj_data


def add_day_night_dummies_to_subj_data(subj_data, day_times=['08:00', '20:00'], night_times=['00:00', '05:00']):
    """ For each minute, add a dummy indicating if the person was sleeping during the minute (1) or not (0).
    Sleeping is defined based on the time and activity type lying down.
    """

    # Single out night + lying data, and day data.
    night_idx = subj_data[subj_data['activity_type'] == 2].between_time(night_times[0], night_times[1],
                                                                        include_start=True, include_end=False).index
    day_idx = subj_data.between_time(day_times[0], day_times[1], include_start=True, include_end=False).index

    # Remove day index items that are also in the night index (i.e lying down early in the morning, when the time frames overlap).
    day_idx = day_idx.drop(np.intersect1d(night_idx, day_idx))

    # Create dummies indicating night and day.
    subj_data['night_dummy'] = 0
    subj_data['day_dummy'] = 0
    subj_data.loc[night_idx, 'night_dummy'] = 1
    subj_data.loc[day_idx, 'day_dummy'] = 1

    return subj_data


def prepare_phone_gps(raw_df, project='apcaps'):
    """ Load phone GPS data and remove intervals where the location didn't change.
    """

    # Convert timestamp to datetime type.
    if project == 'apcaps':
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp']) + pd.DateOffset(hours=5, minutes=30)
    elif project == 'dublin':
        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp']) + pd.DateOffset(hours=1, minutes=0)

    # Find intervals where the GPS location didn't change and keep only the first and last timestamp from them.
    # Saves computational effort later on.
    idx_to_remove = (
                            (raw_df['longitude'].shift() == raw_df['longitude']) &
                            (raw_df['latitude'].shift() == raw_df['latitude'])) & (
                            (raw_df['longitude'].shift(-1) == raw_df['longitude']) &
                            (raw_df['latitude'].shift(-1) == raw_df['latitude'])
                    )

    raw_df = raw_df.loc[~idx_to_remove]

    #         # Set timestamp as the index to be able to select specific time periods (e.g. night).
    #         raw_df.set_index('timestamp', inplace = True)

    return raw_df


def load_apcaps_location_list(file):
    """ Load the list of annotated locations in the village from APCAPS and convert to decimal degrees coordinates.
    """

    # Load annotated GPS locations from the village.
    gps_locs = pd.read_csv(file, usecols=range(1, 5))

    # Parse GPS coordinates from DMS (degrees, minutes, seconds) to DD (decimal degrees).
    gps_locs[['lat_dd', 'long_dd']] = pd.DataFrame(np.zeros((len(gps_locs), 2)))
    for i, loc in enumerate(list(zip(gps_locs['Latitude'], gps_locs['Longitude']))):
        gps_locs.loc[i, ['lat_dd', 'long_dd']] = [parse_dms(loc[0]), parse_dms(loc[1])]

    return gps_locs


def add_gps_to_subj_data(subj_data, gps_data):
    # Create columns to hold the GPS coordinates.
    subj_data[['lat', 'long', 'alt', 'GPS_accuracy']] = pd.DataFrame(np.zeros((len(subj_data), 4)))

    for i in range(len(subj_data)):
        # Pick the closest GPS timestamp for each RESpeck timestamp.
        min_idx = int(abs(gps_data.loc[:, 'timestamp'] - subj_data.index[i]).idxmin())
        subj_data.loc[subj_data.index[i], ['lat', 'long', 'alt', 'GPS_accuracy']] = gps_data.loc[
            min_idx, ['latitude', 'longitude', 'altitude', 'accuracy']].values

    return subj_data


def match_gps_to_location_list(subj_data, loc_list_radians):
    # Convert longitudes and latitudes of the subjects' locations to radians:
    radians_data = pd.DataFrame(np.zeros((len(subj_data), 2)), columns=['long', 'lat'])
    radians_data['long'] = np.array([radians(x) for x in subj_data.loc[:, 'long']])
    radians_data['lat'] = np.array([radians(x) for x in subj_data.loc[:, 'lat']])

    # Calculate differences between subject's longs/lats and annotated locations's longs/lats.
    # dlon = lon2 - lon1
    # dlat = lat2 - lat1
    long_dists = np.subtract.outer(radians_data['long'], loc_list_radians['long'])
    lat_dists = np.subtract.outer(radians_data['lat'], loc_list_radians['lat'])

    # Haversine formula step 1: a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2.
    a = np.square(np.sin(lat_dists / 2)) + np.multiply(
        np.outer(np.cos(radians_data['lat']), np.cos(loc_list_radians['lat'])),
        np.square(np.sin(long_dists / 2))
    )

    # Haversine formula step 2: c = 2 * asin(sqrt(a)).
    c = 2 * np.arcsin(np.sqrt(a))

    # Haversine formula step 3: 6371 * c.
    r = 6371000  # Radius of earth in meters.
    distances = c * r

    # Find closest location in the village.
    subj_data['closest_loc'] = np.argmin(distances, axis=1)

    # Set closest location to NaN if the closest location was farther away than the GPS accuracy.
    subj_data.loc[np.min(distances, axis=1) > subj_data['GPS_accuracy'], 'closest_loc'] = np.nan

    return subj_data


def dms2dd(degrees, minutes, seconds, direction):
    """ Convert GPS location from DMS (degrees, minutes, seconds) to DD (decimal degrees) format.
    """

    dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60);
    #     if direction == 'E' or direction == 'N':
    #         dd *= -1
    return dd


def dd2dms(deg):
    """ Convert GPS location from DD (decimal degrees) to DMS (degrees, minutes, seconds) format.
    """

    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]


def parse_dms(dms):
    """ Find latitude from degrees, minutes, seconds.
    """

    parts = re.findall('\d+|\D+', dms)
    lat = dms2dd(parts[1], parts[3], parts[5] + parts[6] + parts[7], parts[0])

    return lat


def read_participant_details(path, project='apcaps', sheet_name=None):
    participant_details = pd.read_excel(path, sheet_name=sheet_name)

    if isinstance(participant_details, dict):
        participant_details = pd.concat(participant_details.values(), ignore_index=True, sort=True)

    # Restyle column names with underscores and non-capital letters.
    participant_details.columns = participant_details.columns.str.rstrip()
    participant_details.columns = participant_details.columns.str.replace('\s+', '_')
    participant_details.columns = map(lambda x: x.lower(), participant_details.columns)

    if project == 'apcaps':
        participant_details.drop(['unnamed:_17', 'slno'], inplace=True, axis=1)

    if project == 'dublin':
        participant_details.dropna(how='all', inplace=True)

        participant_details['subj_id'] = 'DB' + participant_details['subject_identifier_code'].str.strip('DB')
        # participant_details['subject_identifier_code'].str.extract('(\d+)')[0].str.lstrip('0').values.astype(int)

        # Create columns for average/dummy variables.
        cols_to_add = ['F', 'M']
        participant_details[cols_to_add] = pd.DataFrame(
            np.ones((participant_details.shape[0], len(cols_to_add))) * np.nan)

    if project == 'apcaps':
        # Drop participant 12R, was a repeated measurement.
        participant_details = participant_details[participant_details['sample_id:acp'] != '12R']
        participant_details = participant_details[participant_details['sample_id:acp'] != 101]
        participant_details.reset_index(drop=True, inplace=True)

        # Put sex column in all upper case.
        participant_details['sex'] = [str.upper() for str in participant_details['sex']]

        # Create columns for average/dummy variables.
        cols_to_add = ['n_days', 'n_days_recorded', 'n_nights', 'F', 'M', 'av_br', 'av_br_night', 'av_br_day', 'av_al',
                       'av_al_night', 'av_al_day', 'av_steps', 'av_steps_night', 'av_steps_day', 'std_br',
                       'std_br_night', 'std_br_day', 'std_al', 'std_al_night', 'std_al_day', 'fraction_walking',
                       'fraction_walking_day', 'fraction_lying', 'fraction_lying_day', 'av_n_sleep_interruptions',
                       'fraction_sleep_interruptions', 'av_n_turns', 'dummy_napping', 'woke_up']
        participant_details[cols_to_add] = pd.DataFrame(
            np.ones((participant_details.shape[0], len(cols_to_add))) * np.nan)

        participant_details.set_index('sample_id:acp', inplace=True)

        # Change ACPXXX identifiers to only XXX, which is the subject ID.
        participant_details.index.values[~(participant_details.index.str.findall(r'\d+').isna())] = [int(item[0]) for
                                                                                                     item in
                                                                                                     participant_details.index.str.findall(
                                                                                                         r'\d+').dropna()]
        participant_details.sort_index(inplace=True)

    return participant_details


def add_aggregate_variables_to_participant_details(participant_details, data_dict, loc_list, minute_threshold=5,
                                                   turning_al_threshold=0.05, day_times=['08:00', '20:00'],
                                                   night_times=['00:00', '05:00'], deep_night_times=['00:00', '05:00']):
    # Get list of location categories formatted with an underscore.
    loc_categories = [x.replace(' ', '_') for x in list(loc_list['category'].unique())]

    # Create columns for minutes, fraction and dummy of time spent in each type of location.
    for cat in loc_categories:
        participant_details['mins_in_' + cat] = 0
        participant_details['perc_in_' + cat] = 0
        participant_details['dummy_' + cat] = 0

    participant_details['mins_in_alcohol_shop'] = 0
    participant_details['perc_in_alcohol_shop'] = 0
    participant_details['dummy_alcohol_shop'] = 0
    participant_details['mins_in_gym'] = 0
    participant_details['perc_in_gym'] = 0
    participant_details['dummy_gym'] = 0
    participant_details['mins_in_swimming'] = 0
    participant_details['perc_in_swimming'] = 0
    participant_details['dummy_swimming'] = 0
    participant_details['mins_in_playground/sports'] = 0
    participant_details['perc_in_playground/sports'] = 0
    participant_details['dummy_playground/sports'] = 0
    participant_details['mins_sports'] = 0
    participant_details['dummy_sports'] = 0

    loc_variables = ['mins_in_alcohol_shop', 'perc_in_alcohol_shop', 'dummy_alcohol_shop', 'mins_in_health_service',
                     'perc_in_health_service', 'dummy_health_service', 'mins_in_physical_activity_site',
                     'perc_in_physical_activity_site', 'dummy_physical_activity_site']

    for subj_id, subj_data in data_dict.items():

        # Get the number of days in the data.
        n_dates = len(np.unique([d.strftime('%m-%d-%Y') for d in subj_data.index]))
        if n_dates == 0:
            participant_details.loc[subj_id, 'n_days'] = np.nan
            participant_details.loc[subj_id, loc_variables] = np.nan
        else:
            participant_details.loc[subj_id, 'n_days'] = len(
                np.unique([d.strftime('%m-%d-%Y') for d in subj_data.index]))

        # Get the number of days recorded.
        day_data = subj_data.loc[subj_data['day_dummy'] == 1]
        n_dates = len(np.unique([d.strftime('%m-%d-%Y') for d in day_data.index]))
        if n_dates == 0:
            participant_details.loc[subj_id, 'n_days_recorded'] = np.nan
        else:
            participant_details.loc[subj_id, 'n_days_recorded'] = len(
                np.unique([d.strftime('%m-%d-%Y') for d in day_data.index]))

        # Get the number of nights recorded.
        night_data = subj_data.loc[subj_data['night_dummy'] == 1].between_time(deep_night_times[0], deep_night_times[1])
        n_dates = len(np.unique([d.strftime('%m-%d-%Y') for d in night_data.index]))
        if n_dates == 0:
            participant_details.loc[subj_id, 'n_nights'] = np.nan
        else:
            participant_details.loc[subj_id, 'n_nights'] = n_dates

        # Get the number of minutes in the night.
        # n_minutes_night = subj_data['night_dummy'].dropna().sum()
        n_minutes_night = subj_data.loc[subj_data['night_dummy'] == 1, 'data_recorded'].sum()
        n_minutes_day = subj_data.loc[subj_data['day_dummy'] == 1, 'data_recorded'].sum()

        # If no sleeping data was recorded, print message that night values are set to NaN.
        if n_minutes_night < 120:
            print(subj_id, ' Not enough minutes of night data ({:.0f}).'.format(n_minutes_night))
        else:
            print(subj_id, ' Calculated over {:.0f} minutes.'.format(n_minutes_night))

        participant_details = add_averages_and_std_to_participant_details(subj_id, subj_data, participant_details,
                                                                          n_minutes_night, n_minutes_day)
        participant_details = add_laboured_breathing_to_participant_details(subj_id, subj_data, participant_details)
        participant_details = add_movement_info_to_participant_details(subj_id, subj_data, participant_details,
                                                                       n_minutes_night, n_minutes_day)
        participant_details = add_location_information_to_participant_details(subj_id, subj_data, participant_details,
                                                                              loc_categories, loc_list, n_minutes_night,
                                                                              n_minutes_day, loc_variables,
                                                                              minute_threshold)
        participant_details.loc[subj_id, ['av_n_sleep_interruptions', 'fraction_sleep_interruptions', 'av_n_turns',
                                          'woke_up']] = get_sleep_interruptions(subj_data, n_minutes_night,
                                                                                deep_night_times, turning_al_threshold,
                                                                                participant_details.loc[
                                                                                    subj_id, 'n_nights'])

    # Get dummies from sex variable.
    participant_details[['F', 'M']] = pd.get_dummies(participant_details['sex']).values

    # Add BMI information.
    participant_details['bmi'] = np.nan
    not_disabled = (participant_details.index != 42)  # Exclude one disabled subject in the calculations.
    bmi = participant_details.loc[not_disabled, 'weight(kg)'] / np.power(
        participant_details.loc[not_disabled, 'height(cm)'].astype(float) / 100, 2)
    participant_details.loc[not_disabled, 'bmi'] = bmi
    participant_details.loc[42, 'bmi'] = participant_details.loc[
        not_disabled, 'bmi'].mean()  # Set disabled person's BMI to the average of all BMIs.

    return participant_details


def add_averages_and_std_to_participant_details(subj_id, subj_data, participant_details, n_minutes_night,
                                                n_minutes_day):
    # Calculate averages overall.
    participant_details.loc[subj_id, ['av_br', 'av_al', 'av_steps']] = np.array(
        subj_data.mean(axis=0)[['breathing_rate', 'activity_level', 'step_count']])

    # Calculate standard deviations overall.
    participant_details.loc[subj_id, ['std_br', 'std_al']] = np.array(
        subj_data.std(axis=0)[['breathing_rate', 'activity_level']])

    # Calculate day averages and std.
    if n_minutes_day < 120:
        participant_details.loc[
            subj_id, ['av_br_day', 'av_al_day', 'av_steps_day', 'std_br_day', 'std_al_day']] = np.nan
    else:
        participant_details.loc[subj_id, ['av_br_day', 'av_al_day', 'av_steps_day']] = np.array(
            subj_data.loc[subj_data['day_dummy'] == 1].mean(axis=0)[['breathing_rate', 'activity_level', 'step_count']])
        participant_details.loc[subj_id, ['std_br_day', 'std_al_day']] = np.array(
            subj_data.loc[subj_data['day_dummy'] == 1].std(axis=0)[['breathing_rate', 'activity_level']])

    # If no sleeping data was recorded, set average nightly breathing rate to NaN.
    if n_minutes_night < 120:
        participant_details.loc[
            subj_id, ['av_br_night', 'av_al_night', 'av_steps_night', 'std_br_night', 'std_al_night']] = np.nan
    else:
        participant_details.loc[subj_id, ['av_br_night', 'av_al_night', 'av_steps_night']] = np.array(
            subj_data.loc[subj_data['night_dummy'] == 1].mean(axis=0)[
                ['breathing_rate', 'activity_level', 'step_count']])
        participant_details.loc[subj_id, ['std_br_night', 'std_al_night']] = np.array(
            subj_data.loc[subj_data['night_dummy'] == 1].std(axis=0)[['breathing_rate', 'activity_level']])

    return participant_details


def add_laboured_breathing_to_participant_details(subj_id, subj_data, participant_details):
    # Count number of laboured breathing minutes and whether laboured breathing occured at least once or not.
    participant_details.loc[subj_id, 'perc_laboured_breathing'] = sum(subj_data['laboured_breathing'] == True) / \
                                                                  subj_data['breathing_rate'].count()
    participant_details.loc[subj_id, 'boolean_laboured_breathing'] = participant_details.loc[
                                                                         subj_id, 'perc_laboured_breathing'] > 0

    return participant_details


def add_movement_info_to_participant_details(subj_id, subj_data, participant_details, n_minutes_night, n_minutes_day):
    day_data = subj_data.loc[subj_data['day_dummy'] == 1]
    night_data = subj_data.loc[subj_data['night_dummy'] == 1]

    # Calculate average distance travelled per minute using the adapted definition of walking.
    participant_details.loc[subj_id, 'av_walking_distance'] = subj_data.loc[
                                                                  subj_data['activity_type'] == 1, 'movement'].sum() / \
                                                              subj_data['movement'].count()

    # Calculate average distance travelled per minute overall.
    participant_details.loc[subj_id, 'av_overall_distance'] = subj_data['movement'].sum() / subj_data[
        'movement'].count()

    # Calculate fractions for walking and lying down overall.
    participant_details.loc[subj_id, 'fraction_walking'] = (subj_data['activity_type'] == 1).sum() / subj_data[
        'activity_type'].count()
    participant_details.loc[subj_id, 'fraction_lying'] = (subj_data['activity_type'] == 2).sum() / subj_data[
        'activity_type'].count()

    # Calculate day averages and std.
    if n_minutes_day < 120:
        participant_details.loc[subj_id, ['av_walking_distance_day', 'fraction_walking_day', 'fraction_lying_day',
                                          'dummy_napping']] = np.nan
    else:
        participant_details.loc[subj_id, 'av_walking_distance_day'] = day_data.loc[day_data[
                                                                                       'activity_type'] == 1, 'movement'].sum() / \
                                                                      day_data['movement'].count()
        participant_details.loc[subj_id, 'fraction_walking_day'] = (day_data['activity_type'] == 1).sum() / day_data[
            'activity_type'].count()
        participant_details.loc[subj_id, 'fraction_lying_day'] = (day_data['activity_type'] == 2).sum() / day_data[
            'activity_type'].count()

        # Add a dummy for napping during the day.
        # Get start and end indices for stretches of lying down during the day.
        activity_type = np.array(day_data['activity_type'])
        idx_pairs = np.where(np.diff(np.hstack(([False], np.isin(activity_type, [2, 6, 7, 8]), [False]))))[0].reshape(
            -1, 2)

        # If no lying periods were recorded during the day, set the napping dummy to False.
        if idx_pairs.size == 0:
            participant_details.loc[subj_id, ['dummy_napping']] = 0
        else:
            # Else find the longest stretch of lying down.
            longest_seq = idx_pairs[np.diff(idx_pairs, axis=1).argmax(), :]

            # If there was no stretch of lying down longer than 20 minutes, set napping dummy to False.
            if longest_seq[1] - longest_seq[0] < 20:
                participant_details.loc[subj_id, ['dummy_napping']] = 0

            # Else, set napping dummy to True.
            else:
                participant_details.loc[subj_id, ['dummy_napping']] = 1

    return participant_details


def add_location_information_to_participant_details(subj_id, subj_data, participant_details, loc_categories, loc_list,
                                                    n_minutes_night, n_minutes_day, loc_variables, minute_threshold=5):
    # Calculate day averages and std.
    if n_minutes_day < 120:
        participant_details.loc[subj_id, loc_variables] = np.nan
    else:
        participant_details.loc[subj_id, loc_variables] = 0

        # For each minute, count how many consecutive minutes the participant stayed in that location.
        mins_in_same_place = subj_data['closest_loc'].groupby((
                                                                      subj_data['closest_loc'] != subj_data[
                                                                  'closest_loc'].shift()).cumsum()).transform('size')

        # Count the total number of minutes spent in each specific location (counting only stays longer than minute_threshold).
        locs_visited = pd.DataFrame(subj_data.loc[mins_in_same_place > minute_threshold, 'closest_loc'].value_counts())
        locs_visited.reset_index(inplace=True)
        locs_visited.columns = ['loc', 'minutes']

        # Add the category and description of each location that was visited.
        locs_visited['category'] = loc_list['category'][locs_visited['loc']].values
        locs_visited['description'] = loc_list['description'][locs_visited['loc']].values

        # Aggregate the minutes spent in each location category.
        for cat in locs_visited['category']:
            participant_details.loc[
                subj_id, 'mins_in_' + cat.replace(' ', '_')] = locs_visited.groupby('category').sum().loc[
                cat, 'minutes']
            participant_details.loc[
                subj_id, 'perc_in_' + cat.replace(' ', '_')] = participant_details.loc[
                                                                   subj_id, 'mins_in_' + cat.replace(' ', '_')] / len(
                subj_data['closest_loc'])
            participant_details.loc[
                subj_id, 'dummy_' + cat.replace(' ', '_')] = int(
                participant_details.loc[subj_id, 'mins_in_' + cat.replace(' ', '_')] > 0)

        # Aggregate the minutes spent in each location description.
        for descr in locs_visited['description']:
            if descr in ['gym', 'playground/sports', 'swimming', 'alcohol shop']:
                participant_details.loc[
                    subj_id, 'mins_in_' + descr.replace(' ', '_')] = locs_visited.groupby('description').sum().loc[
                    descr, 'minutes']
                participant_details.loc[
                    subj_id, 'perc_in_' + descr.replace(' ', '_')] = participant_details.loc[
                                                                         subj_id, 'mins_in_' + descr.replace(' ',
                                                                                                             '_')] / len(
                    subj_data['closest_loc'])
                participant_details.loc[
                    subj_id, 'dummy_' + descr.replace(' ', '_')] = int(
                    participant_details.loc[subj_id, 'mins_in_' + descr.replace(' ', '_')] > 0)

        participant_details.loc[subj_id, 'mins_sports'] = participant_details.loc[
            subj_id, ['mins_in_gym', 'mins_in_playground/sports', 'mins_in_swimming']].sum()
        participant_details.loc[subj_id, 'dummy_sports'] = int(
            participant_details.loc[subj_id, 'mins_sports'].sum() > 10)

    return participant_details


def get_sleep_interruptions(subj_data, n_minutes_night, night_times, turning_al_threshold, n_nights):
    if n_minutes_night < 120:
        av_n_sleep_interruptions, fraction_sleep_interruptions, av_n_turns, woke_up = [np.nan, np.nan, np.nan, np.nan]
        return av_n_sleep_interruptions, fraction_sleep_interruptions, av_n_turns, woke_up

    night_data = subj_data.loc[subj_data['night_dummy'] == 1]

    # Reindex the night data so that all minutes are present.
    night_data = night_data.reindex(pd.date_range(night_data.index[0], night_data.index[-1], freq='min'),
                                    fill_value=np.nan)

    # Find minutes that are falsely detected as sleep interruptions due to a missing activity type or wrong orientation.
    false_interruption_idx = subj_data.index[
        ((subj_data['activity_type'] == -1) | (subj_data['activity_type'] == 3))].values

    # Set the activity type in these minutes to NaN and then fill them by the closest non-missing activity type.
    # night_data.loc[np.isin(night_data.index.values, false_interruption_idx), 'activity_type'] = np.nan
    # night_data.loc[np.isin(night_data.index.values, false_interruption_idx), 'activity_type'] = night_data['activity_type'].interpolate(method='nearest').loc[np.isin(night_data.index.values, false_interruption_idx)]

    # Set the activity type in these minutes to NaN and then fill them in the night by the closest non-missing activity type.
    subj_data.loc[night_data.index[np.isin(night_data.index.values, false_interruption_idx)], 'activity_type'] = np.nan
    night_data.loc[np.isin(night_data.index.values, false_interruption_idx), 'activity_type'] = \
    subj_data['activity_type'].interpolate(method='nearest').loc[
        night_data.index[np.isin(night_data.index.values, false_interruption_idx)]]

    # Set interpolated sitting/standing/walking to NaN so that they are detected by the below.
    night_data.loc[night_data['activity_type'] != 2, 'activity_type'] = np.nan

    # Return to only the night minutes.
    night_data = night_data.between_time(night_times[0], night_times[1])

    if night_data['night_dummy'].sum() < 60:
        av_n_sleep_interruptions, fraction_sleep_interruptions, av_n_turns, woke_up = [np.nan, np.nan, np.nan, np.nan]
        return av_n_sleep_interruptions, fraction_sleep_interruptions, av_n_turns, woke_up

    # Get activity type during the night as array.
    a = night_data['activity_type'].values

    # Mask NaNs as True.
    m = np.concatenate(([True], np.isnan(a), [True]))

    # Find intervals without NaNs.
    ss = np.flatnonzero(m[1:] != m[:-1]).reshape(-1, 2)

    # Get length of the NaN intervals, i.e. the sleep interruptions.
    duration_sleep_interruptions = ss[:, 0][1:] - ss[:, 1][:-1]
    fraction_sleep_interruptions = np.sum(duration_sleep_interruptions) / len(night_data)

    # Count the number of sleep interruptions and nights.
    n_sleep_interruptions = len(duration_sleep_interruptions)

    # Get the average number of sleep interruptions.
    av_n_sleep_interruptions = n_sleep_interruptions / n_nights

    # Find if someone woke up or not.
    if n_sleep_interruptions > 0:
        woke_up = 1
    else:
        woke_up = 0

    # # Get the number of turns, defined by a threshold on the activity level.
    # # Current minute should be above the threshold, previous one below it.
    # n_turns = ((night_data['activity_level'] > turning_al_threshold) & (night_data['activity_level'].shift(1) < turning_al_threshold)).sum()

    # Get the number of turns.
    n_turns = night_data['dummy_turn'].sum()
    av_n_turns = n_turns / n_nights

    return av_n_sleep_interruptions, fraction_sleep_interruptions, av_n_turns, woke_up


def get_sleeping_data_no_outliers_fill_nan(data_dict, n_lags=30, n_ahead=10, n_smoothing=10, remove_full_only=True,
                                           day_times=['08:00', '20:00'], night_times=['00:00', '05:00']):
    data_dict_outl_rem = copy.deepcopy(data_dict)
    data_dict_outl_rem, all_data_outl_rem = hf.remove_outliers_fill_nan_and_concat_subj_data(data_dict_outl_rem,
                                                                                             full_only=remove_full_only,
                                                                                             day_times=['08:00',
                                                                                                        '20:00'],
                                                                                             night_times=['00:00',
                                                                                                          '05:00'])
    data_dict_outl_rem_sleeping = {}

    for subj_id, subj_data in data_dict_outl_rem.items():

        # Single out sleeping data.
        sleeping_data = subj_data.loc[subj_data['night_dummy'] == 1]

        # Skip subject if no sleeping data is available.
        if len(sleeping_data) == 0:
            continue

        # Reindex the sleeping data so that all minutes are present.
        sleeping_data = sleeping_data.reindex(
            pd.date_range(sleeping_data.index[0], sleeping_data.index[-1], freq='min'), fill_value=np.nan)
        sleeping_data['subj_id'] = subj_id

        # Create lags and ahead.
        subj_data = create_lagged_and_ahead_vars(subj_data, n_lags, n_ahead,
                                                 variables=['breathing_rate', 'activity_level'])
        sleeping_data = create_lagged_and_ahead_vars(sleeping_data, n_lags, n_ahead,
                                                     variables=['breathing_rate', 'activity_level'])

        # Add to sleeping and normal data dictionary.
        data_dict_outl_rem_sleeping[subj_id] = sleeping_data
        data_dict_outl_rem[subj_id] = subj_data

    data_dict_outl_rem_sleeping, all_data_outl_rem_sleeping = hf.remove_outliers_fill_nan_and_concat_subj_data(
        data_dict_outl_rem_sleeping, full_only=remove_full_only)
    data_dict_outl_rem, all_data_outl_rem = hf.remove_outliers_fill_nan_and_concat_subj_data(data_dict_outl_rem,
                                                                                             full_only=remove_full_only)

    return data_dict_outl_rem_sleeping, all_data_outl_rem_sleeping, data_dict_outl_rem, all_data_outl_rem


def get_variables_per_day(participant_details, data_dict, loc_list, deep_night_times=['00:00', '05:00'],
                          turning_al_threshold=0.05, minute_threshold=5):
    # Get list of location categories formatted with an underscore.
    loc_categories = [x.replace(' ', '_') for x in list(loc_list['category'].unique())]

    # Create columns for minutes, fraction and dummy of time spent in each type of location.
    for cat in loc_categories:
        participant_details['mins_in_' + cat] = 0
        participant_details['perc_in_' + cat] = 0
        participant_details['dummy_' + cat] = 0

    vars_per_day = participant_details.copy()
    vars_per_day = vars_per_day.loc[vars_per_day.index.repeat(vars_per_day['n_days'].replace(np.nan, 1).astype(int))]
    vars_per_day['day_id'] = vars_per_day.groupby(level=0).cumcount() + 1

    # Set variables that need to be recalculated to NaN.
    vars_per_day.loc[:,
    ['av_br_night', 'av_br_day', 'av_al_night', 'av_al_day', 'av_steps_night', 'av_steps_day', 'std_br_night',
     'std_br_day', 'std_al_night', 'std_al_day', 'fraction_walking_day', 'fraction_lying_day',
     'av_n_sleep_interruptions', 'fraction_sleep_interruptions', 'av_n_turns', 'av_walking_distance_day',
     'dummy_napping', 'woke_up']] = np.nan
    vars_per_day.loc[:, ['dummy_alcohol_shop', 'mins_in_alcohol_shop', 'perc_in_alcohol_shop']] = np.nan

    loc_variables = ['mins_in_alcohol_shop', 'perc_in_alcohol_shop', 'dummy_alcohol_shop', 'mins_in_health_service',
                     'perc_in_health_service', 'dummy_health_service', 'mins_in_physical_activity_site',
                     'perc_in_physical_activity_site', 'dummy_physical_activity_site']

    # Create indicator for the presence of night and day data for a particular date and subject.
    vars_per_day['night_recording'] = False
    vars_per_day['day_recording'] = False

    for subj_id, subj_data in data_dict.items():

        # Add the date.
        dates = np.unique([d.strftime('%m-%d-%Y') for d in subj_data.index])
        vars_per_day.loc[subj_id, 'date'] = dates

        # Get total day and night data.
        overall_day_data = data_dict[subj_id].loc[data_dict[subj_id]['day_dummy'] == 1]
        overall_night_data = data_dict[subj_id].loc[data_dict[subj_id]['night_dummy'] == 1]

        for date in dates:

            subj_data = data_dict[subj_id][date]
            row_idx = (vars_per_day['date'] == date) & (vars_per_day.index == subj_id)

            # Check if any day data exists for this date.
            if not date in overall_day_data.index:
                n_minutes_day = 0
            else:
                day_data = overall_day_data[date]
                n_minutes_day = sum(day_data['data_recorded'])

            # Check if any night date exists for this date.
            if not date in overall_night_data.index:
                n_minutes_night = 0
            else:
                night_data = overall_night_data[date]
                n_minutes_night = sum(night_data['data_recorded'])

            # If no sleeping data was recorded, print message that night values are set to NaN.
            if n_minutes_night < 120:
                print(subj_id, date, ' Not enough minutes of night data ({:.0f}).'.format(n_minutes_night))
            # Else store that there was a night recording for this date.
            else:
                print(subj_id, date, ' Calculated night variables over {:.0f} minutes.'.format(n_minutes_night))
                vars_per_day.loc[row_idx, 'night_recording'] = True

            # If no day data was recorded, print message that day values are set to NaN.
            if n_minutes_day < 120:
                print(subj_id, date, ' Not enough minutes of day data ({:.0f}).'.format(n_minutes_day))
            # Else store that there was a day recording for this date.
            else:
                print(subj_id, date, ' Calculated day variables over {:.0f} minutes.'.format(n_minutes_day))
                vars_per_day.loc[row_idx, 'day_recording'] = True

            vars_per_day = add_averages_and_std_to_participant_details(row_idx, subj_data, vars_per_day,
                                                                       n_minutes_night, n_minutes_day)
            vars_per_day = add_laboured_breathing_to_participant_details(row_idx, subj_data, vars_per_day)
            vars_per_day = add_movement_info_to_participant_details(row_idx, subj_data, vars_per_day, n_minutes_night,
                                                                    n_minutes_day)
            vars_per_day = add_location_information_to_participant_details(row_idx, subj_data, vars_per_day,
                                                                           loc_categories, loc_list, n_minutes_night,
                                                                           n_minutes_day, loc_variables,
                                                                           minute_threshold=minute_threshold)
            vars_per_day.loc[row_idx, ['av_n_sleep_interruptions', 'fraction_sleep_interruptions', 'av_n_turns',
                                       'woke_up']] = get_sleep_interruptions(subj_data, n_minutes_night,
                                                                             night_times=deep_night_times,
                                                                             turning_al_threshold=turning_al_threshold,
                                                                             n_nights=1)

    return vars_per_day
