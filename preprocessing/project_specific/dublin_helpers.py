# Set these to suitable local directories
from datetime import timedelta

import numpy as np
import pandas as pd
from pytz import timezone

# import meeke.helper_functions as hf
import project_specific.meeke_data_prep_functions_dublin as dpf
from constants import project_mapping, lying_activities, dublin_participant_details_dir, dublin_data_dir, \
    dublin_timezones_correction_filepath
from download_data import download_respeck_and_personal_airspeck_data
from load_files import load_respeck_file, load_personal_airspeck_file
from misc_utils import load_data_pickle
from pixelgrams import plot_combined_pixelgram_dublin

subjects_with_high_shs = ["DBCA01", "DBCA02", "DBCA04", "DBCA05", "DBCA06", "DBCA07", "DBCA11", "DBCA15", "DBCC01",
                          "DBCC03", "DBCC04", "DBCC06", "DBCC08", "DBCC11", "DBCC12", "DBCC15", "DBCC17", "DBCC18",
                          "DBCC20", "DBIA04", "DBIA04", "DBIA10", "DBSA03", "DBSA04", "DBSC02", "DBSC07", "DBSC10"]


def load_dublin_participant_details():
    participant_details_spain = dpf.read_participant_details(
        dublin_participant_details_dir + "/Data collection Worksheet Spain.xlsx",
        project='dublin', sheet_name='Spanish Data ')
    participant_details_czech = dpf.read_participant_details(
        dublin_participant_details_dir + "/Data collection Worksheet Czech.xlsx",
        project='dublin', sheet_name='Czech Rep Data ')
    participant_details_ireland = dpf.read_participant_details(
        dublin_participant_details_dir + "/Data collection Worksheet Ireland.xlsx",
        project='dublin', sheet_name='Irish Data ')

    participant_details = pd.concat([participant_details_spain, participant_details_czech,
                                     participant_details_ireland], sort=True)
    participant_details.set_index('subj_id', drop=False, inplace=True)
    # Drop participant DBCA03, as the Airspeck cables got loose, and the recording was repeated with ID DBCA11
    participant_details = participant_details.sort_index().drop('DBCA03')
    participant_details = participant_details.drop('DBCA08')
    participant_details = participant_details.drop('DBCA09')
    participant_details = participant_details.drop('DBCC05')
    participant_details = participant_details[~pd.isnull(participant_details.index)]

    participant_details = participant_details.replace('-', np.nan).replace('NaN', np.nan)

    # Correct PFR unit in Spanish subjects
    spanish_mask = [sub[2] == 'S' for sub in participant_details.index]
    participant_details.loc[spanish_mask, ['_baseline_pefr', '_post_exposure_pefr']] = \
        participant_details.loc[spanish_mask, ['_baseline_pefr', '_post_exposure_pefr']].astype(float) * 100.

    return participant_details


def create_dublin_pixelgram_for_subject(subject_id, overwrite_pixelgram_if_already_exists=False):
    download_respeck_and_personal_airspeck_data(subject_id, upload_type='manual')
    respeck_data = load_respeck_file(subject_id, upload_type='manual')
    airspeck_data = load_personal_airspeck_file(subject_id, upload_type='manual')

    # Load correction factors for timezone
    corrections = pd.read_excel(dublin_timezones_correction_filepath).replace(np.nan, 0).set_index('subject_id')

    participant_details = load_dublin_participant_details()
    row = participant_details.loc[subject_id]

    # Load exposure period
    from_time = row['start_of_exposure_time_to_shs']
    to_time = row['end_of_exposure_time_to_shs']
    start_exposure = row['date_of_exposure_to_shs'].replace(
        hour=from_time.hour, minute=from_time.minute, second=from_time.second).to_pydatetime() + timedelta(
        hours=int(corrections.loc[subject_id, 'shs_times_difference']))
    if not pd.isnull(row['end_date_of_exposure_to_shs']):
        end_exposure = row['end_date_of_exposure_to_shs'].replace(
            hour=to_time.hour, minute=to_time.minute, second=to_time.second).to_pydatetime() + timedelta(
            hours=int(corrections.loc[subject_id, 'shs_times_difference']))
    else:
        end_exposure = row['date_of_exposure_to_shs'].replace(
            hour=to_time.hour, minute=to_time.minute, second=to_time.second).to_pydatetime() + timedelta(
            hours=int(corrections.loc[subject_id, 'shs_times_difference']))

    # Load recording period
    from_time = row['start_time_of_monitoring']
    start_recording = row['start_date_of_monitoring'].replace(
        hour=from_time.hour, minute=from_time.minute, second=from_time.second).to_pydatetime() + timedelta(
        hours=int(corrections.loc[subject_id, 'recording_times_difference']))

    to_time = row['end_time_of_monitoring']
    end_recording = row['end_date_of_monitoring'].replace(
        hour=to_time.hour, minute=to_time.minute, second=to_time.second).to_pydatetime() + timedelta(
        hours=int(corrections.loc[subject_id, 'recording_times_difference']))

    # Look up timezone
    tz = timezone(project_mapping[subject_id[:3]][1])

    print("Creating pixelgram for subject {}".format(subject_id))

    plot_combined_pixelgram_dublin(subject_id, respeck_data, airspeck_data,
                                   exposure_period=[tz.localize(start_exposure), tz.localize(end_exposure)],
                                   recording_period=[tz.localize(start_recording), tz.localize(end_recording)],
                                   overwrite_if_already_exists=overwrite_pixelgram_if_already_exists)


def download_all_dublin_data(force_download=False):
    participant_details = load_dublin_participant_details()

    for idx, row in participant_details.iterrows():
        subj_id = row.name

        if not pd.isnull(subj_id):
            tz = timezone(project_mapping[subj_id[:3]][1])

            from_time = row['start_time_of_monitoring']
            start_date = tz.localize(row['start_date_of_monitoring'].replace(
                hour=from_time.hour, minute=from_time.minute, second=from_time.second).to_pydatetime())
            to_time = row['end_time_of_monitoring']
            end_date = tz.localize(row['end_date_of_monitoring'].replace(hour=to_time.hour, minute=to_time.minute,
                                                                         second=to_time.second).to_pydatetime())

            # We don't use this timeframe as the logged timestamps aren't always correct.
            timeframe = [start_date, end_date]

            # Set timeframe to None for now, as we want all data.
            download_respeck_and_personal_airspeck_data(subj_id, upload_type='manual',
                                                        overwrite_if_already_exists=force_download,
                                                        timeframe=None)


def get_localised_smoking_period(row, corrected=True):
    subj_id = row.name

    from_time = row['start_of_exposure_time_to_shs']
    to_time = row['end_of_exposure_time_to_shs']
    start_exposure = row['date_of_exposure_to_shs'].replace(hour=from_time.hour, minute=from_time.minute,
                                                            second=from_time.second).to_pydatetime()
    if not pd.isnull(row['end_date_of_exposure_to_shs']):
        end_exposure = row['end_date_of_exposure_to_shs'].replace(hour=to_time.hour, minute=to_time.minute,
                                                                  second=to_time.second).to_pydatetime()
    else:
        end_exposure = row['date_of_exposure_to_shs'].replace(hour=to_time.hour, minute=to_time.minute,
                                                              second=to_time.second).to_pydatetime()

    tz = timezone(project_mapping[subj_id[:3]][1])

    shs_period = [tz.localize(start_exposure), tz.localize(end_exposure)]

    if corrected:
        corrections = pd.read_excel(dublin_timezones_correction_filepath).replace(np.nan, 0).set_index('subject_id')
        shs_period = [shs_period[0] + timedelta(hours=int(corrections.loc[subj_id, 'shs_times_difference'])),
                      shs_period[1] + timedelta(hours=int(corrections.loc[subj_id, 'shs_times_difference']))]

    return shs_period


def get_static_during_day(data):
    '''
    # Don't look at night: between 23-6:00 or lying
    data['is_day'] = ~(data['activity_type'].isin(lying_activities) | ((data.index.hour >= 0) & (data.index.hour <= 6)))

    # Only look at periods where the subject was mostly static in the past minute
    is_static = data['activity_type'].isin(static_activities)
    for shift_i in range(1, 2):
        is_static = is_static & data['activity_type'].shift(shift_i).isin(
            static_activities)
    data['is_static'] = is_static

    valid_data = data.loc[data['is_static'] & data['is_day']]
    '''
    # Don't look at night: between 23-6:00 or lying
    data['is_day'] = ~(data['activity_type'].isin(lying_activities) | ((data.index.hour >= 0) & (data.index.hour <= 6)))
    # Only look at periods where the subject was mostly static in the past minute
    data['is_static'] = data['activity_level'] <= 0.08

    valid_data = data.loc[data['is_static'] & data['is_day']]
    return valid_data


def at_least_15_minutes_of_more_than_x(airspeck_data, x=10):
    # Was there at least 15 minutes of > 10 PM2.5?
    count = 0
    is_high = False

    if 'pm2_5' in airspeck_data:
        pm_data = airspeck_data['pm2_5']
    else:
        # A series was passed
        pm_data = airspeck_data

    for pm in pm_data:
        if pm > x:
            count += 1
            if count == 15:
                is_high = True
                break
        else:
            count = 0
    return is_high


def load_dublin_data_pickle():
    return load_data_pickle(dublin_data_dir + "all_data.cpickle")
