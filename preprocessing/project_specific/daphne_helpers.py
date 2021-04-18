import numpy as np
import pandas as pd
from pytz import timezone

from constants import project_mapping, daphne_logs_filepath, \
    daphne_questionnaire_database_aap_filepath, daphne_questionnaire_database_mc_filepath
from download_data import download_static_airspeck, \
    download_respeck_data, download_personal_airspeck_data
from pixelgrams import download_data_and_plot_combined_pixelgram


def load_daphne_subject_details():
    logs_ap = pd.read_excel(daphne_logs_filepath, sheet_name="Asthma Cohort WP1.1", engine = 'openpyxl')
    logs_mc = pd.read_excel(daphne_logs_filepath, sheet_name="MC Cohort", engine = 'openpyxl')
    logs = pd.concat([logs_ap, logs_mc], sort=True)
    # Set Subject ID as index but keep column as well
    logs = logs.set_index('Subject ID')
    logs['Subject ID'] = logs.index

    # Only keep non-empty rows
    logs = logs[~pd.isnull(logs.index)]
    return logs


def generate_new_daphne_pixelgrams(overwrite_data_if_already_exists=False, overwrite_pixelgram_if_already_exists=False):
    logs = load_daphne_subject_details()
    logs.loc[:, 'Visit number'] = logs['Visit number'].fillna(1).astype(int)

    for idx, row in logs.iterrows():
        subject_id = row['Subject ID']
        visit_number = int(row['Visit number'])

        if pd.isnull(subject_id):
            continue

        print("Subject {}".format(subject_id))

        tz = timezone(project_mapping['daphne'][1])

        from_time = row['From time personal sensors']
        start_date = tz.localize(row['From date all sensors'].replace(hour=from_time.hour, minute=from_time.minute,
                                                                      second=from_time.second).to_pydatetime())

        to_time = row['To time personal sensors']
        end_date = tz.localize(row['To date all sensors'].replace(hour=to_time.hour, minute=to_time.minute,
                                                                  second=to_time.second).to_pydatetime())

        download_data_and_plot_combined_pixelgram(subject_id=subject_id, timeframe=[start_date, end_date],
                                                  overwrite_pixelgram_if_already_exists=overwrite_pixelgram_if_already_exists,
                                                  subject_visit_number=visit_number,
                                                  overwrite_data_if_already_exists=overwrite_data_if_already_exists,
                                                  upload_type='manual')


def load_daphne_questionnaire_database(sheetname, cohort='aap'):
    if cohort == 'aap':
        filepath = (daphne_questionnaire_database_aap_filepath)
    elif cohort == 'mc':
        filepath = (daphne_questionnaire_database_mc_filepath)
    else:
        print("Cohort must be either 'aap' or 'mc'")
        return
    return pd.read_excel(filepath, sheetname=sheetname)

def daphne_get_recording_timeframe(subj_id, visit_number):

    daphne_logs = load_daphne_subject_details()    
    row = daphne_logs.loc[subj_id]
    if row.ndim > 1:
        row = row.loc[row ['Visit number'] ==  visit_number].squeeze()

    if row ['Visit number'] !=  visit_number:
        print('No visit number {} for subject {}'.format(visit_number, subj_id))
        return

    tz = timezone('Asia/Kolkata')

    from_time = row['From time personal sensors']
    start_date = tz.localize(row['From date all sensors'].replace(hour=from_time.hour, minute=from_time.minute,
                                                                  second=from_time.second))#.to_pydatetime())

    to_time = row['To time personal sensors']
    end_date = tz.localize(row['To date all sensors'].replace(hour=to_time.hour, minute=to_time.minute,
                                                          second=to_time.second))#.to_pydatetime())

    return [start_date, end_date]


def remove_daphne_gps_outliers(airspeck_data):
    airspeck_data.loc[airspeck_data['gpsAccuracy'] >= 1000, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_data.loc[airspeck_data['gpsLatitude'] < 28.4, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_data.loc[airspeck_data['gpsLatitude'] > 28.9, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_data.loc[airspeck_data['gpsLongitude'] < 76.8, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_data.loc[airspeck_data['gpsLongitude'] > 77.6, 'gpsLongitude':'gpsLatitude'] = np.nan


def download_all_daphne_data(overwrite_if_already_exists = False):
    # Load Excel spreadsheet
    logs = load_daphne_subject_details()

    # Until all cells are filled in, assume that the start and end times are the same for personal and static sensors
    project_name = 'daphne'

    for idx, row in logs.iterrows():
        subject_id = row['Subject ID']
        visit_number = int(row['Visit number'])
        subject_label = "{}({})".format(subject_id, visit_number)
        print("Downloading data for {}".format(subject_label))

        # If there's an error with converting the following ('replace takes no arguments' or others), check that the
        # Excel dates are indeed formatted as dates. Sometimes dates look like dates in Excel but are just text.
        # Convert with "datevalue(text)"
        from_time = row['From time personal sensors']
        start_date = row['From date all sensors'].replace(hour=from_time.hour, minute=from_time.minute,
                                                          second=from_time.second)#.to_pydatetime()
        to_time = row['To time personal sensors']
        end_date = row['To date all sensors'].replace(hour=to_time.hour, minute=to_time.minute,
                                                      second=to_time.second)#.to_pydatetime()
        timeframe = [start_date, end_date]

        personal_upload = row["Personal upload"]
        # Download personal data if not yet present
        #download_respeck_and_personal_airspeck_data(subject_id, upload_type='manual', timeframe=timeframe,
        #                                            subject_visit_number=visit_number,
        #                                            overwrite_if_already_exists=overwrite_if_already_exists,
        #                                           is_minute_averaged=True)
        
        download_respeck_data(subject_id, upload_type="manual", is_minute_averaged=True, timeframe=timeframe,
                          overwrite_if_already_exists=overwrite_if_already_exists, subject_visit_number=visit_number, project_name="daphne")
        
        download_personal_airspeck_data(subject_id, upload_type=personal_upload, is_minute_averaged=True, timeframe=timeframe,
                                    overwrite_if_already_exists=overwrite_if_already_exists, subject_visit_number=visit_number, project_name="daphne")
    

        # Download static data in home
        if not pd.isnull(row['Static sensor inside home ID']):
            # In some cases, a personal monitor was used as stationary Airspeck
            if len(row['Static sensor inside home ID']) == 6:
                # We download the data from the personal sensor into a file which looks just like the other static
                # sensors so that it's easier to work with it.
                download_personal_airspeck_data(row['Static sensor inside home ID'], upload_type='manual',
                                                timeframe=timeframe,
                                                overwrite_if_already_exists=overwrite_if_already_exists,
                                                filename="{}_static_airspeck_automatic_home.csv".format(subject_label))
            else:
                home_upload = row["Home upload"]
                download_static_airspeck(row['Static sensor inside home ID'], sensor_label=subject_label,
                                         project_name=project_name,
                                         upload_type=home_upload, suffix_filename='_home',
                                         timeframe=[start_date, end_date],
                                         overwrite_if_already_exists=overwrite_if_already_exists)

        # Download community static sensor
        print("Downloading community sensor data")
        if not pd.isnull(row['Nearest community static sensor ID']):
            download_static_airspeck(row['Nearest community static sensor ID'], sensor_label=subject_label,
                                     project_name=project_name,
                                     upload_type='automatic', suffix_filename='_community',
                                     timeframe=[start_date, end_date],
                                     overwrite_if_already_exists=overwrite_if_already_exists)

        # Download school static sensor
        print("Downloading school sensor data")

        if 'School static sensor ID' in row and not pd.isnull(row['School static sensor ID']):
            download_static_airspeck(row['School static sensor ID'], sensor_label=subject_label,
                                     project_name=project_name,
                                     upload_type='automatic', suffix_filename='_school',
                                     timeframe=[start_date, end_date],
                                     overwrite_if_already_exists=overwrite_if_already_exists)

    print("Done!")
