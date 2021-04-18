import pandas as pd

from constants import philap_participant_details_filepath
from download_data import download_personal_airspeck_data, download_respeck_data


def load_philap_participant_details():
    details = pd.read_excel(philap_participant_details_filepath)
    # Only keep rows where the subject ID is six characters long (discard retaken recordings)
    details = details.loc[details['Subject ID'].str.len() == 6]

    # Make subject IDs index, but also keep as column
    details = details.set_index('Subject ID')
    details['Subject ID'] = details.index

    return details


def download_all_philap_data(overwrite_if_already_exists=False, raw_airspeck=False):
    # Load Excel spreadsheet
    logs = load_philap_participant_details()
    
    for idx, row in logs.iterrows():
        subject_id = row['Subject ID']
        visit_number = row['Visit number']
        
        if visit_number == 0: 
            continue

        if len(subject_id) == 6:
            print("Downloading data for {}, visit {}".format(subject_id, visit_number))

            # If there's an error with converting the following ('replace takes no arguments' or others), check that the
            # Excel dates are indeed formatted as dates. Sometimes dates look like dates in Excel but are just text.
            # Convert with "datevalue(text)"
            start_date = row['From date all sensors']
            end_date = row['To date all sensors'].replace(hour=23, minute=59, second=59)#.to_pydatetime()
            timeframe = [start_date, end_date]

            # Download personal data if not yet present
            download_respeck_data(subject_id, upload_type='manual', timeframe=timeframe,
                                  is_minute_averaged=True, overwrite_if_already_exists=overwrite_if_already_exists, subject_visit_number=visit_number)
            download_personal_airspeck_data(subject_id, upload_type='manual', timeframe=timeframe,
                                            is_minute_averaged=not raw_airspeck,
                                            overwrite_if_already_exists=overwrite_if_already_exists, subject_visit_number=visit_number)
