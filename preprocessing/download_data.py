import os
from datetime import datetime

import numpy as np
import pandas as pd
#from google.cloud import storage
from pytz import timezone

from constants import project_mapping
from decrypt_file import decrypt_file
from misc_utils import partly_decrypt_file, get_datastore_client, get_project_for_subject

'''
The below functions set most parameters to default values, such as getting the project name from the subject ID, 
and determining the data directory and filename from that. If you want more control over what you download, use
the functions further below in this file
'''


def download_personal_airspeck_data(subject_id, upload_type='automatic', is_minute_averaged=True, timeframe=None,
                                    overwrite_if_already_exists=False, subject_visit_number=None,
                                    suffix_filename="", filename=None, project_name=None, out_directory=None):
    assert upload_type in ['automatic', 'manual',
                           'sd_card'], "Upload type has to be either automatic, manual or sd_card"

    assert is_minute_averaged or upload_type is not 'automatic', \
        "Only minute averaged data is automatically uploaded. If manual, set is_minute_average=False. If automatic, set is_minute_average=True"

    if project_name is None:
        project_name = get_project_for_subject(subject_id)

    if out_directory is None:
        out_directory = project_mapping[project_name][2]

    if subject_visit_number is None:
        label_files = "{}".format(subject_id)
    else:
        label_files = "{}({})".format(subject_id, int(subject_visit_number))

    if filename is None:
        if is_minute_averaged:
            filename = "{}_airspeck_personal_{}{}.csv".format(label_files, upload_type, suffix_filename)
        else:
            filename = "{}_airspeck_personal_{}_raw{}.csv".format(label_files, upload_type, suffix_filename)

    if os.path.isfile(out_directory + filename) and not overwrite_if_already_exists:
        print("Data already downloaded")
        return

    if timeframe is None:
        # Set to a timeframe which will definitely include all data
        timeframe = [datetime(2016, 1, 1), datetime(2100, 1, 1)]

    if upload_type == 'automatic':
        download_airspeck_from_datastore(subject_id, out_filepath=out_directory + filename,
                                         project_name=project_name, timeframe=timeframe,
                                         upload_type='automatic')
    elif upload_type == 'manual':
        download_airspeck_from_google_storage(subject_id, out_directory=out_directory, out_filename=filename,
                                              timeframe=timeframe, project_name=project_name,
                                              overwrite_file_if_existing=overwrite_if_already_exists,
                                              store_raw=not is_minute_averaged)
    elif upload_type == 'sd_card':
        download_airspeck_from_datastore(subject_id, out_filepath=out_directory + filename,
                                         project_name=project_name, timeframe=timeframe,
                                         upload_type='sd_card')
    print('Done')


def download_respeck_data(subject_id, upload_type='automatic', is_minute_averaged=True, timeframe=None,
                          overwrite_if_already_exists=False, subject_visit_number=None, suffix_filename="",
                          filename=None, project_name=None, out_directory=None):
    assert upload_type in ['automatic', 'manual'], "Upload type has to be either automatic or manual"

    assert is_minute_averaged or upload_type is not 'automatic', \
        "Only minute averaged data is automatically uploaded. Set is_minute_average=False."

    if project_name is None:
        project_name = get_project_for_subject(subject_id)

    if out_directory is None:
        out_directory = project_mapping[project_name][2]

    if subject_visit_number is None:
        label_files = "{}".format(subject_id)
    else:
        label_files = "{}({})".format(subject_id, int(subject_visit_number))

    if filename is None:
        if is_minute_averaged:
            filename = "{}_respeck_{}{}.csv".format(label_files, upload_type, suffix_filename)
        else:
            filename = "{}_respeck_{}_raw{}.csv".format(label_files, upload_type, suffix_filename)

    if timeframe is None:
        # Set to a timeframe which will definitely include all data
        timeframe = [datetime(2016, 1, 1), datetime(2100, 1, 1)]

    if os.path.isfile(out_directory + "/" + filename) and not overwrite_if_already_exists:
        print("Data already downloaded")
        return

    if upload_type == 'automatic':
        download_respeck_minute_from_datastore(subject_id, out_filepath=out_directory + filename, timeframe=timeframe,
                                               project_name=project_name, upload_type='automatic')
    elif upload_type == 'manual':
        if is_minute_averaged:
            download_respeck_minute_from_datastore(subject_id, out_filepath=out_directory + filename,
                                                   timeframe=timeframe, project_name=project_name, upload_type='manual')
        else:
            download_raw_respeck_from_google_storage(subject_id, out_directory=out_directory, out_filename=filename,
                                                     timeframe=timeframe, project_name=project_name,
                                                     overwrite_file_if_existing=overwrite_if_already_exists,
                                                     subject_visit_number=subject_visit_number)
    print('Done')


def download_respeck_and_personal_airspeck_data(subject_id, upload_type='automatic', is_minute_averaged=True,
                                                timeframe=None, overwrite_if_already_exists=False,
                                                subject_visit_number=None, suffix_filename="", project_name=None):
    download_respeck_data(subject_id, upload_type, is_minute_averaged, timeframe,
                          overwrite_if_already_exists, subject_visit_number, suffix_filename, project_name=project_name)
    download_personal_airspeck_data(subject_id, upload_type, is_minute_averaged, timeframe,
                                    overwrite_if_already_exists, subject_visit_number, suffix_filename, project_name=project_name)
    print('Done')


def download_static_airspeck(subj_or_uuid, sensor_label=None, project_name=None, overwrite_if_already_exists=False,
                             timeframe=None, upload_type='automatic', suffix_filename="", filename=None,
                             subject_visit_number=None, out_directory=None):
    assert upload_type in ['automatic', 'sd_card'], "upload_type has to be either 'automatic' or 'sd_card'"

    if project_name is None:
        if len(subj_or_uuid) == 6:
            project_name = get_project_for_subject(subj_or_uuid)
        else:
            raise ValueError("When passing a UUID and not a subject ID, also specify a project_name so that the "
                             "correct directory can be selected")

    if out_directory is None:
        out_directory = project_mapping[project_name][2]

    if sensor_label is None:
        if len(subj_or_uuid) == 6 and subject_visit_number is not None:
            sensor_label = "{}({})".format(subj_or_uuid, subject_visit_number)
        else:
            sensor_label = subj_or_uuid

    if filename is None:
        filename = "{}_static_airspeck_{}{}.csv".format(sensor_label, upload_type, suffix_filename)

    out_filepath = out_directory + filename

    if not overwrite_if_already_exists and os.path.isfile(out_filepath):
        print('Skipping file as it already exists')
        return

    client = get_datastore_client()

    with open(out_filepath, "w") as out:

        out.write("timestamp,pm1,pm2_5,pm10,bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9,bin10,bin11,bin12,"
                  "bin13,bin14,bin15,temperature,humidity,battery,no2_ae,no2_we,ox_ae,ox_we,"
                  "gpsLatitude,gpsLongitude\n")

        # Did user pass timeframe? If not, load all data
        if timeframe is None:
            timeframe = [datetime(2016, 1, 1), datetime(2100, 1, 1)]

        tz = timezone(project_mapping[project_name][1])

        if timeframe[0].tzinfo is None:
            utc_start = tz.localize(timeframe[0]).astimezone(timezone('UTC')).replace(
                tzinfo=None)
            utc_end = tz.localize(timeframe[1]).astimezone(timezone('UTC')).replace(
                tzinfo=None)
        else:
            utc_start = timeframe[0]
            utc_end = timeframe[1]

        if upload_type == 'automatic':
            kind_name = 'StaticAirspeck'
            if len(subj_or_uuid) == 16:
                id_name = 'uuid'
            else:
                id_name = 'subject_id'
        else:
            kind_name = 'StaticAirspeckSDCard'
            if len(subj_or_uuid) == 16:
                id_name = 'airspeck_uuid'
            else:
                id_name = 'subject_id'

        query = client.query(kind=kind_name,
                             filters=[(id_name, '=', subj_or_uuid), ('timestamp', '>=', utc_start),
                                      ('timestamp', '<', utc_end)], order=['timestamp']).fetch()

        for e in query:
            out.write("{},{},{},{},".format(e['timestamp'].replace(tzinfo=None), e['pm1'], e['pm2_5'], e['pm10']))
            for i in range(0, 16):
                out.write("{},".format(e['bins'][i]))
            if upload_type == 'automatic':
                out.write("{},{},{},{},{},{},{},{},{}\n".format(e['temperature'], e['humidity'], e['battery'],
                                                                e['no2_ae'], e['no2_we'], e['ox_ae'], e['ox_we'],
                                                                e['location']['latitude'], e['location']['longitude']))
            else:
                out.write("{},{},{},{},{},{},{},{},{}\n".format(e['temperature'], e['humidity'], e['battery'],
                                                                e['no2_ae'], e['no2_we'], e['ox_ae'], e['ox_we'],
                                                                e['latitude'], e['longitude']))

    print('Done')


'''
Everything below doesn't need to be called directly in most cases.
'''


def download_from_google_storage(subject_id, prefix_storage_filename, timestamp_label, out_filename, out_directory=None,
                                 project_name=None, timeframe=None, force_download=False, store_raw=False):
    if project_name is None:
        project_name = get_project_for_subject(subject_id)

    if out_directory is None:
        out_directory = project_mapping[project_name][2]

    if os.path.isfile(out_directory + out_filename) and not force_download:
        print("Data already downloaded")
        return

    # Did user pass timeframe? If not, load all data
    if timeframe is None:
        timeframe = [datetime(2016, 1, 1), datetime(2100, 1, 1)]

    # Select the timeframe, after accounting for timezone difference
    tz = timezone(project_mapping[project_name][1])

    if timeframe[0].tzinfo is None:
        localised_start = tz.localize(timeframe[0])
        localised_end = tz.localize(timeframe[1])
    else:
        localised_start = timeframe[0]
        localised_end = timeframe[1]

    data = pd.DataFrame()
    storage_client = storage.Client('specknet-pyramid-test')
    bucket = storage_client.get_bucket(project_mapping[project_name][0])

    for blob in bucket.list_blobs(prefix='AirRespeck/{}'.format(subject_id)):
        filename = blob.name.split("/")[-1]
        if subject_id in filename and prefix_storage_filename in filename:
            if timeframe is not None:
                date_of_file = tz.localize(datetime.strptime(filename[-14:-4], "%Y-%m-%d"))
                # Skip file if it's not in the timeframe we're interested in!
                if date_of_file < localised_start.replace(hour=0, minute=0, second=0) or \
                        date_of_file > localised_end:
                    continue

            temp_file = out_directory + "temp/" + filename

            # Create temp directory if it doesn't exist yet
            if not os.path.exists(out_directory + "temp"):
                os.makedirs(out_directory + "temp")

            if not os.path.isfile(temp_file):
                blob.download_to_filename(temp_file)

            # If data is encrypted, overwrite with decrypted version
            with open(temp_file) as file:
                if file.readline().strip() == "Encrypted":
                    # Decrypt file before continuing
                    print("File is being decrypted")
                    decrypt_file(temp_file, temp_file)
                else:
                    # Try converting all dates. If this failes, some lines are probably encrypted
                    temp_data = pd.read_csv(temp_file, error_bad_lines=False)
                    try:
                        pd.to_datetime(temp_data[timestamp_label], unit='ms', exact=False)
                    except:
                        partly_decrypt_file(temp_file, temp_file)

            data = data.append(pd.read_csv(temp_file, error_bad_lines=False))

    if len(data) > 0:
        data[timestamp_label] = pd.to_datetime(data[timestamp_label], unit='ms', exact=False)

        # Calculate minute averages
        if prefix_storage_filename in ["Airspeck", "GPSPhone"] and not store_raw:
            data = data.groupby(data[timestamp_label].apply(lambda d: d.replace(second=0, microsecond=0))).mean()
        else:
            # Don't minute average here, but simply set the timestamp column as index
            data = data.set_index(data[timestamp_label]).sort_index()
            # Delete the original column
            data = data.drop(timestamp_label, axis=1)

        # Re-insert (copy) timestamp column from index, so that it is saved
        if 'timestamp' not in data.columns:
            data.insert(0, 'timestamp', data.index)

        # Remove NaTs from index
        data = data.loc[data.index.notnull()]

        # If we are downloading Respeck data, remove Respeck timestamps and sequence number
        if prefix_storage_filename == "RESpeck":
            data = data.drop(['respeckTimestamp', 'sequenceNumber'], axis=1)

        data = data[localised_start.astimezone(timezone('UTC')).replace(tzinfo=None):localised_end.astimezone(
            timezone('UTC')).replace(tzinfo=None)]

        data.to_csv(out_directory + "/" + out_filename, index=False)


def download_logs_from_google_storage(subject_id, force_download=False):
    out_directory = project_mapping[subject_id[:2]][2]
    out_filename = '{}_logs.csv'.format(subject_id)
    prefix_storage_filename = "Logs"

    if os.path.isfile(out_directory + "/" + out_filename) and not force_download:
        print("Data already downloaded")
        return

    storage_client = storage.Client('specknet-pyramid-test')
    bucket = storage_client.get_bucket(project_mapping[subject_id[:2]][0])

    out = []
    for blob in bucket.list_blobs(prefix='AirRespeck'):
        filename = blob.name.split("/")[-1]
        if subject_id in filename and prefix_storage_filename in filename:
            temp_file = out_directory + "/temp/" + filename

            # Create temp directory if it doesn't exist yet
            if not os.path.exists(out_directory + "/temp"):
                os.makedirs(out_directory + "/temp")

            if not os.path.isfile(temp_file):
                blob.download_to_filename(temp_file)

            with open(temp_file, 'r+') as f:
                lines = [line.rstrip().split(': ', 1) for line in f.readlines()]
                out.extend(lines[1:])
    np.savetxt(out_directory + "/" + out_filename, out, fmt="%s", delimiter=",")


def download_airspeck_from_google_storage(subject_id, out_directory, out_filename, timeframe,
                                          project_name, overwrite_file_if_existing=False, store_raw=False):
    print("Download personal Airspeck data from Google storage")
    download_from_google_storage(subject_id, "Airspeck", "phoneTimestamp", out_filename, out_directory,
                                 project_name, timeframe, overwrite_file_if_existing, store_raw=store_raw)

def download_raw_respeck_from_google_storage(subject_id, out_directory=None, out_filename=None, timeframe=None,
                                             project_name=None, overwrite_file_if_existing=False,
                                             subject_visit_number=None):
    if subject_visit_number is None:
        label_files = "{}".format(subject_id)
    else:
        label_files = "{}({})".format(subject_id, subject_visit_number)

    if out_filename is None:
        out_filename = "{}_respeck_raw.csv".format(label_files)

    print("Downloading raw files. This will take up a lot of space. (~100MB per day-file)")
    download_from_google_storage(subject_id, "RESpeck", "interpolatedPhoneTimestamp", out_filename, out_directory,
                                 project_name, timeframe, overwrite_file_if_existing)


def download_gps_from_google_storage(subject_id, out_directory, out_filename, timeframe, project_name,
                                     overwrite_file_if_existing=False):
    download_from_google_storage(subject_id, "GPSPhone", "timestamp", out_filename, out_directory, project_name,
                                 timeframe, overwrite_file_if_existing)


def download_respeck_minute_from_datastore(subject_id, out_filepath, timeframe, project_name, upload_type='automatic'):
    assert upload_type in ['automatic', 'manual'], "Data type has to be either automatic or manual"

    tz = timezone(project_mapping[project_name][1])

    if timeframe[0].tzinfo is None:
        localised_start = tz.localize(timeframe[0]).astimezone(timezone('UTC')).replace(tzinfo=None)
        localised_end = tz.localize(timeframe[1]).astimezone(timezone('UTC')).replace(tzinfo=None)
    else:
        localised_start = timeframe[0].astimezone(timezone('UTC')).replace(tzinfo=None)
        localised_end = timeframe[1].astimezone(timezone('UTC')).replace(tzinfo=None)

    if upload_type == 'automatic':
        kind = 'RespeckAverage'
    else:
        kind = 'RespeckMinuteManualUpload'

    query = get_datastore_client().query(
        kind=kind,
        filters=[('subject_id', '=', subject_id),
                 ('timestamp', '>=', localised_start),
                 ('timestamp', '<', localised_end)],
        order=['timestamp']).fetch()

    results = [dict(e) for e in query]

    data = pd.DataFrame(results, columns=['timestamp', 'breathing_rate', 'sd_br', 'activity', 'act_type',
                                          'step_count'])
    data = data.rename(columns={'activity': 'activity_level', 'act_type': 'activity_type'})

    if len(data) > 0:
        data.loc[:, 'timestamp'] = data['timestamp'].dt.tz_localize(None)
        data.to_csv(out_filepath, index=False)


def download_airspeck_from_datastore(subject_id, out_filepath, project_name, timeframe, upload_type):
    assert upload_type == 'automatic', "Only automatic upload type implemented so far. Passed {}".format(upload_type)
    # assert upload_type in ['automatic', 'sd_card'], \
    #    "Upload type needs to be either automatic or sd_card. Download manual upload from Google storage."

    client = get_datastore_client()

    tz = timezone(project_mapping[project_name][1])

    if timeframe[0].tzinfo is None:
        utc_start = tz.localize(timeframe[0]).astimezone(timezone('UTC')).replace(
            tzinfo=None)
        utc_end = tz.localize(timeframe[1]).astimezone(timezone('UTC')).replace(
            tzinfo=None)
    else:
        utc_start = timeframe[0]
        utc_end = timeframe[1]

    if upload_type == 'automatic':
        kind = 'MobileAirspeck'
    else:
        kind = 'MobileAirspeckSDCard'

    with open(out_filepath, "w") as out:
        out.write("timestamp,pm1,pm2_5,pm10,bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9,bin10,bin11,bin12,"
                  "bin13,bin14,bin15,temperature,humidity,luxLevel,motion,battery,gpsLatitude,gpsLongitude,"
                  "gpsAccuracy\n")

        query = client.query(
            kind=kind,
            filters=[('subject_id', '=', subject_id), ('timestamp', '>=', utc_start),
                     ('timestamp', '<', utc_end)], order=['timestamp']).fetch()

        for e in query:
            out.write("{},{},{},{},".format(e['timestamp'].replace(tzinfo=None), e['pm1'], e['pm2_5'], e['pm10']))
            for i in range(0, 16):
                out.write("{},".format(e['bins'][i]))
            out.write("{},{},{},{},{},{},{},{}\n".format(e['temperature'], e['humidity'], e['lux'], e['motion'], e['battery'],
                                                   e['location'].latitude, e['location'].longitude,
                                                   e['gps_accuracy']))
