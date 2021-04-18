# coding=utf-8
import os
import shutil

import dateutil
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyPDF2 import PdfFileReader, PdfFileWriter
from google.cloud import storage
from matplotlib.lines import Line2D
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

from airspeck_map import get_maps_image
from constants import CB_color_cycle, project_mapping, peeps_reports_dir
from load_files import load_personal_airspeck_file, load_respeck_file, load_static_airspeck_file
from misc_utils import get_home_gps_for_subject, get_work_id_for_subject, get_home_id_for_subject
from project_specific.peeps_helpers import download_all_peeps_data, peeps_work_id_to_gps, peeps_work_id_to_gps_phase1, load_peeps_participant_details
from reports import get_image


def peeps_generate_graphs_and_create_report(subject_id, subject_visit_number=visit_number, upload_report_to_storage=False,
                                            remove_temporary_files=False, max_value=100, phase=1):
    temp_dir = peeps_reports_dir + 'temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print("Creating graphs")
    plot_peeps_report_graphs_for_subject(subject_id, subject_visit_number=visit_number, graphs_dir=temp_dir, max_value=max_value, phase=phase)

    report_filepath = peeps_reports_dir + "{}({})_report.pdf".format(subject_id, visit_number)
    merge_report_peeps(subject_id, report_filepath=report_filepath, graphs_dir=temp_dir)

    # Upload report to Google storage
    if upload_report_to_storage:
        print("Uploading to Google storage")
        client = storage.Client(project='specknet-pyramid-test')
        bucket = client.get_bucket('peeps-data')
        blob = bucket.blob(report_filepath)
        blob.upload_from_filename()

    # Send via email
    '''
    print("Sending via email")
    send_mail("peeps.report@gmail.com", ["dariusjfischer@gmail.com"], "Peeps report {}".format(subj_id),
              "Attached the report for subject {}. This report was automatically generated. "
              "Please contact dariusjfischer@gmail.com if there are any issues.".format(subj_id),
              files=[report_filepath])
    print("Report sent.")
    '''

    # Remove temporary files
    if remove_temporary_files:
        shutil.rmtree(temp_dir)

    print("Done")


def plot_peeps_report_graphs_for_subject(subject_id, subject_visit_number=1, graphs_dir, max_value=100):
    # Download raw data if not present
    #download_all_peeps_data(download_raw_airspeck_data=True, phase=phase)
    participant_details = load_peeps_participant_details()

    calibration_date_pers, is_calibrated_pm_pers, airspeck = get_resampled_data(subject_id, subject_visit_number) 
    
    if not is_calibrated_pm_pers:
        airspeck['pm2_5'] = airspeck['pm2_5'] * 0.60456 + 37.26849
        #calibration_date_pers = 'Calibrated with median'
        
    respeck = load_respeck_file(subject_id,  'peeps', subject_visit_number=subject_visit_number, upload_type='manual')
    
    #All subject details
    all_visit_subj_details = participant_details.loc[participant_details['Subject ID'] == subject_id]
    #Details for specific visit
    subj_details = all_visit_subj_details[all_visit_subj_details['Visit number'] == subject_visit_number] 
    
    infer = subj_details['Infer Home Coordinates'][0] 

    if infer :
        home_gps = airspeck.loc[(1 < airspeck.index.hour) & (airspeck.index.hour <= 3)].mean()
        print('Subject ID: ' + str(subject_id) + ' home_gps: ' + str(home_gps))
    # If there was no personal data during the night, fall back on the GPS coordinates the researchers provided
    #if pd.isnull(home_gps['gpsLatitude']):
    else:
        home_gps = get_home_gps_for_subject(subject_id, participant_details)

    # Select locations near home
    #radius_home = 0.01
    radius_home = 0.002
    home_id = get_home_id_for_subject(subject_id, participant_details)    
    
    correction_factor = airspeck['gpsAccuracy'] * 0.00001
    home_mask = (np.abs(airspeck['gpsLatitude'] - home_gps['gpsLatitude']) < radius_home + correction_factor) & \
                (np.abs(airspeck['gpsLongitude'] - home_gps['gpsLongitude']) < radius_home + correction_factor)
    
    all_factors_ids = ['33B45C90B13731DE', 'E1EFA8FCA05B3FF9']
    
    
    #Declare work and commute masks in case there is no work
    work_mask = np.zeros(len(airspeck)).astype(bool)
    commute_mask = np.zeros(len(airspeck)).astype(bool)
    
    radius_work = 0.01
    if subject_id in ['PEV018','PEV066','PEV047','PEV076']:
        radius_work = 0.002
        print('Smaller work radius set: {}'.format(radius_work))
    
    work_id = get_work_id_for_subject(subject_id, participant_details)
    hasWorkLocation = True
    if (work_id == 'Not deployed'):
        hasWorkLocation = False
        work_airspeck = []
    
    if hasWorkLocation: #We need a work location to establish commuting
        
        if subject_visit_number == 1:
            work_gps = peeps_work_id_to_gps_phase1[get_work_id_for_subject(subject_id, participant_details)]
        if subject_visit_number ==2:  
            work_gps = peeps_work_id_to_gps[get_work_id_for_subject(subject_id, participant_details)]

        use_all_features = False
            
        calibration_date_work, is_calibrated_spmwork, is_calibrated_sgaswork, work_airspeck = load_static_airspeck_file(work_id, sensor_label = "{}".format(subject_id), suffix_filename='_work', project_name='peeps', upload_type='automatic', calibrate_pm_and_gas=False, return_calibration_flag=False, use_all_features_for_pm_calibration=use_all_features)
            
        if pd.isnull(work_gps['gpsLatitude']):
            work_airspeck_lat = work_airspeck.gpsLatitude.loc[work_airspeck.gpsLatitude > 0].dropna().mean()
            work_airspeck_lng = work_airspeck.gpsLongitude.loc[work_airspeck.gpsLongitude > 0].dropna().mean()
            work_gps = {'gpsLatitude': work_airspeck_lat, 'gpsLongitude': work_airspeck_lng}
            

        if pd.isnull(work_gps['gpsLatitude']):
            print('Assuming work is wherever subjcet is between 10 and 11am')
            work_gps = airspeck.loc[(10 < airspeck.index.hour) & (airspeck.index.hour <= 11)].mean()
        
        work_mask = (np.abs(airspeck['gpsLatitude'] - work_gps['gpsLatitude']) < radius_work + correction_factor) & \
                (np.abs(airspeck['gpsLongitude'] - work_gps['gpsLongitude']) < radius_work + correction_factor)


        # Go through whole array and search for commuting
        begin_commute = 0
        last_location = ""
        for idx in range(1, len(airspeck)):

            if (home_mask[idx] == True and home_mask[idx - 1] == False and last_location == "work") or \
                (work_mask[idx] == True and work_mask[idx - 1] == False and last_location == "home"):
                #print('Finished commute')
                #commute_length = min(idx - begin_commute, 180) #if the commute is more than 3 hours it's not really a commute
                #print(commute_length)
                if (idx - begin_commute < 180):
                    commute_mask[begin_commute:idx] = True
            elif (home_mask[idx] == False and home_mask[idx - 1] == True):
                print('Entering leaving home commute')
                begin_commute = idx
                last_location = "home"
            elif (work_mask[idx] == False and work_mask[idx - 1] == True):
                print('Entering leaving work commute')
                begin_commute = idx
                last_location = "work"
                
        #print('hello2')

    ##################################
    # Draw detailed exposure plot
    ##################################
    sns.set_style('whitegrid', {'xtick.bottom': True, 'xtick.major.size': 5})

    fig, ax = plt.subplots(figsize=(15, 5))
    #print(airspeck.loc[home_mask])

    if np.count_nonzero(home_mask) > 0:
        for ts in airspeck.loc[home_mask].index:
            ax.axvspan(ts, ts + pd.DateOffset(minutes=1), facecolor=CB_color_cycle[0], alpha=0.3, zorder=1, lw=0)

    if np.count_nonzero(work_mask) > 0:
        for ts in airspeck.loc[work_mask].index:
            ax.axvspan(ts, ts + pd.DateOffset(minutes=1), facecolor=CB_color_cycle[1], alpha=0.3, zorder=1, lw=0)

    #print(np.count_nonzero(commute_mask))
    if np.count_nonzero(commute_mask) > 0:
        for ts in airspeck.loc[commute_mask].index:
            ax.axvspan(ts, ts + pd.DateOffset(minutes=1), facecolor=CB_color_cycle[2], alpha=0.3, zorder=1, lw=0)

    #airspeck_plot = airspeck.resample('10min').mean()
    ax.scatter(airspeck.resample('10min').mean().index, airspeck.resample('10min').mean()['pm2_5'], s=2, color='black', zorder=2)

    # Plot stationary airspeck home
    
    use_all_features = False
    #if home_id in all_factors_ids:
    #    use_all_features = True
            
    calibration_date_home, is_calibrated_spmhome, is_calibrated_sgashome, home_airspeck = load_static_airspeck_file(home_id, sensor_label = "{}".format(subject_id), suffix_filename='_home', project_name='peeps', upload_type='automatic', calibrate_pm_and_gas=False, use_all_features_for_pm_calibration=use_all_features, return_calibration_flag=False)

    home_airspeck = load_static_airspeck_file(subject_id, suffix_filename='_home')
    #home_aispeck_plot = home_airspeck.resample('10min').mean()
    start_personal = airspeck.index[0].replace(hour=0, minute=0, second=0)
    end_personal = airspeck.index[-1].replace(hour=0, minute=0, second=0) + pd.DateOffset(days=1)
    
    start_home = start_personal
    end_home = end_personal
    if len(home_airspeck) > 0:
        ax.scatter(home_airspeck.resample('10min').mean().index, home_airspeck.resample('10min').mean()['pm2_5'], s=2, color=CB_color_cycle[0], zorder=2)
        start_home = home_airspeck.index[0].replace(hour=0, minute=0, second=0)
        end_home = home_airspeck.index[-1].replace(hour=0, minute=0, second=0) + pd.DateOffset(days=1)
        
    # Plot stationary airspeck work
    #work_airpseck is already loaded above
    #work_airspeck = load_static_airspeck_file(subject_id, suffix_filename='_work')
    #work_aispeck_plot = work_airspeck.resample('10min').mean()
    start_work = start_personal
    end_work = end_personal
    if len(work_airspeck) > 0:
        ax.scatter(work_airspeck.resample('10min').mean().index, work_airspeck.resample('10min').mean()['pm2_5'], s=2, color=CB_color_cycle[1], zorder=2)
        start_work = work_airspeck.index[0].replace(hour=0, minute=0, second=0)
        end_work = work_airspeck.index[-1].replace(hour=0, minute=0, second=0) + pd.DateOffset(days=1)

    ax.set_ylabel("PM2.5 (μg/m³)")
    
    ax.set_xlim(min(start_personal, start_home, start_work), max( end_personal, end_home, end_work))

    formatter = mdates.DateFormatter('%d.%m %Hh', tz=dateutil.tz.gettz(project_mapping['peeps'][1]))

    ax.xaxis.set_major_formatter(formatter)

    ax.set_title("Continuous PM2.5 personal exposure levels and ambient concentrations")
    fig.autofmt_xdate()

    home_patch = mpatches.Patch(color=CB_color_cycle[0], label='Home', alpha=0.3)
    work_patch = mpatches.Patch(color=CB_color_cycle[1], label='Work', alpha=0.3)
    commute_patch = mpatches.Patch(color=CB_color_cycle[2], label='Commute', alpha=0.5)

    airs_home_patch = Line2D(range(1), range(1), marker='o', color='#00000000', markerfacecolor=CB_color_cycle[0],
                             label='Home sensor')
    airp_patch = Line2D(range(1), range(1), marker='o', color='#00000000', markerfacecolor="black",
                        label='Personal sensor')
    airs_work_patch = Line2D(range(1), range(1), marker='o', color='#00000000', markerfacecolor=CB_color_cycle[1],
                             label='Work sensor')
    plt.legend(handles=[home_patch, work_patch, commute_patch, airp_patch, airs_home_patch, airs_work_patch])

    plt.tight_layout()
    plt.savefig(graphs_dir + "{}_detailed_exposure.png".format(subject_id), dpi=300)
    plt.show()

    ##################################
    # Draw summary bar graph
    ##################################
    sns.set_style('darkgrid', {'xtick.bottom': False, 'xtick.major.size': 0.0})
    home = airspeck.loc[home_mask, 'pm2_5'].mean()
    work = airspeck.loc[work_mask, 'pm2_5'].mean()
    commute = airspeck.loc[commute_mask, 'pm2_5'].mean()
    other = airspeck.loc[~(work_mask | home_mask | commute_mask), 'pm2_5'].mean()
    overall = airspeck['pm2_5'].mean()
    home_ambient = home_airspeck['pm2_5'].mean()
    work_ambient = work_airspeck['pm2_5'].mean()

    mean_values = [home, work, commute, other, overall, home_ambient, work_ambient]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Mean PM2.5 personal exposure levels and ambient concentrations")
    ax.bar(np.arange(7), mean_values, width=0.5, color=CB_color_cycle, edgecolor="none")
    plt.xticks(np.arange(7),
               ["Home\npersonal", "Work\npersonal", "Commute\npersonal", "Other\npersonal", "Overall\npersonal",
                "Home\nambient", "Work\nambient"])
    ax.set_ylabel("PM2.5 (μg/m³)")
    plt.savefig(graphs_dir + "{}_mean_exposure.png".format(subject_id, subject_id), dpi=300)
    plt.show()

    ##################################
    # Draw map
    ##################################
    #print(airspeck)
    get_maps_image(airspeck, graphs_dir + "{}_airspeck_map.png".format(subject_id), zoom=13, max_value=max_value)

    ##################################
    # Other statistics
    ##################################
    # Append stats to this file
    with open(graphs_dir + "{}_stats.txt".format(subject_id), 'a') as f:
        
        if not is_calibrated_pm_pers:
            calibration_date_pers = 'Calibrated with median'
        if not is_calibrated_spmhome:
            calibration_date_home = 'Calibrated with median'
        if not is_calibrated_spmwork:
            calibration_date_work = 'Calibrated with median'
                    
        f.write("The personal, home and work data in this report were calibrated using colocation data from the following dates. If colocation data was unavailable, a median calibration based on other similar sensors was used.\n")
          
        f.write("Personal sensor: {}\n".format(calibration_date_pers))
        f.write("Home sensor: {}\n".format(calibration_date_home))
        f.write("Work sensor: {}\n".format(calibration_date_work))
        
        f.write("Step count: {}\n".format(respeck['step_count'].sum()))
        f.write("Mean breathing rate during night: {:.2f} breaths per minute\n".format(
            respeck.loc[(0 < respeck.index.hour) & (respeck.index.hour < 6),
                        'breathing_rate'].mean()))
        f.write("Mean breathing rate during day: {:.2f} breaths per minute\n".format(
            respeck.loc[(6 <= respeck.index.hour) & (respeck.index.hour <= 23),
                        'breathing_rate'].mean()))
        
        f.write("\nStart of recording: {}\n".format(airspeck.index[0].replace(tzinfo=None)))
        f.write("End of recording: {}\n".format(airspeck.index[-1].replace(tzinfo=None)))
        f.write("Total duration: {}\n".format(airspeck.index[-1] - airspeck.index[0]))

        f.write("Total recording time at work: {:.1f} h\n".format(np.count_nonzero(work_mask) / 60.))
        f.write("Total recording time at home: {:.1f} h\n".format(np.count_nonzero(home_mask) / 60.))
        f.write("Total recording time during the journey between home and work: {:.1f} h\n".format(
            np.count_nonzero(commute_mask) / 60.))
        
def get_resampled_data(subject_id, visit_number):
    calibration_date_pers, is_calibrated_pm_pers, airspeck_raw = load_personal_airspeck_file(subject_id, subject_visit_number=visit_number, project_name='peeps', upload_type='manual', is_minute_averaged=False, calibrate_pm_and_gas=False, return_calibration_flag=False)
    # If upload type is automatic -- Change to 'gpsLatitude':'gpsLongitude'
    #    airspeck_raw['gpsAccuracy'] = pd.to_numeric(airspeck_raw['gpsAccuracy'])

    airspeck_raw.loc[airspeck_raw['gpsAccuracy'] > 1000, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLatitude'] < 28.4, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLatitude'] > 28.9, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLongitude'] < 76.8, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLongitude'] > 77.6, 'gpsLongitude':'gpsLatitude'] = np.nan
    
    airspeck = airspeck_raw.resample('1min').mean()

    return calibration_date_pers, is_calibrated_pm_pers, airspeck


def merge_report_peeps(subj_id, report_filepath, graphs_dir):
    ps = ParagraphStyle(
        name='Normal',
        fontName='Helvetica',
        fontSize=10,
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(graphs_dir + "{}_graphs.pdf".format(subj_id), rightMargin=50,
                            leftMargin=50, topMargin=50, bottomMargin=50)
    parts = []

    image = get_image(graphs_dir + "{}_mean_exposure.png".format(subj_id), width=480)
    parts.append(image)

    parts.append(Spacer(width=0, height=35))
    lines = tuple(open(graphs_dir + "{}_stats.txt".format(subj_id), 'r'))
    for line in lines:
        parts.append(Paragraph(line, ps))

    parts.append(PageBreak())

    image = get_image(graphs_dir + "{}_airspeck_map.png".format(subj_id), width=480)
    parts.append(image)

    parts.append(Spacer(width=0, height=20))

    image = get_image(graphs_dir + "viridis_legend.png".format(subj_id), width=300) #was 220
    parts.append(image)

    parts.append(PageBreak())

    #image = get_image(graphs_dir + "{}_detailed_exposure.png".format(subj_id), rotated=True, width=720)
    image_2 = get_image(graphs_dir + "{}_detailed_exposure.png".format(subj_id), rotated=True, width=720)

    parts.append(image_2)

    doc.build(parts)

    output = PdfFileWriter()

    with open(peeps_reports_dir + "first_page.pdf", "rb") as f:
        cover_pdf = PdfFileReader(f)
        output.addPage(cover_pdf.getPage(0))

        with open(graphs_dir + "{}_graphs.pdf".format(subj_id), "rb") as f:
            rest_pdf = PdfFileReader(f)
            for p_idx in range(rest_pdf.getNumPages()):
                output.addPage(rest_pdf.getPage(p_idx))

            with open(report_filepath, "wb") as f:
                output.write(f)