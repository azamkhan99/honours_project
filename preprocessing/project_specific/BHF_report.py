# coding=utf-8
import os
import shutil
import copy

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
from constants import CB_color_cycle, project_mapping, bhf_reports_dir, bhf_participant_details_filepath
from download_data import download_personal_airspeck_data
from load_files import load_personal_airspeck_file
from reports import get_image


def bhf_generate_graphs_and_create_report(subject_id, upload_report_to_storage=False,
                                            remove_temporary_files=False, max_value=100, phase=1):
    temp_dir = bhf_reports_dir + 'temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print("Creating graphs")
    plot_bhf_report_graphs_for_subject(subject_id, graphs_dir=temp_dir, max_value=max_value, phase=phase)

    report_filepath = bhf_reports_dir + "{}_report.pdf".format(subject_id)
    merge_report_bhf(subject_id, report_filepath=report_filepath, graphs_dir=temp_dir)

    # Upload report to Google storage
    if upload_report_to_storage:
        print("Uploading to Google storage")
        client = storage.Client(project='specknet-pyramid-test')
        bucket = client.get_bucket('british-heart-data')
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
    
def download_all_bhf_data(overwrite_if_already_exists=False, download_raw_airspeck_data=True):
    project_name = 'BH'
    
    logs = pd.read_excel(bhf_participant_details_filepath)

    for idx, row in logs.iterrows():
        subject_id = row['Subject ID']

        if len(subject_id) != 6:
            continue
            
        if subject_id not in ['BHX001','BHF001']:
            continue

        from_time = row['Start Time']
        start_date = row['Start Date'].replace(hour=from_time.hour, minute=from_time.minute,
                                                          second=from_time.second)#.to_pydatetime()

        to_time = row['End Time']
        end_date = row['End Date'].replace(hour=to_time.hour, minute=to_time.minute,
                                                      second=to_time.second)#.to_pydatetime()

        timeframe = [start_date, end_date]
        # Download personal data if not yet present
        #is_minute_averaged=not download_raw_airspeck_data
        download_personal_airspeck_data(subject_id, upload_type='automatic', project_name=project_name,
                                        timeframe=timeframe,
                                        overwrite_if_already_exists=overwrite_if_already_exists)
    print("Finished!")


def plot_bhf_report_graphs_for_subject(subject_id, graphs_dir, max_value=100, phase=1):
    # Download raw data if not present
    download_all_bhf_data(download_raw_airspeck_data=True)
    participant_details = pd.read_excel(bhf_participant_details_filepath)

    airspeck = get_resampled_data(subject_id) #airspeck_raw.resample('1min').mean()
         
    subj_details = participant_details.loc[participant_details['Subject ID'] == subject_id]
    green_gps = {'gpsLatitude':subj_details['Green Space Lat'].values[0], 'gpsLongitude':subj_details['Green Space Lng'].values[0]}
    
    transport_gps = {'gpsLatitude':subj_details['Transport Hub Lat'].values[0], 'gpsLongitude':subj_details['Transport Hub Lng'].values[0]}
    
    school_gps = {'gpsLatitude':subj_details['School Lat'].values[0], 'gpsLongitude':subj_details['School Lng'].values[0]}
    
   
    
     #print('Subject ID: ' + str(subject_id) + ' home_gps: ' + str(home_gps))

    # Select locations near green space
    radius_green = 0.01
    #radius_home = 0.002
    radius_transport = 0.01
    radius_school = 0.01
    correction_factor = airspeck['gpsAccuracy'] * 0.00001
    
    #0.001 is approx 100m
    
    green_mask = (np.abs(airspeck['gpsLatitude'] - green_gps['gpsLatitude']) < radius_green + correction_factor) & \
                (np.abs(airspeck['gpsLongitude'] - green_gps['gpsLongitude']) < radius_green + correction_factor)
    
    transport_mask = (np.abs(airspeck['gpsLatitude'] - transport_gps['gpsLatitude']) < radius_transport + correction_factor) & \
                (np.abs(airspeck['gpsLongitude'] - transport_gps['gpsLongitude']) < radius_transport + correction_factor)

    school_mask = (np.abs(airspeck['gpsLatitude'] - school_gps['gpsLatitude']) < radius_school + correction_factor) & \
                (np.abs(airspeck['gpsLongitude'] - school_gps['gpsLongitude']) < radius_school + correction_factor)

    #Declare work and commute masks in case there is no work

    ## Get mean for each day and plot with date as label.
    subj_details = participant_details.loc[participant_details['Subject ID'] == subject_id]
    print(subj_details)
    print(subj_details['Start Date'])
    start_date = subj_details['Start Date'][1]
    end_date = subj_details['End Date'][1]
    
    current_date = start_date
    mean_values = []
    dates = []
    labels = []
    while current_date <= end_date:
        next_date = current_date + pd.Timedelta(days=1)
        mean_exposure = airspeck['pm2_5'][current_date:next_date].mean()
        print(current_date)
        print(mean_exposure)
        print(np.isnan(mean_exposure))
        if np.isnan(mean_exposure): #Don't plot if there is no data that day
            print('hello')
            current_date = next_date
            continue
        mean_values.append(mean_exposure)
        dates.append(current_date)
        labels.append(current_date.strftime('%a %d %b %Y'))
        current_date = next_date
        
    
    
    print(dates)
    print(mean_values)
    
    sns.set_style('darkgrid', {'xtick.bottom': False, 'xtick.major.size': 1.0})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Daily mean PM2.5 personal exposure levels vs WHO mean PM2.5 guidelines")
    ax.bar(range(len(dates)), mean_values, width=0.5, color=CB_color_cycle, edgecolor="none")
   
    #ax.xaxis.set_major_locator(plt.MaxNLocator((end_date - start_date).days - 1))
    #formatter = mdates.DateFormatter('%d.%m.%y', tz=dateutil.tz.gettz(project_mapping['british-heart'][1]))

    #ax.xaxis.set_major_formatter(formatter)
    
    ax.set_ylabel("PM2.5 (μg/m³)")
    #fig.autofmt_xdate()

    #ax.hlines([10, 25], start_date - pd.Timedelta(days=0.5), end_date + pd.Timedelta(days=0.5), color='black', linestyle='--')
    #ax.hlines(25)
    line_start =  start_date - pd.Timedelta(days=0.5)
    line_end = end_date + pd.Timedelta(days=0.5)
    l2 = ax.axhline(25, 0, 1, color='black', linestyle='--')
    l1 = ax.axhline(10, 0, 1, color='grey', linestyle='--')
    
    
    
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(labels)#, rotation=65)
    #plt.xticks( [ range(len(dates)) ] , dates) #.strftime('%a %d %b %Y'))

    #plt.xticks([r + barWidth for r in range(60)], out_df['Subject ID'][DAP_mask][:60])

    plt.legend([l2,l1],['WHO 24-hour mean guideline', 'WHO annual mean guideline'])
    plt.savefig(graphs_dir + "{}_mean_exposure.png".format(subject_id), dpi=300)

    plt.show()
    
    ## Also plot horizontal line at WHO limit
    ## 10 micro g per m3 annual mean
    ## 25 micro g per m3 24-hour mean
    #World Health Organization. Air quality guidelines: global update 2005: particulate matter, ozone, nitrogen dioxide, and sulfur dioxide. World Health Organization, 2006.

    #Note: update is planned for 2020
    

    ##################################
    # Draw detailed exposure plot
    ##################################
   
    '''
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
    home_airspeck = load_static_airspeck_file(subject_id, suffix_filename='_home')
    #home_aispeck_plot = home_airspeck.resample('10min').mean()
    if len(home_airspeck) > 0:
        ax.scatter(home_airspeck.resample('10min').mean().index, home_airspeck.resample('10min').mean()['pm2_5'], s=2, color=CB_color_cycle[0], zorder=2)

    # Plot stationary airspeck work
    #work_airspeck = load_static_airspeck_file(subject_id, suffix_filename='_work')
    #work_aispeck_plot = work_airspeck.resample('10min').mean()
    if len(work_airspeck) > 0:
        ax.scatter(work_airspeck.resample('10min').mean().index, work_airspeck.resample('10min').mean()['pm2_5'], s=2, color=CB_color_cycle[1], zorder=2)

    ax.set_ylabel("PM2.5 (μg/m³)")

    start_personal = airspeck.index[0].replace(hour=0, minute=0, second=0)
    start_home = home_airspeck.index[0].replace(hour=0, minute=0, second=0)
    end_personal = airspeck.index[-1].replace(hour=0, minute=0, second=0) + pd.DateOffset(days=1)
    end_home = home_airspeck.index[-1].replace(hour=0, minute=0, second=0) + pd.DateOffset(days=1)
    ax.set_xlim(min(start_personal, start_home), max( end_personal, end_home))

    formatter = mdates.DateFormatter('%d.%m %Hh', tz=dateutil.tz.gettz(project_mapping['british-heart'][1]))

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
    '''

    ##################################
    # Draw map
    ##################################
    #print(airspeck)
        
    airspeck21 = copy.copy(airspeck)
        
    airspeck21.loc[airspeck21['gpsLatitude'] < 53.304846, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck21.loc[airspeck21['gpsLatitude'] > 53.331722, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck21.loc[airspeck21['gpsLongitude'] < -3.513392, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck21.loc[airspeck21['gpsLongitude'] >  -3.444864, 'gpsLatitude':'gpsLongitude'] = np.nan

    markers1 = {'Rhyl Botanical Garden and Tennis Club':{'gpsLatitude':53.319804, 'gpsLongitude':-3.473707},'Rhyl Train Station':{'gpsLatitude':53.318637, 'gpsLongitude':-3.488818}}
    get_maps_image(airspeck21, graphs_dir + "{}_airspeck_map1.png".format(subject_id), project_name='british-heart',zoom=15, max_value=max_value, markers=markers1)
    
    #print(airspeck)
    
    airspeck2 = copy.copy(airspeck)

    airspeck2.loc[airspeck2['gpsLatitude'] < 53.31421, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck2.loc[airspeck2['gpsLatitude'] > 53.349586, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck2.loc[airspeck2['gpsLongitude'] < -3.434269, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck2.loc[airspeck2['gpsLongitude'] >  -3.373114, 'gpsLatitude':'gpsLongitude'] = np.nan
    
    markers2 = {'Meliden, Meliden Road':{'gpsLatitude':53.316697, 'gpsLongitude':-3.408955},'Prestatyn, High Street':{'gpsLatitude':53.332781, 'gpsLongitude':-3.401725}}
    get_maps_image(airspeck2, graphs_dir + "{}_airspeck_map2.png".format(subject_id), project_name='british-heart',zoom=15, max_value=max_value, markers=markers2)
    
    #53.270086, -3.472625
    #53.241077, -3.392396
    airspeck3 = copy.copy(airspeck)
    
    airspeck3.loc[airspeck3['gpsLatitude'] < 53.241077, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck3.loc[airspeck3['gpsLatitude'] > 53.270086, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck3.loc[airspeck3['gpsLongitude'] < -3.472625, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck3.loc[airspeck3['gpsLongitude'] >  -3.392396, 'gpsLatitude':'gpsLongitude'] = np.nan
    
    markers3 = {'Ysgol Uwchradd Glan Clwyd \n (Secondary School), St Astaph':{'gpsLatitude':53.255817, 'gpsLongitude':-3.439077}}
    get_maps_image(airspeck3, graphs_dir + "{}_airspeck_map3.png".format(subject_id), project_name='british-heart',zoom=15, max_value=max_value, markers=markers3)
    
    airspeck4 = copy.copy(airspeck)
    
    airspeck4.loc[airspeck4['gpsLatitude'] < 53.177262, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck4.loc[airspeck4['gpsLatitude'] > 53.192222, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck4.loc[airspeck4['gpsLongitude'] < -3.435402, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck4.loc[airspeck4['gpsLongitude'] > -3.404903, 'gpsLatitude':'gpsLongitude'] = np.nan
    
    markers4 = {'Highstreet in Denbigh, Vale Street':{'gpsLatitude':53.184038, 'gpsLongitude':-3.418185}}
    get_maps_image(airspeck4, graphs_dir + "{}_airspeck_map4.png".format(subject_id), project_name='british-heart',zoom=15, max_value=max_value, markers=markers4)
    
    get_maps_image(airspeck, graphs_dir + "{}_airspeck_map.png".format(subject_id), project_name='british-heart',zoom=13, max_value=max_value, markers=[])


    ##################################
    # Other statistics
    ##################################
    # Append stats to this file
    with open(graphs_dir + "{}_stats.txt".format(subject_id), 'a') as f:
        '''f.write("Step count: {}\n".format(respeck['step_count'].sum()))
        f.write("Mean breathing rate during night: {:.2f} breaths per minute\n".format(
            respeck.loc[(0 < respeck.index.hour) & (respeck.index.hour < 6),
                        'breathing_rate'].mean()))
        f.write("Mean breathing rate during day: {:.2f} breaths per minute\n".format(
            respeck.loc[(6 <= respeck.index.hour) & (respeck.index.hour <= 23),
                        'breathing_rate'].mean()))'''

        #f.write("\nStart of recording: {}\n".format(airspeck.index[0].replace(tzinfo=None)))
        #f.write("End of recording: {}\n".format(airspeck.index[-1].replace(tzinfo=None)))
        #f.write("Total duration of data collected: {}\n".format(airspeck.index[-1] - airspeck.index[0]))

        #f.write("Total recording time at green space: {:.1f} h\n".format(np.count_nonzero(green_mask) / 60.))
        #f.write("Total recording time at transport hub: {:.1f} h\n".format(np.count_nonzero(transport_mask) / 60.))
        #f.write("Total recording time at school: {:.1f} h\n".format(
            #np.count_nonzero(school_mask) / 60.))
        
        num_highest_exposure = 3
        
        airspeck5 = airspeck.resample('5min').mean()
        largest_pms = airspeck5.nlargest(num_highest_exposure, 'pm2_5')
        
        largest_pms.insert(2, "location",[None,None,None ], True)
        radius = 0.003
        POIs = participant_details[["POI1 Name","POI1 Lat","POI1 Lng","POI2 Name","POI2 Lat","POI2 Lng","POI3 Name","POI3 Lat","POI3 Lng","POI4 Name","POI4 Lat","POI4 Lng","POI5 Name","POI5 Lat","POI5 Lng",'POI6 Name',"POI6 Lat", 'POI6 Lng']]

        for i in range(num_highest_exposure):
            if (np.abs(largest_pms['gpsLatitude'][i] - POIs["POI1 Lat"][1]) < radius and
                np.abs(largest_pms['gpsLongitude'][i] - POIs["POI1 Lng"][1]) < radius) :
                largest_pms['location'][i] = POIs["POI1 Name"][1]
            if (np.abs(largest_pms['gpsLatitude'][i] - POIs["POI2 Lat"][1]) < radius and
                np.abs(largest_pms['gpsLongitude'][i] - POIs["POI2 Lng"][1]) < radius) :
                largest_pms['location'][i] = POIs["POI2 Name"][1]
            if (np.abs(largest_pms['gpsLatitude'][i] - POIs["POI3 Lat"][1]) < radius and
                np.abs(largest_pms['gpsLongitude'][i] - POIs["POI3 Lng"][1]) < radius) :
                largest_pms['location'][i] = POIs["POI3 Name"][1]
            if (np.abs(largest_pms['gpsLatitude'][i] - POIs["POI4 Lat"][1]) < radius and
                np.abs(largest_pms['gpsLongitude'][i] - POIs["POI4 Lng"][1]) < radius) :
                largest_pms['location'][i] = POIs["POI4 Name"][1]
            if (np.abs(largest_pms['gpsLatitude'][i] - POIs["POI5 Lat"][1]) < radius and
                np.abs(largest_pms['gpsLongitude'][i] - POIs["POI5 Lng"][1]) < radius) :
                largest_pms['location'][i] = POIs["POI5 Name"][1]
            if (np.abs(largest_pms['gpsLatitude'][i] - POIs["POI6 Lat"][1]) < radius and
                np.abs(largest_pms['gpsLongitude'][i] - POIs["POI6 Lng"][1]) < radius) :
                largest_pms['location'][i] = POIs["POI6 Name"][1]
        

    
   
        f.write("The {} highest PM exposure averaged over 5 minute periods were:\n".format(num_highest_exposure))
        for i in range(num_highest_exposure):
            timestamp =  largest_pms.index[i].replace(tzinfo=None)
            timestring = timestamp.strftime('%H:%M on %A %d %B')
            
            if largest_pms['location'][i] is None:
            
                f.write("{:.2f} (μg/m³) at {} near location {:.5f}, {:.5f}\n".format(largest_pms['pm2_5'][i], timestring, largest_pms['gpsLatitude'][i], largest_pms['gpsLongitude'][i]))
            else: 
                f.write("{:.2f} (μg/m³) at {} near location {}\n".format(largest_pms['pm2_5'][i], timestring, largest_pms['location'][i]))
    
            #f.write("{:.2f} (μg/m³) at time {} and location {:.5f}, {:.5f}\n".format(largest_pms['pm2_5'][i], largest_pms.index[i].replace(tzinfo=None), largest_pms['gpsLatitude'][i], largest_pms['gpsLongitude'][i]))
        

        
def get_resampled_data(subject_id):
    calibration_date_pers, is_calibrated_pm_pers, airspeck_raw = load_personal_airspeck_file(subject_id, 'british-heart',    upload_type='automatic', calibrate_pm_and_gas=True, return_calibration_flag=True)
    #airspeck_raw = load_personal_airspeck_file(subject_id, project_name='british-heart', upload_type='automatic')

    airspeck_raw['gpsAccuracy'] = pd.to_numeric(airspeck_raw['gpsAccuracy'])
    
    airspeck_raw.loc[airspeck_raw['gpsAccuracy'] > 1000, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLatitude'] < 49.88, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLatitude'] > 55.79, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLongitude'] < -5.9, 'gpsLatitude':'gpsLongitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLongitude'] > 1.8, 'gpsLatitude':'gpsLongitude'] = np.nan
    
    airspeck = airspeck_raw.resample('1min').mean()
    
    return airspeck

def merge_report_bhf(subj_id, report_filepath, graphs_dir):
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
    
    image = get_image(graphs_dir + "{}_airspeck_map1.png".format(subj_id), width=480)
    parts.append(image)
    
    image = get_image(graphs_dir + "{}_airspeck_map2.png".format(subj_id), width=480)
    parts.append(image)
    
    parts.append(PageBreak())
    
    image = get_image(graphs_dir + "{}_airspeck_map3.png".format(subj_id), width=480)
    parts.append(image)
    
    image = get_image(graphs_dir + "{}_airspeck_map4.png".format(subj_id), width=480)
    parts.append(image)

    #parts.append(PageBreak())

    #image = get_image(graphs_dir + "{}_detailed_exposure.png".format(subj_id), rotated=True, width=720)
    #image_2 = get_image(graphs_dir + "{}_detailed_exposure.png".format(subj_id), rotated=True, width=720)

    #parts.append(image_2)

    doc.build(parts)

    output = PdfFileWriter()

    with open(bhf_reports_dir + "first_page.pdf", "rb") as f:
        cover_pdf = PdfFileReader(f)
        output.addPage(cover_pdf.getPage(0))

        with open(graphs_dir + "{}_graphs.pdf".format(subj_id), "rb") as f:
            rest_pdf = PdfFileReader(f)
            for p_idx in range(rest_pdf.getNumPages()):
                output.addPage(rest_pdf.getPage(p_idx))

            with open(report_filepath, "wb") as f:
                output.write(f)
