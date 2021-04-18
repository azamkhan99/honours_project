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
from constants import CB_color_cycle, project_mapping, philap_reports_dir
from load_files import load_respeck_file, load_static_airspeck_file, load_personal_airspeck_file
from misc_utils import get_home_gps_for_subject
from project_specific.philap_helpers import load_philap_participant_details
from reports import get_image


def philap_generate_graphs_and_create_report(subject_id, upload_report_to_storage=False,
                                             remove_temporary_files=False):
    temp_dir = philap_reports_dir + 'temp/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print("Creating graphs")
    plot_philap_report_graphs_for_subject(subject_id, graphs_dir=temp_dir)

    print("Creating report")
    report_filepath = philap_reports_dir + "{}_report.pdf".format(subject_id)
    merge_report_philap(subject_id, report_filepath=report_filepath, graphs_dir=temp_dir)

    # Upload report to Google storage
    if upload_report_to_storage:
        print("Uploading to Google storage")
        client = storage.Client(project='specknet-pyramid-test')
        bucket = client.get_bucket('philap-data')
        blob = bucket.blob(report_filepath)
        blob.upload_from_filename()

    # Remove temporary files
    if remove_temporary_files:
        shutil.rmtree(temp_dir)

    print("Done")


def plot_philap_report_graphs_for_subject(subject_id, graphs_dir):
    # Download raw data if not present
    participant_details = load_philap_participant_details()

    try:
        airspeck_raw = load_personal_airspeck_file(subject_id, upload_type='manual', is_minute_averaged=False)
        airspeck = airspeck_raw.resample('1min').mean()
        respeck = load_respeck_file(subject_id, upload_type='manual')
    except:
        print("Please download all Peeps data via download_all_philap_data(raw_airspeck=True) "
              "before calling this function")

    # Delete incorrect GPS. These coordinates are just outside the larger area of Delhi
    airspeck_raw.loc[airspeck_raw['gpsAccuracy'] > 1000, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLatitude'] < 10, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLatitude'] > 40, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLongitude'] < 10, 'gpsLongitude':'gpsLatitude'] = np.nan
    airspeck_raw.loc[airspeck_raw['gpsLongitude'] > 80, 'gpsLongitude':'gpsLatitude'] = np.nan

    home_gps = airspeck.loc[(1 < airspeck.index.hour) & (airspeck.index.hour <= 3)].mean()
    # If there was no personal data during the night, fall back on the GPS coordinates the researchers provided
    if pd.isnull(home_gps['gpsLatitude']):
        home_gps = get_home_gps_for_subject(subject_id, participant_details)

    # Select locations near home
    radius_home = 0.002
    correction_factor = airspeck['gpsAccuracy'] * 0.00001
    home_mask = (np.abs(airspeck['gpsLatitude'] - home_gps['gpsLatitude']) < radius_home + correction_factor) & \
                (np.abs(airspeck['gpsLongitude'] - home_gps['gpsLongitude']) < radius_home + correction_factor)

    ##################################
    # Draw detailed exposure plot
    ##################################
    sns.set_style('darkgrid', {'xtick.bottom': True, 'xtick.major.size': 5})

    fig, ax = plt.subplots(figsize=(15, 5))

    if np.count_nonzero(home_mask) > 0:
        for ts in airspeck.loc[home_mask].index:
            ax.axvspan(ts, ts + pd.DateOffset(minutes=1), facecolor=CB_color_cycle[0], alpha=0.3, zorder=1)

    ax.scatter(airspeck.index, airspeck['pm2_5'], s=2, color='black', zorder=2)

    # Plot stationary airspeck home
    home_airspeck = load_static_airspeck_file(subject_id, suffix_filename='_home')
    if home_airspeck is not None and len(home_airspeck) > 0:
        ax.scatter(home_airspeck.index, home_airspeck['pm2_5'], s=2, color='blue')

    ax.set_ylabel("PM2.5 (μg/m³)")

    start = airspeck.index[0].replace(hour=0, minute=0, second=0)
    end = airspeck.index[-1].replace(hour=0, minute=0, second=0) + pd.DateOffset(days=1)
    ax.set_xlim(start, end)

    formatter = mdates.DateFormatter('%d.%m %Hh', tz=dateutil.tz.gettz(project_mapping['philap'][1]))

    ax.xaxis.set_major_formatter(formatter)

    ax.set_title("Continuous PM2.5 personal exposure levels and ambient concentrations")
    fig.autofmt_xdate()

    home_patch = mpatches.Patch(color=CB_color_cycle[0], label='Home', alpha=0.3)

    airs_home_patch = Line2D(range(1), range(1), marker='o', color='#00000000', markerfacecolor="blue",
                             label='Home sensor')
    airp_patch = Line2D(range(1), range(1), marker='o', color='#00000000', markerfacecolor="black",
                        label='Personal sensor')
    plt.legend(handles=[home_patch, airp_patch, airs_home_patch])

    plt.tight_layout()
    plt.savefig(graphs_dir + "{}_detailed_exposure.png".format(subject_id), dpi=300)
    plt.show()

    ##################################
    # Draw summary bar graph
    ##################################
    sns.set_style('darkgrid', {'xtick.bottom': False, 'xtick.major.size': 0.0})
    home = airspeck.loc[home_mask, 'pm2_5'].mean()
    other = airspeck.loc[~home_mask, 'pm2_5'].mean()
    overall = airspeck['pm2_5'].mean()
    if home_airspeck is not None:
        home_ambient = home_airspeck['pm2_5'].mean()
    else:
        home_ambient = np.nan

    mean_values = [home, other, overall, home_ambient]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Mean PM2.5 personal exposure levels and ambient concentrations")
    ax.bar(np.arange(len(mean_values)), mean_values, width=0.5, color=CB_color_cycle, edgecolor="none")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.set_xlim(-0.5, len(mean_values) - 0.5)
    plt.xticks(np.arange(len(mean_values)), ["Home\npersonal", "Other\npersonal", "Overall\npersonal", "Home\nambient"])
    plt.savefig(graphs_dir + "{}_mean_exposure.png".format(subject_id, subject_id), dpi=300)
    plt.show()

    ##################################
    # Draw map
    ##################################
    get_maps_image(airspeck_raw, graphs_dir + "{}_airspeck_map.png".format(subject_id), zoom=13)

    ##################################
    # Other statistics
    ##################################
    # Create new empty file
    open(graphs_dir + "{}_stats.txt".format(subject_id), 'w').close()

    # Append stats to this file
    with open(graphs_dir + "{}_stats.txt".format(subject_id), 'a') as f:
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

        f.write("Total recording time at home: {:.1f} h\n".format(np.count_nonzero(home_mask) / 60.))


def merge_report_philap(subj_id, report_filepath, graphs_dir):
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

    image = get_image(philap_reports_dir + "viridis_legend.png".format(subj_id), width=220)
    parts.append(image)

    parts.append(PageBreak())

    image = get_image(graphs_dir + "{}_detailed_exposure.png".format(subj_id), rotated=True, width=720)
    parts.append(image)

    doc.build(parts)

    output = PdfFileWriter()

    with open(philap_reports_dir + "first_page.pdf", "rb") as f:
        cover_pdf = PdfFileReader(f)
        output.addPage(cover_pdf.getPage(0))

        with open(graphs_dir + "{}_graphs.pdf".format(subj_id), "rb") as f:
            rest_pdf = PdfFileReader(f)
            for p_idx in range(rest_pdf.getNumPages()):
                output.addPage(rest_pdf.getPage(p_idx))

            with open(report_filepath, "wb") as f:
                output.write(f)
