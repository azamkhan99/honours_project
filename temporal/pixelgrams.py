import os
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pytz import timezone

from constants import *
from download_data import download_respeck_and_personal_airspeck_data, download_respeck_data
from load_files import get_uuid_for_subj_id_airspeck_personal, load_respeck_file, load_personal_airspeck_file
from misc_utils import add_speed_to_gps_data, get_project_for_subject

recording_present_colours = ["#ffffff", "#000000", "#ffb400", "#377eb8"]


def plot_combined_pixelgram(subject_id, respeck_data, airspeck_data, plot_filepath=None,
                            overwrite_if_already_exists=False, subject_visit_number=None):
    """
    airspeck_data and respeck_data should be minute average pandas dataframes, with the timestamp as index
    """

    if (respeck_data is None or len(respeck_data) == 0) and (airspeck_data is None or len(airspeck_data) == 0):
        print("Skipping Pixelgram as no data was passed")
        return

    if plot_filepath is None:
        directory = project_mapping[get_project_for_subject(subject_id)][3]
        if subject_visit_number is None:
            plot_filepath = directory + "{}_combined_pixelgram.png".format(subject_id)
        else:
            plot_filepath = directory + "{}({})_combined_pixelgram.png".format(subject_id, subject_visit_number)

    if not overwrite_if_already_exists and os.path.exists(plot_filepath):
        print("Pixelgram already exists.")
        return  # The file was already created, so stop execution of this function

    sns.reset_orig()
    # Hour offset so that the legend can be displayed inside graph. Legend doesn't show at the moment,
    # so set to 0
    hours_offset = 9
    hours, num_days, time_grid = prepare_grid(respeck_data, airspeck_data, hours_offset)

    # Calculate GPS speed
    add_speed_to_gps_data(airspeck_data, 'gpsLatitude', 'gpsLongitude')

    # Normalise data
    norm_GPS = normalise_into_range(airspeck_data['speed'], 0, 20)
    norm_lux = normalise_into_range(airspeck_data['luxLevel'], 0, 4)

    pm_95_percentile = np.nanpercentile(airspeck_data['pm2_5'], 95)
    pm_max = pm_95_percentile if pm_95_percentile > 50 else 50
    norm_pm2_5 = normalise_into_range(airspeck_data['pm2_5'], 0, pm_max)

    norm_act = normalise_into_range(respeck_data['activity_level'], 0, 0.7)
    norm_br = normalise_into_range(respeck_data['breathing_rate'], 10, 40)

    # Create grids
    pm25_grid = np.zeros_like(time_grid, dtype=float)
    lux_grid = np.zeros_like(time_grid, dtype=float)
    gps_grid = np.zeros_like(time_grid, dtype=float)
    actlevel_grid = np.zeros_like(time_grid, dtype=float)
    br_grid = np.zeros_like(time_grid, dtype=float)
    acttype_grid = np.zeros_like(time_grid, dtype=int)

    # Fill in RESpeck data
    for idx in range(len(respeck_data['timestamp'])):
        time_diff_minutes = int((respeck_data['timestamp'][idx] -
                                 time_grid[0][0]).total_seconds() / 60.)
        idx2d = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))

        br_grid[idx2d] = norm_br[idx]
        actlevel_grid[idx2d] = norm_act[idx]
        if not np.isnan(norm_act[idx]):
            acttype_grid[idx2d] = respeck_data['activity_type'][idx] + 1

    # Same for Airspeck
    for idx in range(len(airspeck_data['timestamp'])):
        time_diff_minutes = int((airspeck_data['timestamp'][idx] -
                                 time_grid[0][0]).total_seconds() / 60.)
        idx2d = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))

        lux_grid[idx2d] = norm_lux[idx]
        gps_grid[idx2d] = norm_GPS[idx]
        pm25_grid[idx2d] = norm_pm2_5[idx]

    fig, axes = plt.subplots(1, 6)
    fig.set_size_inches((30, num_days * 4.5))

    prepare_axes(axes, hours_offset, hours, num_days, time_grid)

    plot_column(axes[0], lux_grid, "Lux level (0-4)")
    plot_column(axes[1], br_grid, "Breathing rate (10-30 BrPM)")
    plot_column(axes[2], pm25_grid, "PM 2.5, in ug/m3 (0 - {})".format(int(pm_max)))
    plot_column(axes[3], acttype_grid, is_activity_type=True)
    plot_column(axes[4], actlevel_grid, "Activity level (0-0.7)")
    plot_column(axes[5], gps_grid, "Speed (0-20 km/h)")

    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)


def plot_airspeck_pixelgram(subject_id, airspeck_data, plot_filepath=None, overwrite_if_already_exists=False,
                            subject_visit_number=None):
    if airspeck_data is None or len(airspeck_data) == 0:
        print("Skipping Pixelgram as no data was passed")
        return

    if plot_filepath is None:
        directory = project_mapping[get_project_for_subject(subject_id)][3]
        if subject_visit_number is None:
            plot_filepath = directory + "{}_airspeck_pixelgram.png".format(subject_id)
        else:
            plot_filepath = directory + "{}({})_airspeck_pixelgram.png".format(subject_id, subject_visit_number)

    if not overwrite_if_already_exists and os.path.exists(plot_filepath):
        print("Pixelgram already exists.")
        return  # The file was already created, so stop execution of this function

    sns.reset_orig()
    hours, num_days, time_grid = prepare_grid(airspeck_data)

    # Load data into grid
    pm_95_percentile = np.nanpercentile(airspeck_data['pm2_5'], 95)
    pm_max = pm_95_percentile if pm_95_percentile > 50 else 50
    norm_pm2_5 = normalise_into_range(airspeck_data['pm2_5'], 0, pm_max)
    norm_lux = normalise_into_range(airspeck_data['luxLevel'], 0, 3)

    add_speed_to_gps_data(airspeck_data, 'gpsLatitude', 'gpsLongitude')
    norm_GPS = normalise_into_range(airspeck_data['speed'], 0, 20)
    norm_gps_accuracy = normalise_into_range(airspeck_data['gpsAccuracy'], 0, 100)
    norm_motion = normalise_into_range(airspeck_data['motion'], 0, 500)

    pm25_grid = np.zeros_like(time_grid, dtype=float)
    lux_grid = np.zeros_like(time_grid, dtype=float)
    recording_grid = np.zeros_like(time_grid, dtype=float)
    gps_grid = np.zeros_like(time_grid, dtype=float)
    gps_acc_grid = np.zeros_like(time_grid, dtype=float)
    motion_grid = np.zeros_like(time_grid, dtype=float)

    for idx in range(len(airspeck_data['timestamp'])):
        time_diff_minutes = int((airspeck_data['timestamp'][idx] - time_grid[0][0]).total_seconds() / 60.)
        idx2d = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))

        pm25_grid[idx2d] = norm_pm2_5[idx]
        lux_grid[idx2d] = norm_lux[idx]
        recording_grid[idx2d] = 1
        gps_grid[idx2d] = norm_GPS[idx]
        gps_acc_grid[idx2d] = norm_gps_accuracy[idx]
        motion_grid[idx2d] = norm_motion[idx]

    # Plot
    fig, axes = plt.subplots(1, 6)
    fig.set_size_inches((30, num_days * 4.5))

    prepare_axes(axes, 0, hours, num_days, time_grid)

    plot_column(axes[0], recording_grid, "Recording present (dark = yes)")
    plot_column(axes[1], pm25_grid, "PM 2.5, in ug/m3 (0 - {})".format(int(pm_max)))
    plot_column(axes[2], lux_grid, "Lux level (0-4)")
    plot_column(axes[3], gps_grid, "Speed (0-20 km/h)")
    plot_column(axes[4], gps_acc_grid, "Accuracy of GPS in meters.\nDark = low accuracy = probably inside.")
    plot_column(axes[5], motion_grid, "Level of activity. The darker, the more activity.")

    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)


def plot_respeck_pixelgram(subject_id, respeck_data, plot_filepath=None, overwrite_if_already_exists=False,
                           subject_visit_number=None):
    if respeck_data is None or len(respeck_data) == 0:
        print("Skipping Pixelgram as no data was passed")
        return

    if plot_filepath is None:
        directory = project_mapping[get_project_for_subject(subject_id)][3]
        if subject_visit_number is None:
            plot_filepath = directory + "{}_respeck_pixelgram.png".format(subject_id)
        else:
            plot_filepath = directory + "{}({})_respeck_pixelgram.png".format(subject_id, subject_visit_number)

    if not overwrite_if_already_exists and os.path.exists(plot_filepath):
        print("Pixelgram already exists.")
        return  # The file was already created, so stop execution of this function

    sns.reset_orig()
    # Hour offset so that the legend can be displayed inside graph.
    hours_offset = 9
    hours, num_days, time_grid = prepare_grid(respeck_data, hours_offset=hours_offset)

    # Load data into grid
    norm_act = normalise_into_range(respeck_data['activity_level'], 0, 0.7)
    norm_br = normalise_into_range(respeck_data['breathing_rate'], 10, 40)
    norm_stepcount = normalise_into_range(respeck_data['step_count'], 0, 120.)

    actlevel_grid = np.zeros_like(time_grid, dtype=float)
    br_grid = np.zeros_like(time_grid, dtype=float)
    acttype_grid = np.zeros_like(time_grid, dtype=int)
    stepcount_grid = np.zeros_like(time_grid, dtype=float)

    for idx in range(len(respeck_data)):
        time_diff_minutes = int((respeck_data.index[idx] - time_grid[0][0]).total_seconds() / 60.)
        idx2d = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))

        br_grid[idx2d] = norm_br[idx]
        actlevel_grid[idx2d] = norm_act[idx]
        acttype_grid[idx2d] = respeck_data['activity_type'][idx] + 1
        stepcount_grid[idx2d] = norm_stepcount[idx]

    # Plot
    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches((20, (num_days + hours_offset / 24.) * 4.5))

    prepare_axes(axes, hours_offset, hours, num_days, time_grid)

    plot_column(axes[0], br_grid, "Breathing rate (10-40 BrPM)")
    plot_column(axes[1], acttype_grid, is_activity_type=True)
    plot_column(axes[2], actlevel_grid, "Activity level (0-0.7)")
    plot_column(axes[3], stepcount_grid, "Step counts (0-120 steps)")

    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)


def prepare_grid(first_data_minute_values, second_data_minute_values=None, hours_offset=0):
    if first_data_minute_values is not None and len(
            first_data_minute_values) > 0 and second_data_minute_values is not None and len(
        second_data_minute_values) > 0:
        start_recording_day = min(second_data_minute_values['timestamp'].iloc[0].replace(hour=0, minute=0, second=0),
                                  first_data_minute_values['timestamp'].iloc[0].replace(hour=0, minute=0,
                                                                                        second=0)) - timedelta(
            hours=hours_offset)
        end_recording_day = (max(second_data_minute_values['timestamp'].iloc[-1].replace(hour=0, minute=0, second=0),
                                 first_data_minute_values['timestamp'].iloc[-1].replace(hour=0, minute=0, second=0))
                             + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif first_data_minute_values is not None and len(first_data_minute_values) > 0:
        start_recording_day = first_data_minute_values['timestamp'].iloc[0].replace(hour=0, minute=0,
                                                                                    second=0) - timedelta(
            hours=hours_offset)
        end_recording_day = (
                first_data_minute_values['timestamp'].iloc[-1].replace(hour=0, minute=0, second=0) + timedelta(
            days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif second_data_minute_values is not None and len(second_data_minute_values) > 0:
        start_recording_day = second_data_minute_values['timestamp'].iloc[0].replace(hour=0, minute=0,
                                                                                     second=0) - timedelta(
            hours=hours_offset)
        end_recording_day = (
                second_data_minute_values['timestamp'].iloc[-1].replace(hour=0, minute=0, second=0) + timedelta(
            days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError("Both respeck and airspeck data are empty")

    number_of_timestamps_hour = int(np.ceil((end_recording_day - start_recording_day).total_seconds() / 60 ** 2))
    hours_datetime = np.array(
        [start_recording_day + timedelta(hours=i) for i in range(number_of_timestamps_hour)])

    hours = np.asarray([datetime.strftime(date, "%d.%m %Hh") for date in hours_datetime])

    num_days = int((end_recording_day - start_recording_day).total_seconds() / (60 * 60 * 24.))

    number_of_timestamps_minutes = int(np.ceil((end_recording_day - start_recording_day).total_seconds() / 60.))
    time_grid = np.array(
        [start_recording_day + timedelta(minutes=i) for i in range(number_of_timestamps_minutes)]).reshape(-1, 60)

    return hours, num_days, time_grid


def prepare_axes(axes, hours_offset, hours, num_days, time_grid):
    # Prepare all columns with ticks, lines for start of day, and ticks for those lines.
    every_fifth = np.arange(hours_offset, len(time_grid), 4)

    # Same for every grid plot
    for ax in axes:
        ax.yaxis.set_ticklabels(hours[every_fifth])
        ax.set_yticks(np.arange(0, len(time_grid)), minor=True)
        ax.set_yticks(np.arange(0, len(time_grid))[every_fifth])
        ax.set_xlabel("Minutes")
        ax.invert_yaxis()

        # Draw lines at the start of each day
        for ts_idx in np.linspace(hours_offset, len(time_grid), num_days + 1)[:-1]:
            ax.axhline(int(ts_idx), 0, 60, color='black', linewidth=1.5)

            # Draw minute ticks for those lines
            for m in range(10, 60, 10):
                line_pos = 1 - (ts_idx / len(time_grid))
                tick_length = 0.03 / ((num_days + hours_offset / 24.) * 4.5)
                ax.axvline(m, line_pos - tick_length, line_pos + tick_length, color='black', linewidth=1)


def plot_column(ax, values, title="", is_activity_type=False, is_recording_present_column=False):
    if is_activity_type:
        ax.set_title("Activity type")
        ax.pcolormesh(values, cmap=ListedColormap(activity_colors), vmin=0, vmax=11)

        handles = []
        # Create legend in more readable form (cluster similar activities)
        for idx in [0, 4, 3, 7, 8, 9, 1, 5, 6, 10, 2]:
            handles.append(Patch(color=activity_colors[idx], label=activity_name[idx]))
        ax.legend(handles=handles, loc="upper left")
    elif is_recording_present_column:
        ax.set_title("PM recording present?")
        ax.pcolormesh(values, cmap=ListedColormap(recording_present_colours), vmin=0, vmax=3)

        handles = [Patch(color=recording_present_colours[0], label="No recording"),
                   Patch(color=recording_present_colours[1], label="Recording"),
                   Patch(color=recording_present_colours[2], label="Recording during SHS"),
                   Patch(color=recording_present_colours[3], label="Logged beginning and end of recording")]

        ax.legend(handles=handles, loc="upper left")
    else:
        ax.set_title(title)
        ax.pcolormesh(values, cmap='Greys', vmin=0, vmax=1)


def normalise_into_range(series, minimum, maximum):
    norm_series = series.copy()
    norm_series = (norm_series - float(minimum)) / (maximum - minimum)
    norm_series[norm_series > 1.] = 1.
    norm_series[norm_series < 0.] = 0.
    return norm_series


def add_row_to_timegrid_airspeck_plot(time_grid, row_idx, values, pm_col_name, first_day):
    # 10 minute average
    values_indexed = values.set_index('timestamp')
    values_10_avg = values_indexed.resample('10Min').mean()

    # Normalise pm2.5
    norm_pm2_5 = normalise_into_range(values_10_avg[pm_col_name], 0, 150)

    for idx in range(len(values_10_avg)):
        idx_in_row = int((values_10_avg.index[idx] - first_day).total_seconds() / (60. * 10))
        time_grid[row_idx, idx_in_row] = norm_pm2_5[idx]


######################################################################
### Specific Pixelgrams for projects with changes to the standard one
######################################################################

def plot_combined_pixelgram_dublin(subject_id, respeck_data, airspeck_data, exposure_period,
                                   recording_period, plot_filepath=None, overwrite_if_already_exists=False):
    """
    airspeck_data and respeck_data should be minute average pandas dataframes, with the timestamp as index
    """
    if plot_filepath is None:
        plot_filepath = dublin_plots_dir + "{}_combined_pixelgram.png".format(subject_id)

    if os.path.isfile(plot_filepath) and not overwrite_if_already_exists:
        print("Pixelgram already exists. Skipping subject.")
        return

    sns.reset_orig()
    # Hour offset so that the legend can be displayed inside graph. Legend doesn't show at the moment,
    # so set to 0
    hours_offset = 9
    hours, num_days, time_grid = prepare_grid(respeck_data, airspeck_data, hours_offset)

    # Calculate GPS speed
    add_speed_to_gps_data(airspeck_data, 'gpsLatitude', 'gpsLongitude')

    # Normalise data
    norm_GPS = normalise_into_range(airspeck_data['speed'], 0, 20)

    pm_95_percentile = np.nanpercentile(airspeck_data['pm2_5'], 95)
    norm_pm2_5 = normalise_into_range(airspeck_data['pm2_5'], 0, pm_95_percentile)

    norm_act = normalise_into_range(respeck_data['activity_level'], 0, 0.7)
    norm_br = normalise_into_range(respeck_data['breathing_rate'], 10, 40)

    # Create grids
    pm25_grid = np.zeros_like(time_grid, dtype=float)
    is_recording_grid = np.zeros_like(time_grid, dtype=float)
    gps_grid = np.zeros_like(time_grid, dtype=float)
    actlevel_grid = np.zeros_like(time_grid, dtype=float)
    br_grid = np.zeros_like(time_grid, dtype=float)
    acttype_grid = np.zeros_like(time_grid, dtype=int)

    # Fill in RESpeck data
    for idx in range(len(respeck_data['timestamp'])):
        time_diff_minutes = int((respeck_data['timestamp'][idx] -
                                 time_grid[0][0]).total_seconds() / 60.)
        idx2d = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))

        br_grid[idx2d] = norm_br[idx]
        actlevel_grid[idx2d] = norm_act[idx]
        acttype_grid[idx2d] = respeck_data['activity_type'][idx] + 1

    # Same for Airspeck
    for idx in range(len(airspeck_data['timestamp'])):
        time_diff_minutes = int((airspeck_data['timestamp'][idx] -
                                 time_grid[0][0]).total_seconds() / 60.)
        idx2d = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))

        # Does current timestamp lie within exposure period?
        if exposure_period[0] <= airspeck_data['timestamp'].iloc[idx] <= exposure_period[1]:
            is_recording_grid[idx2d] = 2
        else:
            is_recording_grid[idx2d] = 1
        gps_grid[idx2d] = norm_GPS[idx]
        pm25_grid[idx2d] = norm_pm2_5[idx]

    # Mark beginning and end of recording form participant details
    time_diff_minutes = int((recording_period[0] - time_grid[0][0]).total_seconds() / 60.)
    idx2d_start = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))
    time_diff_minutes = int((recording_period[1] - time_grid[0][0]).total_seconds() / 60.)
    idx2d_end = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))
    is_recording_grid[idx2d_start] = 3
    is_recording_grid[idx2d_end] = 3

    # Get airspeck uuid for subject
    uuid = get_uuid_for_subj_id_airspeck_personal(subject_id, airspeck_data.index[0], subject_id[:3])

    fig, axes = plt.subplots(1, 6)
    fig.set_size_inches((30, num_days * 4.5))

    prepare_axes(axes, hours_offset, hours, num_days, time_grid)

    plot_column(axes[0], is_recording_grid, is_recording_present_column=True)
    plot_column(axes[1], br_grid, "Breathing rate (10-30 BrPM)")
    plot_column(axes[2], pm25_grid, "PM 2.5, in ug/m3 (0 - {})".format(pm_95_percentile))
    plot_column(axes[3], acttype_grid, is_activity_type=True)
    plot_column(axes[4], actlevel_grid, "Activity level (0-0.7)")
    plot_column(axes[5], gps_grid, "Speed (0-20 km/h)")

    fig.suptitle("Pixelgram for subject {}. Airspeck UUID: {}".format(subject_id, uuid))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, top=0.88)
    plt.savefig(plot_filepath, dpi=300)
    plt.close(fig)
    plt.show()


# Have a separate column which highlights when the breathing rate goes below 10 BrPM.
# Add colorbars in the top of the column
def plot_respeck_pixelgram_qip(subject_id, respeck_minute, plot_filepath=None):
    if plot_filepath is None:
        plot_filepath = qip_plots_dir + "{}_respeck_pixelgram.png".format(subject_id)

    if respeck_minute is None or len(respeck_minute) == 0:
        print("Skipping Pixelgram as no data was passed")
        return

    sns.reset_orig()
    # Hour offset so that the legend can be displayed inside graph.
    hours_offset = 9
    hours, num_days, time_grid = prepare_grid(respeck_minute, hours_offset=hours_offset)

    actlevel_grid = np.full_like(time_grid, np.nan, dtype=float)
    br_grid = np.full_like(time_grid, np.nan, dtype=float)
    br_grid_critical = np.full_like(time_grid, np.nan, dtype=float)
    acttype_grid = np.full_like(time_grid, np.nan, dtype=float)

    for idx in range(len(respeck_minute)):
        time_diff_minutes = int((respeck_minute.index[idx] - time_grid[0][0]).total_seconds() / 60.)
        idx2d = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))

        br_grid[idx2d] = respeck_minute['breathing_rate'][idx]
        br_grid_critical[idx2d] = int(respeck_minute['breathing_rate'][idx] <= 10)
        actlevel_grid[idx2d] = respeck_minute['activity_level'][idx]
        acttype_grid[idx2d] = respeck_minute['activity_type'][idx] + 1

    # Plot
    fig, axes = plt.subplots(1, 4)
    plot_height = (num_days + hours_offset / 24.) * 4.5
    fig.set_size_inches((20, plot_height))

    prepare_axes(axes, hours_offset, hours, num_days, time_grid)

    br_plot = axes[0].pcolormesh(br_grid, cmap='coolwarm_r', vmin=5, vmax=40)
    axes[0].set_title("Breathing rate (5-40 BrPM)")

    # Add colorbar as floating axis
    standard_plot_height = (3 + hours_offset / 24.) * 4.5

    # Axis shape: bottom left x, bottom left y, width, height. Only bottom left y and height change with plot height
    # The dimensions are relative to plot size, so we first multiple them times the standard plot size (3 days), and
    # then divide by the actual plot height
    cax = fig.add_axes(
        [0.06, 1 - ((1 - 0.89) * standard_plot_height / plot_height), 0.165,
         0.012 * standard_plot_height / plot_height])
    fig.colorbar(br_plot, cax=cax, orientation='horizontal')

    plot_column(axes[1], br_grid_critical, "Minutes with average BR <= 10 marked black")
    plot_column(axes[2], acttype_grid, is_activity_type=True)

    br_plot = axes[3].pcolormesh(actlevel_grid, cmap='Greys', vmin=0, vmax=0.5)
    axes[3].set_title("Level of activity")

    # Add colorbar as floating axis
    cax_act = fig.add_axes(
        [0.835, 1 - ((1 - 0.89) * standard_plot_height / plot_height), 0.12,
         0.012 * standard_plot_height / plot_height])
    colorbar_act = fig.colorbar(br_plot, cax=cax_act, orientation='horizontal')
    colorbar_act.set_ticks([0, 0.5])
    colorbar_act.set_ticklabels(["No activity", "Moderate activity"])

    fig.suptitle('Pixelgram for subject {}'.format(subject_id), fontsize=16,
                 y=1 - 0.03 * standard_plot_height / plot_height)
    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()
    plt.subplots_adjust(top=1 - ((1 - 0.92) * standard_plot_height / plot_height))
    plt.savefig(plot_filepath, dpi=300)


def plot_airspeck_pixelgram_inhale(subject_id, airspeck_data, plot_filepath=None):
    if plot_filepath is None:
        plot_filepath = inhale_plots_dir + "{}_airspeck_pixelgram.png".format(subject_id)
        
    if os.path.isfile(plot_filepath):# and not overwrite_if_already_exists:
        print("Pixelgram already exists. Skipping subject.")
        return

    if airspeck_data is None or len(airspeck_data) == 0:
        print("Skipping Pixelgram as no data was passed")
        return
    
    sns.reset_orig()
    hours, num_days, time_grid = prepare_grid(airspeck_data)

    # Load data into grid
    pm_95_percentile = np.nanpercentile(airspeck_data['pm2_5'], 95)
    pm_max = pm_95_percentile if pm_95_percentile > 50 else 50
    norm_pm2_5 = normalise_into_range(airspeck_data['pm2_5'], 0, pm_max)

    add_speed_to_gps_data(airspeck_data, 'gpsLatitude', 'gpsLongitude')
    norm_GPS = normalise_into_range(airspeck_data['speed'], 0, 20)
    norm_gps_accuracy = normalise_into_range(airspeck_data['gpsAccuracy'], 0, 100)

    pm25_grid = np.zeros_like(time_grid, dtype=float)
    recording_grid = np.zeros_like(time_grid, dtype=float)
    gps_grid = np.zeros_like(time_grid, dtype=float)
    gps_acc_grid = np.zeros_like(time_grid, dtype=float)

    for idx in range(len(airspeck_data['timestamp'])):
        time_diff_minutes = int((airspeck_data['timestamp'][idx] - time_grid[0][0]).total_seconds() / 60.)
        idx2d = (int(time_diff_minutes / 60), int(time_diff_minutes % 60))

        pm25_grid[idx2d] = norm_pm2_5[idx]
        #print(norm_pm2_5[idx])
        if not np.isnan(norm_pm2_5[idx]):
            recording_grid[idx2d] = 1
        gps_grid[idx2d] = norm_GPS[idx]
        gps_acc_grid[idx2d] = norm_gps_accuracy[idx]

    # Plot
    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches((20, num_days * 4.5))

    prepare_axes(axes, 0, hours, num_days, time_grid)

    plot_column(axes[0], recording_grid, "Recording present (dark = yes)")
    plot_column(axes[1], pm25_grid, "PM 2.5, in ug/m3 (0 - {})".format(int(pm_max)))
    plot_column(axes[2], gps_grid, "Speed (0-20 km/h)")
    plot_column(axes[3], gps_acc_grid, "Accuracy of GPS in meters.\nDark = low accuracy = probably inside.")

    plt.subplots_adjust(wspace=0.2)
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)


####################################################
### Download data and plot Pixelgrams
####################################################

def download_data_and_plot_combined_pixelgram(subject_id, timeframe=None, filter_out_not_worn_respeck=True,
                                              overwrite_pixelgram_if_already_exists=False, subject_visit_number=None,
                                              overwrite_data_if_already_exists=False, upload_type='automatic'):
    project_name = get_project_for_subject(subject_id)
    plot_dir = project_mapping[project_name][3]

    if subject_visit_number is None:
        label_files = "{}".format(subject_id)
    else:
        label_files = "{}({})".format(subject_id, subject_visit_number)

    pixelgram_filepath = plot_dir + "{}_combined_pixelgram.png".format(label_files)

    # Check if pixelgram already exists
    if not overwrite_pixelgram_if_already_exists and os.path.isfile(pixelgram_filepath):
        print("Pixelgram for subject {} already exists. Skipping subject.".format(label_files))
        return

    # Download data if not present
    download_respeck_and_personal_airspeck_data(subject_id, upload_type=upload_type, timeframe=timeframe,
                                                overwrite_if_already_exists=overwrite_data_if_already_exists,
                                                subject_visit_number=subject_visit_number)

    # Load data and create plot
    respeck_data = load_respeck_file(subject_id, project_name=project_name, upload_type=upload_type,
                                     subject_visit_number=subject_visit_number,
                                     filter_out_not_worn=filter_out_not_worn_respeck)
    airspeck_data = load_personal_airspeck_file(subject_id, project_name=project_name, upload_type=upload_type,
                                                subject_visit_number=subject_visit_number)

    if len(respeck_data) == 0:
        print("RESpeck data for subject {} empty. Skipping subject.".format(label_files))
        return

    if len(airspeck_data) == 0:
        print("Airspeck data for subject {} empty. Skipping subject.".format(label_files))
        return

    if timeframe is not None:
        tz = timezone(project_mapping[project_name][1])

        if timeframe[0].tzinfo is None:
            start_time = tz.localize(timeframe[0])
            end_time = tz.localize(timeframe[1])
        else:
            start_time = timeframe[0]
            end_time = timeframe[1]

        plot_combined_pixelgram(subject_id, respeck_data[start_time:end_time],
                                airspeck_data[start_time:end_time], pixelgram_filepath,
                                overwrite_if_already_exists=overwrite_pixelgram_if_already_exists,
                                subject_visit_number=subject_visit_number)
    else:
        plot_combined_pixelgram(subject_id, respeck_data, airspeck_data, pixelgram_filepath,
                                overwrite_if_already_exists=overwrite_pixelgram_if_already_exists,
                                subject_visit_number=subject_visit_number)


def download_respeck_data_and_plot_pixelgram(subject_id, project_name=None, upload_type='automatic',
                                             timeframe=None, overwrite_pixelgram_if_already_exists=False,
                                             filter_out_not_worn=True,
                                             overwrite_data_if_already_exists=False,
                                             subject_visit_number=None):
    if project_name is None:
        project_name = get_project_for_subject(subject_id)

    plot_dir = project_mapping[project_name][3]
    label_files = "{}({})".format(subject_id, subject_visit_number)

    pixelgram_filepath = plot_dir + "{}_respeck_pixelgram.png".format(label_files)

    # Check if pixelgram already exists
    if not overwrite_pixelgram_if_already_exists and os.path.isfile(pixelgram_filepath):
        print("Pixelgram for subject {} already exists. Skipping subject.".format(label_files))
        return

    # Download files if they weren't there yet before
    download_respeck_data(subject_id, upload_type=upload_type, timeframe=timeframe,
                          overwrite_if_already_exists=overwrite_data_if_already_exists,
                          subject_visit_number=subject_visit_number)

    respeck_data = load_respeck_file(subject_id, project_name, subject_visit_number=subject_visit_number,
                                     upload_type=upload_type, filter_out_not_worn=filter_out_not_worn)

    if len(respeck_data) == 0:
        print("File for subject {} empty. Skipping subject.".format(subject_id))
        return

    if timeframe is not None:
        tz = timezone(project_mapping[project_name][1])

        if timeframe[0].tzinfo is None:
            start_time = tz.localize(timeframe[0])
            end_time = tz.localize(timeframe[1])
        else:
            start_time = timeframe[0]
            end_time = timeframe[1]
        plot_respeck_pixelgram(subject_id, respeck_data[start_time:end_time], pixelgram_filepath,
                               overwrite_if_already_exists=overwrite_pixelgram_if_already_exists,
                               subject_visit_number=subject_visit_number)
    else:
        plot_respeck_pixelgram(subject_id, respeck_data, pixelgram_filepath,
                               overwrite_if_already_exists=overwrite_pixelgram_if_already_exists,
                               subject_visit_number=subject_visit_number)
