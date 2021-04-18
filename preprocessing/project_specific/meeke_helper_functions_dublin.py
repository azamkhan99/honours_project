'''
These functions are from Meeke Roet's thesis project from 2018.
They are used to generate correlation response graphs for Dublin and don't need to be used for anything else.
'''

import pickle
from math import radians

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split


def create_metrics_df(cols):
    metrics_df = pd.DataFrame(np.zeros(shape = (0, len(cols))), columns = cols)

    return metrics_df


def store_metrics(metrics, model_description, y_test, y_hat_test, y_train=None, y_hat_train=None,
                  model_id=None, num_lags_dict=None, model_dict=None):

    row_to_append = pd.DataFrame(np.ones(shape=(1,metrics.shape[1]))*np.nan, columns = metrics.columns)

    if isinstance(model_id, int):
        row_to_append['model'] = int(model_id)
    elif isinstance(model_id, str):
        row_to_append['model'] = model_id

    row_to_append['description'] = model_description

    if isinstance(num_lags_dict, dict):
        for var, n_lags in num_lags_dict.items():
            row_to_append[var] = int(n_lags)

    row_to_append['R_sq_test'] = np.round(r2_score(y_test, y_hat_test), 3)
    row_to_append['RMSE_test'] = np.round(np.sqrt(mean_squared_error(y_test, y_hat_test)), 3)
    row_to_append['MAE_test'] = np.round(mean_absolute_error(y_test, y_hat_test), 3)

    if (isinstance(y_train, (pd.DataFrame, pd.Series))) & (isinstance(y_hat_train, (pd.DataFrame, pd.Series))):
        row_to_append['R_sq_train'] = np.round(r2_score(y_train, y_hat_train), 3)
        row_to_append['RMSE_train'] = np.round(np.sqrt(mean_squared_error(y_train, y_hat_train)), 3)
        row_to_append['MAE_train'] = np.round(mean_absolute_error(y_train, y_hat_train), 3)

    metrics = metrics.append(row_to_append, ignore_index=True, sort=True)

    if (isinstance(model_dict, dict)) & (isinstance(model_id, int)):
        if isinstance(num_lags_dict, dict):
            model_dict[model_id] = num_lags_dict
        else:
            model_dict[model_id] = model_description

    return metrics


def ridge_with_lags(data, num_lags_dict, minutes_ahead, model_num, metrics_df,
                    model_dict=None, test_split='random', test_size=0.2,
                    random_state=None, external_testing=False, y_var='breathing_rate'):
    '''

    Inputs:
        - data:          Dataframe containing all variables.
        - num_lags_dict: Dictionary specifying the number of lags for breathing rate and activity level.
        - minutes_ahead: Integer specificying how far ahead predictions should be made.
        - model_num:     Model identifier.
        - metrics_df:    Dataframe to append the performance metrics to.
        - model_dict:    Dictionary to add the model settings to.
        - test_split:    'random' for a random train/test split, or 2-element list with
                         the timeframe that should be used as testing data.
        - test_size:     Size of the test set if random test splitting is used.
        - random_state:  Random state seed.

    '''

    import warnings

    print(model_num)

    if not minutes_ahead > 0:
        raise ValueError('Prediction needs to be at least 1 minute ahead.')

    variables = [y_var]

    if y_var in ['breathing_rate', 'activity_level']:
        for i in range(num_lags_dict['br_lags']):
            i += minutes_ahead
            variables.append('br_lag{}'.format(i))
        if i > 120:
            raise ValueError('Not enough breathing rate lags available in the df for the number of lags and minutes ahead specified.')
        for i in range(num_lags_dict['al_lags']):
            i += minutes_ahead
            variables.append('al_lag{}'.format(i))
        if i > 120:
            raise ValueError('Not enough activity level lags available in the df for the number of lags and minutes ahead specified.')

    if y_var in ['br_smoothed', 'al_smoothed']:
        for i in range(num_lags_dict['br_lags']):
            i += minutes_ahead
            variables.append('br_smoothed_lag{}'.format(i))
        if i > 120:
            raise ValueError('Not enough breathing rate lags available in the df for the number of lags and minutes ahead specified.')
        for i in range(num_lags_dict['al_lags']):
            i += minutes_ahead
            variables.append('al_smoothed_lag{}'.format(i))
        if i > 120:
            raise ValueError('Not enough activity level lags available in the df for the number of lags and minutes ahead specified.')

    data_nonan = data.dropna(subset=variables)
    X = data_nonan[variables].copy(deep = True)
    X = X.drop(y_var, axis = 1)
    y = data_nonan[y_var]

    if len(y) < 5000:
        warnings.warn('Less than 5000 usable observations.')

    if external_testing == False:
        if test_split == 'random':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
        elif isinstance(test_split, list):
            X_test = X.between_time(test_split[0], test_split[1])
            X_train = X.between_time(test_split[1], test_split[0], include_start=False, include_end=False)
            y_test = y.between_time(test_split[0], test_split[1])
            y_train = y.between_time(test_split[1], test_split[0], include_start=False, include_end=False)
    else:
        X_train = X
        y_train = y

    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
    alpha_grid = [0.25, 0.5, 0.75, 1.0]

    ridge = RidgeCV(alphas=alpha_grid, cv=kfold, scoring='neg_mean_squared_error')
    ridge.fit(X_train, y_train)

    if (num_lags_dict['br_lags'] == 20) & (num_lags_dict['al_lags'] == 0):
        print(ridge.alpha_)

    if external_testing == False:
        y_hat_test = ridge.predict(X_test)
        y_hat_train = ridge.predict(X_train)

        metrics_df = store_metrics(metrics_df, 'ridge_{}lagsbr_{}lagsal'.format(num_lags_dict['br_lags'], num_lags_dict['al_lags']),
                                   y_test, y_hat_test, y_train, y_hat_train,
                                   model_num, num_lags_dict, model_dict)

        return metrics_df

    else:
        return ridge, X.columns


def save_pickle(object, path):
    with open(path, "wb") as f:
        pickle.dump(object, f)


def load_pickle(path):
    with open(path, "rb") as f:
        object = pickle.load(f)
    return object


def add_outlier_to_list(subj_id, reason, project='apcaps'):
    if project == 'apcaps':
        file = 'C:/Users/ikke_/Documents/Studie/2017-2018-MScOperationalResearch-UniversityOfEdingburgh/Scriptie/data/APCAPS/apcaps_outliers.p'
    else:
        raise ValueError("The project you specified is not implemented yet.")

    try:
        outlier_list = load_pickle(file)

        if reason in outlier_list['reason'].values:
            i = int(np.where(outlier_list['reason'] == reason)[0])

            if not subj_id in outlier_list.loc[i, 'subjects']:
                outlier_list.loc[i, 'subjects'].append(subj_id)
        else:
            outlier_list = pd.concat([outlier_list, pd.DataFrame([[reason, [subj_id]]], columns=['reason', 'subjects'], index=[outlier_list.index.max() + 1])], ignore_index=False)

    except FileNotFoundError:
        outlier_list = pd.DataFrame(np.zeros(shape=(0,2)), columns=['reason', 'subjects'], dtype='object')
        outlier_list = pd.concat([outlier_list, pd.DataFrame([[reason, [subj_id]]], columns=['reason', 'subjects'])])


    save_pickle(outlier_list, file)


def load_outlier_list(only_ids=True, project='apcaps', full_only=True):
    if project == 'apcaps':
        file = 'C:/Users/ikke_/Documents/Studie/2017-2018-MScOperationalResearch-UniversityOfEdingburgh/Scriptie/data/APCAPS/apcaps_outliers.p'
    else:
        raise ValueError("The project you specified is not implemented yet.")

    outlier_list = load_pickle(file)

    if full_only == True:
        outlier_list = outlier_list.loc[np.isin(outlier_list['reason'], ['Defective sensor', 'No data recorded or data lost', 'Sensor worn incorrectly or not consistently (whole recording)'])]

    if only_ids == True:
        lists = outlier_list['subjects'].tolist()
        outlier_list = list(set([item for sublist in lists for item in sublist]))

    return outlier_list


def remove_subj_data_outliers(data, project='apcaps', full_only=True):
     # Load outliers.
    outlier_list = load_outlier_list(only_ids=True, project=project, full_only=full_only)

    # Remove outliers from data dictionary.
    if isinstance(data, dict):
        for outl in outlier_list:
            data.pop(outl, None)

    # Remove outliers from concatenated dataframe.
    if isinstance(data, pd.DataFrame):
        outlier_mask = np.isin(data['subj_id'], outlier_list)
        data = data[~outlier_mask]

    return data


def remove_outliers_fill_nan_and_concat_subj_data(data_dict, project='apcaps', full_only=True, day_times=['08:00', '20:00'], night_times=['00:00', '05:00']):
    """ From a dictionary of subject data, fill certain NaNs and remove outliers.

    Input:
        - data_dict: Dictionary of subject data.
        - project:   Project that the data belongs to.

    Outputs:
        - Data dictionary with some NaNs filled and without outliers.
        - Concatenated subject dataframes with some NaNs filled and without outliers.

    """

    # Fill some of the NaN values in the subject data.
    for subj_id, subj_data in data_dict.items():
        temp_data = subj_data.select_dtypes(['number'])
        temp_data.drop(['activity_type', 'activity_type_extended', 'activity_type_adapted', 'closest_loc', 'night_dummy', 'day_dummy'], axis=1, inplace=True)
        temp_data.fillna(temp_data.rolling(7, center=True, min_periods=2).mean(), inplace=True)
        subj_data.loc[:, temp_data.columns] = temp_data.values

        # Single out night + lying data, and day data.
        night_idx = subj_data[subj_data['activity_type'] == 2].between_time(night_times[0], night_times[1], include_start=True, include_end=False).index
        day_idx = subj_data.between_time(day_times[0], day_times[1], include_start=True, include_end=False).index

        # Remove day index items that are also in the night index (i.e lying down early in the morning, when the time frames overlap).
        day_idx = day_idx.drop(np.intersect1d(night_idx, day_idx))

        # Create dummies indicating night and day.
        subj_data['night_dummy'] = 0
        subj_data['day_dummy'] = 0
        subj_data.loc[night_idx, 'night_dummy'] = 1
        subj_data.loc[day_idx, 'day_dummy'] = 1

    data_dict_outl_rem = remove_subj_data_outliers(data_dict, project=project, full_only=full_only)

    # Aggregate all data and remove outliers.
    all_data = pd.concat(data_dict.values(), ignore_index=False, sort=True)
    all_data_outl_rem = remove_subj_data_outliers(all_data, project=project, full_only=full_only)

    # Sort alldata df by subject and then timestamp.
    all_data_outl_rem['timestamp'] = all_data.index
    all_data_outl_rem.sort_values(['subj_id', 'timestamp'], inplace=True)

    return data_dict_outl_rem, all_data_outl_rem


def remove_outliers_participant_details(participant_details, project='apcaps', save=True, folder=None, full_only=True):
    outlier_list = load_outlier_list(only_ids=True, project=project, full_only=full_only)

    participant_details_outl_rem = participant_details.copy(deep=True)

    for outl in outlier_list:
        try:
            participant_details_outl_rem.drop(outl, inplace=True)
        except KeyError:
            continue

    if save == True:
        # Export participant details without outliers.
        participant_details_outl_rem.to_csv(folder, index=True)

    return participant_details_outl_rem


def regression_participant_details(data, x_variables, y_variable, x_var_names=None, type='linear', constant=True, standardize=True):
    if type == 'linear':
        X = data[x_variables]
        y = data[y_variable]

        if standardize == True:
            # not_dummy = [var for var in x_variables if not 'dummy' in var]
            # if 'Female' in x_variables:
            #     not_dummy.remove('Female')

            if 'Average AL day' in x_variables:
                data['Average AL day'] = data['Average AL day'] / data['Average AL day'].std()
            if 'Average AL preceding day' in x_variables:
                data['Average AL preceding day'] = data['Average AL preceding day'] / data['Average AL preceding day'].std()
            if y_variable == 'Average AL day':
                y = y / y.std()

        if isinstance(x_var_names, list):
            X.columns = x_var_names

        if constant == True:
            model = sm.OLS(y, sm.add_constant(X), missing='drop')
        else:
            model = sm.OLS(y, X, missing='drop')
        results = model.fit()

    elif type == 'logistic':
        variables = x_variables + [y_variable]
        data = data.dropna(subset=variables)
        X = data[[x for x in variables if x != y_variable]]
        y = np.array(data[y_variable].astype(bool))
        if constant == True:
            model = sm.Logit(y, sm.add_constant(X), missing='none')
        else:
            model = sm.Logit(y, X, missing='none')

        results = model.fit()

    return results


def plot_colourline(x, y, c, ax):
    from matplotlib import cm
    c = cm.jet(c/3)
#     c = cm.jet((c-np.min(c)) / (np.max(c)-np.min(c)))
    # ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], c=c[i])
    return


def plot_correlations(data, subject_id='all', include=None, exclude=None, title=False, show=False, save=False, path=None, size=(7,7), mode='average', labels=None):
    """ Plot correlations between variables in a heatmap.

    Inputs:
        - data: Dataframe to be used.
        - subject_id: Subject ID corresponding to the data, or simply all subjects.
        - exclude: List of variables to exclude.
        - title: Whether to show a title or not.
        - show: Whether to show the plot or not.
        - save: Whether to save the plot or not.
        - path: Path to save the plot to, only required if save=True.
        - size: Desired size of the figure.
    """

    # Create plot title.
    if isinstance(subject_id, int):
        plot_name = 'corr_mat_sub{:03d}.png'.format(subject_id)
    else:
        plot_name = 'corr_mat_sub_{}.png'.format(subject_id)

    if mode == 'all_at_once':
        if isinstance(include, list):
            corr_mat = data.loc[:, include].dropna().corr()
        elif isinstance(exclude, list):
            corr_mat = data.drop(exclude, axis=1).corr() # Get correlations excluding specified variables.
        else:
            corr_mat = data.corr()

    elif mode == 'average':
        if isinstance(include, list):
            corr_mat = data.loc[:, include + ['subj_id']].dropna().groupby('subj_id').corr()
            corr_mat.index.rename('var', level=1, inplace=True)
            corr_mat = corr_mat.groupby('var').mean()
            corr_mat = corr_mat.loc[include, include]
        elif isinstance(exclude, list):
            corr_mat = data.drop(exclude, axis=1).dropna().groupby('subj_id').corr()
            corr_mat.index.rename('var', level=1, inplace=True)
            corr_mat = corr_mat.groupby('var').mean()
        else:
            corr_mat = data.dropna().groupby('subj_id').corr()
            corr_mat.index.rename('var', level=1, inplace=True)
            corr_mat = corr_mat.groupby('var').mean()

    corr_mat = rename_variables(corr_mat)

    # Generate a mask for the upper triangle.
    mask = np.zeros_like(corr_mat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    if isinstance(labels, list):
        plot_labels = labels
    else:
        plot_labels = corr_mat.columns

    # Plot correlations as heatmap.
    fig = plt.figure(figsize=size)
    sns.heatmap(corr_mat, xticklabels=corr_mat.columns, yticklabels=corr_mat.columns,
                vmin=0., vmax=1., annot=True, annot_kws={"size": 13}, fmt='.2f', square=True,
                cmap="YlGnBu", mask=mask, cbar_kws={"shrink": .65})

    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=45, ha='right')

    # plt.xticks(rotation=0)
    #
    # locs, labels = plt.xticks()
    # for i, label in enumerate(labels):
    #     label.set_y(label.get_position()[1] - (i % 2) * 0.075)

    if title == True:
        plt.title('Correlation between numeric variables for subject {}'.format(subject_id))

    if save == True:
        if path == None:
            raise ValueError("A path needs to be specified to save the figure to.")
        else:
            plt.tight_layout()
            plt.savefig(path, dpi=300)

    if show == True:
        plt.show()
    else:
        plt.close(fig)


def plot_breathing_rate(subj_id, subj_data, interval, variable='breathing_rate', show=True, ax_given=None, save=False, path=None):
    """ Plot the breathing rate of subject *subj_id* between the interval specified.
    """

    # Interpolate missing values.
    subj_data = subj_data.interpolate(method='linear', axis=0)

    if interval[0] == 'start':
        start = 0
    else:
        start = [i for i, x in enumerate(subj_data.index == interval[0]) if x][0]
    if interval[1] == 'end':
        end = subj_data.shape[0]
    else:
        end = [i for i, x in enumerate(subj_data.index == interval[1]) if x][0]

    plot_data = subj_data.iloc[start:end, :]

    fig, ax = plt.subplots(figsize=(10,5))

    ax = ax_given or plt.gca()

    if ax_given is not None:
        predefined_ax = True
    else:
        predefined_ax = False

    color_data = plot_data['activity_type'].replace(-1, 3)

    # Plot data coloured by activity type.
    # plot_colourline(plot_data.index.values, plot_data[variable].values, plot_data['activity_type'], ax=ax)
    plot_colourline(plot_data.index.values, plot_data[variable].values, color_data, ax=ax)

    if variable == 'breathing_rate':
        # Plot standard error of the breathing rate.
        ax.fill_between(plot_data.index.values,
                        plot_data['breathing_rate'] + plot_data['sd_br'],
                        plot_data['breathing_rate'] - plot_data['sd_br'],
                        alpha=0.2)

    # Format axes.
    if predefined_ax == False:
        xformatter = md.DateFormatter('%H:%M')
        xlocator = md.MinuteLocator(byminute=[0, 15, 30, 45], interval=1)
        ax.xaxis.set_major_locator(xlocator)
        plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
        fig.autofmt_xdate()
        plt.xlabel('Time')

    if variable == 'breathing_rate':
        plt.ylabel('Breathing rate per minute')
    elif variable == 'smoothed_br':
        plt.ylabel('Smoothed breathing rate')
    elif variable == 'sd_br':
        plt.ylabel('Standard deviation of breathing rate')
    elif variable == 'rolling_sd_br':
        plt.ylabel('Rolling mean st. dev. of breathing rate')
    elif variable == 'activity_level':
        plt.ylabel('Activity level per minute')
    elif variable == 'smoothed_al':
        plt.ylabel('Smoothed activity level')

    # Plot legend.
    custom_lines = [Line2D([0], [0], color=cm.jet(0.), lw = 2),
                    Line2D([0], [0], color=cm.jet(1/3), lw = 2),
                    Line2D([0], [0], color=cm.jet(2/3), lw = 2),
                    Line2D([0], [0], color=cm.jet(1.), lw = 2)]
    lgd = ax.legend(custom_lines, ['Sitting/standing', 'Walking', 'Lying down', "Wrong orientation/undetermined"], loc='center left', bbox_to_anchor=(1.01, 0.5))

    if save == True:
        if path == None:
            raise ValueError("A path needs to be specified to save the figure to.")
        else:
            plt.savefig(path, bbox_extra_artists = (lgd,), bbox_inches = 'tight', dpi = 300)

    if show == True:
        plt.show()




def element_wise_haversine(lats1, longs1, lats2, longs2):
    """ Calculate element-wise distance between two (equally long) lists of GPS locations.

    Inputs:
        - lats1:  Latitude coordinates of first list of locations.
        - longs1: Longitude coordinates of first list of locations.
        - lats2:  Latitude coordinates of second list of locations.
        - longs2: Longitude coordinates of second list of locations.

    Outputs:
        - Array of element-wise distances in meters (1 x length of input vectors).
    """

    lats1 = np.array(lats1)
    longs1 = np.array(longs1)
    lats2 = np.array(lats2)
    longs2 = np.array(longs2)

     # Convert longitudes and latitudes to radians:
    rad_lats1 = np.array([radians(x) for x in lats1])
    rad_longs1 = np.array([radians(x) for x in longs1])
    rad_lats2 = np.array([radians(x) for x in lats2])
    rad_longs2 = np.array([radians(x) for x in longs2])

    # Calculate differences between longs/lats.
    lat_dists = rad_lats1 - rad_lats2
    long_dists = rad_longs1 - rad_longs2

    # Haversine formula step 1: a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2.
    a = np.square(np.sin(lat_dists/2)) + np.multiply(
                                            np.multiply(np.cos(lats1), np.cos(lats2)),
                                            np.square(np.sin(long_dists/2))
                                         )

    # Haversine formula step 2: c = 2 * asin(sqrt(a)).
    c = 2 * np.arcsin(np.sqrt(a))

    # Haversine formula step 3: 6371 * c.
    r = 6371 # Radius of earth in kilometers.
    distances = c * r

    return distances*1000



def rename_variables(data):
    rename_dict = {'age': 'Age',
                   'blood_pressure': 'Blood pressure',
                   'height(cm)': 'Height (cm)',
                   'pulse': 'Pulse',
                   'sex': 'Sex',
                   'weight(kg)': 'Weight (kg)',
                   'n_days': 'Days recorded',
                   'n_nights': 'Nights recorded',
                   'F': 'Female',
                   'M': 'Male',
                   'av_br': 'Average RR (BPM)',
                   'av_br_night': 'Average RR night (BPM)',
                   'av_br_day': 'Average RR day (BPM)',
                   'av_br_day_preceding_day': 'Average RR preceding day (BPM)',
                   'av_al': 'Average AL',
                   'av_al_night': 'Average AL night',
                   'av_al_day': 'Average AL day',
                   'av_al_day_preceding_day': 'Average AL preceding day',
                   'av_steps': 'Average step count (steps/min)',
                   'av_steps_night': 'Average step count night (steps/min)',
                   'av_steps_day': 'Average step count day (steps/min)',
                   'av_steps_day_preceding_day': 'Average step count preceding day (steps/min)',
                   'std_br': 'Std. RR (BPM)',
                   'std_br_night': 'Std. RR night (BPM)',
                   'std_br_day': 'Std. RR day (BPM)',
                   'std_br_day_preceding_day': 'Std. RR preceding day (BPM)',
                   'std_al': 'Std. AL',
                   'std_al_night': 'Std. AL night',
                   'std_al_day': 'Std. AL day',
                   'std_al_day_preceding_day': 'Std. AL preceding day',
                   'fraction_walking': 'Walking time fraction total',
                   'fraction_walking_day': 'Walking time fraction',
                   'fraction_walking_day_preceding_day': 'Walking time fraction preceding day',
                   'fraction_lying': 'Lying time fraction total',
                   'fraction_lying_day': 'Lying time fraction',
                   'fraction_lying_day_preceding_day': 'Lying time fraction preceding day',
                   'av_n_sleep_interruptions': 'Sleep interruptions',
                   'fraction_sleep_interruptions': 'WASO',
                   'av_n_turns': 'Turns',
                   'dummy_napping': 'Napping dummy',
                   'dummy_napping_preceding_day': 'Napping dummy preceding day',
                   'woke_up': 'Awakening dummy',
                   'mins_in_shops_for_food/tobacco/alcohol': 'Minutes in shops',
                   'perc_in_shops_for_food/tobacco/alcohol': 'Shops time fraction',
                   'dummy_shops_for_food/tobacco/alcohol': 'Shops dummy',
                   'mins_in_physical_activity_site': 'Minutes activity',
                   'perc_in_physical_activity_site': 'Activity time fraction',
                   'dummy_physical_activity_site': 'Activity dummy',
                   'dummy_physical_activity_site_preceding_day': 'Activity dummy preceding day',
                   'mins_in_health_service': 'Minutes in health service',
                   'perc_in_health_service': 'Health service time fraction',
                   'dummy_health_service': 'Health dummy',
                   'dummy_health_service_preceding_day': 'Health dummy preceding day',
                   'mins_in_education_service': 'Minutes in education',
                   'perc_in_education_service': 'Education time fraction',
                   'dummy_education_service': 'Education dummy',
                   'mins_in_alcohol_shop': 'Minutes in alcohol shop',
                   'perc_in_alcohol_shop': 'Alcohol shop time fraction',
                   'dummy_alcohol_shop': 'Alcohol dummy',
                   'dummy_alcohol_shop_preceding_day': 'Alcohol dummy preceding day',
                   'mins_in_gym': 'Minutes in gym',
                   'perc_in_gym': 'Gym time fraction',
                   'dummy_gym': 'Gym dummy',
                   'mins_in_swimming': 'Minutes in swimming',
                   'perc_in_swimming': 'Swimming time fraction',
                   'dummy_swimming': 'Swimming dummy',
                   'mins_in_playground/sports': 'Minutes in sports site',
                   'perc_in_playground/sports': 'Sports site time fraction',
                   'dummy_playground/sports': 'Sports site dummy',
                   'mins_sports': 'Minutes doing sports',
                   'dummy_sports': 'Sports dummy',
                   'perc_laboured_breathing': 'Laboured breathing time fraction',
                   'boolean_laboured_breathing': 'Laboured breathing dummy',
                   'av_walking_distance': 'Average walking distance total (m/min)',
                   'av_overall_distance': 'Average overall distance total (m/min)',
                   'av_walking_distance_day': 'Average walking distance (m/min)',
                   'av_walking_distance_day_preceding_day': 'Average walking distance preceding day (m/min)',
                   'bmi': 'BMI',
                   'GPS_accuracy': 'GPS accuracy (m)',
                   'activity_level': 'Activity level',
                   'activity_type': 'Activity type (grouped)',
                   'activity_type_adapted': 'Activity type adapted',
                   'activity_type_extended': 'Activity type',
                   'al_ahead1': 'AL lead 1',
                   'al_ahead10': 'AL lead 10',
                   'al_ahead2': 'AL lead 2',
                   'al_ahead3': 'AL lead 3',
                   'al_ahead4': 'AL lead 4',
                   'al_ahead5': 'AL lead 5',
                   'al_ahead6': 'AL lead 6',
                   'al_ahead7': 'AL lead 7',
                   'al_ahead8': 'AL lead 8',
                   'al_ahead9': 'AL lead 9',
                   'al_lag1': 'AL lag 1',
                   'al_lag10': 'AL lag 10',
                   'al_lag11': 'AL lag 11',
                   'al_lag12': 'AL lag 12',
                   'al_lag13': 'AL lag 13',
                   'al_lag14': 'AL lag 14',
                   'al_lag15': 'AL lag 15',
                   'al_lag16': 'AL lag 16',
                   'al_lag17': 'AL lag 17',
                   'al_lag18': 'AL lag 18',
                   'al_lag19': 'AL lag 19',
                   'al_lag2': 'AL lag 2',
                   'al_lag20': 'AL lag 20',
                   'al_lag21': 'AL lag 21',
                   'al_lag22': 'AL lag 22',
                   'al_lag23': 'AL lag 23',
                   'al_lag24': 'AL lag 24',
                   'al_lag25': 'AL lag 25',
                   'al_lag26': 'AL lag 26',
                   'al_lag27': 'AL lag 27',
                   'al_lag28': 'AL lag 28',
                   'al_lag29': 'AL lag 29',
                   'al_lag3': 'AL lag 3',
                   'al_lag30': 'AL lag 30',
                   'al_lag4': 'AL lag 4',
                   'al_lag5': 'AL lag 5',
                   'al_lag6': 'AL lag 6',
                   'al_lag7': 'AL lag 7',
                   'al_lag8': 'AL lag 8',
                   'al_lag9': 'AL lag 9',
                   'alt': 'Altitude',
                   'br_ahead1': 'RR lead 1',
                   'br_ahead10': 'RR lead 10',
                   'br_ahead2': 'RR lead 2',
                   'br_ahead3': 'RR lead 3',
                   'br_ahead4': 'RR lead 4',
                   'br_ahead5': 'RR lead 5',
                   'br_ahead6': 'RR lead 6',
                   'br_ahead7': 'RR lead 7',
                   'br_ahead8': 'RR lead 8',
                   'br_ahead9': 'RR lead 9',
                   'br_lag1': 'RR lag 1',
                   'br_lag10': 'RR lag 10',
                   'br_lag11': 'RR lag 11',
                   'br_lag12': 'RR lag 12',
                   'br_lag13': 'RR lag 13',
                   'br_lag14': 'RR lag 14',
                   'br_lag15': 'RR lag 15',
                   'br_lag16': 'RR lag 16',
                   'br_lag17': 'RR lag 17',
                   'br_lag18': 'RR lag 18',
                   'br_lag19': 'RR lag 19',
                   'br_lag2': 'RR lag 2',
                   'br_lag20': 'RR lag 20',
                   'br_lag21': 'RR lag 21',
                   'br_lag22': 'RR lag 22',
                   'br_lag23': 'RR lag 23',
                   'br_lag24': 'RR lag 24',
                   'br_lag25': 'RR lag 25',
                   'br_lag26': 'RR lag 26',
                   'br_lag27': 'RR lag 27',
                   'br_lag28': 'RR lag 28',
                   'br_lag29': 'RR lag 29',
                   'br_lag3': 'RR lag 3',
                   'br_lag30': 'RR lag 30',
                   'br_lag4': 'RR lag 4',
                   'br_lag5': 'RR lag 5',
                   'br_lag6': 'RR lag 6',
                   'br_lag7': 'RR lag 7',
                   'br_lag8': 'RR lag 8',
                   'br_lag9': 'RR lag 9',
                   'breathing_rate': 'RR (BPM)',
                   'closest_loc': 'Closest location',
                   'convoluted_mean_br': 'Convoluted average RR (BPM)',
                   'convoluted_median_br': 'Convoluted median RR (BPM)',
                   'day_dummy': 'Day dummy',
                   'dummy_turn': 'Turning dummy',
                   'laboured_breathing': 'Laboured breathing dummy',
                   'lat': 'Latitude',
                   'loc_type': 'Location category',
                   'long': 'Longitude',
                   'movement': 'Distance moved (m)',
                   'night_dummy': 'Night dummy',
                   'sd_br': 'Std. RR (BPM)',
                   'step_count': 'Step count'}

    return data.rename(columns=rename_dict)



def create_smoothed_vars(data_dict, all_data, variables=['breathing_rate', 'activity_level'], n_smooth=15, n_lags=30):

    if 'breathing_rate' in variables:
        all_data['br_smoothed'] = all_data.groupby('subj_id')['breathing_rate'].rolling(n_smooth).mean().values
        for t in range(n_lags):
            t += 1
            all_data['br_smoothed_lag{}'.format(t)] = all_data.groupby('subj_id')['br_smoothed'].shift(t).values

            # Create smoothed breathing rate first difference.
            all_data['br_lag{}_deriv1'.format(t)] = pd.DataFrame((all_data.groupby('subj_id')['br_smoothed_lag{}'.format(t)].shift(0) - all_data.groupby('subj_id')['br_smoothed_lag{}'.format(t)].shift(1)).values, index=all_data.index)

            # Create smoothed breathing rate second difference.
            first_diff = pd.DataFrame(all_data['br_lag{}_deriv1'.format(t)])
            first_diff['subj_id'] = all_data['subj_id']
            all_data['br_lag{}_deriv2'.format(t)] = pd.DataFrame((first_diff.groupby('subj_id').shift(0) - first_diff.groupby('subj_id').shift(1)).values, index=all_data.index)

        for subj_id, subj_data in data_dict.items():
            subj_data['br_smoothed'] = subj_data['breathing_rate'].rolling(n_smooth).mean()
            for t in range(n_lags):
                t += 1
                subj_data['br_smoothed_lag{}'.format(t)] = subj_data['br_smoothed'].shift(t).values



    if 'activity_level' in variables:
        all_data['al_smoothed'] = all_data.groupby('subj_id')['activity_level'].rolling(n_smooth).mean().values
        for t in range(n_lags):
            t += 1
            all_data['al_smoothed_lag{}'.format(t)] = all_data.groupby('subj_id')['al_smoothed'].shift(t).values

        for subj_id, subj_data in data_dict.items():
            subj_data['al_smoothed'] = subj_data['activity_level'].rolling(n_smooth).mean()
            for t in range(n_lags):
                t += 1
                subj_data['al_smoothed_lag{}'.format(t)] = subj_data['al_smoothed'].shift(t).values

    return data_dict, all_data
