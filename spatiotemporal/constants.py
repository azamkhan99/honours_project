CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

# Format: name/abbreviation: bucket name, timezone offset, data directory
# These will need to be adjusted for your local file system paths
dublin_data_dir = '/Users/zoepetard/Documents/Speckled/projects/dublin/data/'
dublin_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/dublin/plots/'
dublin_participant_details_dir = "/Users/zoepetard/Documents/Speckled/projects/dublin/participant_details/"
# This can be found in the Dublin repository
dublin_timezones_correction_filepath = "/Users/zoepetard/Documents/Speckled/projects/dublin/participant_details/" \
                                       "Dublin timezone corrections.xlsx"

daphne_data_dir = '/Users/azamkhan/Speckled/Daphne/data/Daphne subject data/'
daphne_metadata_dir = '/Users/azamkhan/Speckled/Daphne/data/'
daphne_plots_dir = '/Users/azamkhan/Speckled/Daphne/plots/'
# This can be found in the Daphne repository
daphne_logs_filepath = "/Users/azamkhan/speckled/Daphne/Daphne device logging Main Study.xlsx"
daphne_questionnaire_database_aap_filepath = '/Users/zoepetard/Documents/Speckled/projects/daphne/Daphne documents Github folder/Deployment/Questionnaire and Biomarker results/AAP/DAPHNE_AAP_DATABASE_ALL CRF_18.02.2019_NP.xlsx'
daphne_questionnaire_database_mc_filepath = '/Users/zoepetard/Documents/Speckled/projects/daphne/Daphne documents Github folder/Deployment/Questionnaire and Biomarker results/MC/DAPHNE_MC_DATABASE_ALL CRFs_18.02.2019.xlsx'

apcaps_data_dir = '/Users/zoepetard/Documents/Speckled/projects/APCaPS/data/'
apcaps_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/APCaPS/figures/'

#Change below to paths to peeps files

peeps_metadata_dir = '/Users/azamkhan/speckled/Peeps/data/'
peeps_data_dir = '/Users/azamkhan/speckled/Peeps/data/Subject data/'
peeps_plots_dir = '/Users/azamkhan/speckled/Peeps/plots/'
peeps_reports_dir = '/Users/azamkhan/speckled/Peeps/reports/'
peeps_participant_details_filepath = "/Users/azamkhan/speckled/Peeps/PEEPS device logging sheet.xls"

specknet_data_dir = '/Users/zoepetard/Documents/Speckled/projects/General Specknet/data/'
specknet_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/General Specknet/plots/'

cundall_data_dir = '/Users/zoepetard/Documents/Speckled/projects/cundall/data/'
cundall_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/cundall/plots/'

inhale_data_dir = '/Users/azamkhan/speckled/london/static_data/'
inhale_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/inhale/plots/'

philap_data_dir = '/Users/azamkhan/speckled/philap/data/'
philap_metadata_dir = '/Users/azamkhan/speckled/philap/'
philap_plots_dir = '/Users/azamkhan/speckled/philap/plots/'
philap_reports_dir = '/Users/azamkhan/speckled/philap/reports/'
philap_participant_details_filepath = "/Users/azamkhan/speckled/philap/Philap device logging.xlsx"

qip_data_dir = '/Users/zoepetard/Documents/Speckled/projects/QIP/data/'
qip_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/QIP/plots/'

#Change below to paths to Leon files

leon_data_dir = '/Users/azamkhan/speckled/Mexico/Calibrated Data/leon/'
leon_uncalibrated = '/Users/azamkhan/speckled/Mexico/static_uncalibrated/leon/'
leon_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/leon/plots/'
leon_maps_dir = '/Users/zoepetard/Documents/Speckled/projects/leon/images/maps/'
leon_logs_filepath = '/Users/azamkhan/speckled/Mexico/Logging Sheet.xlsx'

#Change below to paths to Guadalajara files

gdl_data_dir = '/Users/azamkhan/speckled/Mexico/Calibrated Data/gdl/'
gdl_uncalibrated = '/Users/azamkhan/speckled/Mexico/static_uncalibrated/gdl/'
gdl_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/leon/plots/'
gdl_maps_dir = '/Users/zoepetard/Documents/Speckled/projects/leon/images/maps/'
gdl_logs_filepath = '/Users/azamkhan/speckled/Mexico/Logging Sheet.xlsx'

bhf_data_dir = '/Users/zoepetard/Documents/Speckled/projects/british-heart/data/'
bhf_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/british-heart/plots/'
bhf_reports_dir = '/Users/zoepetard/Documents/Speckled/projects/british-heart/reports/'
bhf_participant_details_filepath = "/Users/zoepetard/Documents/Speckled/projects/british-heart/February Recess Logging Sheet.xlsx"

windmill_data_dir = '/Users/zoepetard/Documents/Speckled/projects/windmill/data/'
windmill_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/windmill/plots/'

general_data_dir = '/Users/zoepetard/Documents/Speckled/projects/general/data/'
general_plots_dir = '/Users/zoepetard/Documents/Speckled/projects/general/plots/'


# Mapping from project name to Google storage bucket name, timezone name, data directory, plots directory
# Each project is stored with the actual name and the project ID, i.e. the first two characters of the subject ID
project_mapping = {
    # 'dublin': ('dublin-data', 'Europe/Dublin', dublin_data_dir, dublin_plots_dir),
    # As the Dublin project runs through multiple timezones, create a mapping for each deployment
    'DBI': ('dublin-data', 'Europe/Dublin', dublin_data_dir, dublin_plots_dir),  # Phones were running Dublin time
    # Phones were running in Dublin time some of the time. Use correction factors!
    'DBC': ('dublin-data', 'Europe/Prague', dublin_data_dir, dublin_plots_dir),
    'DBS': ('dublin-data', 'Europe/Dublin', dublin_data_dir, dublin_plots_dir),  # Phones were running Dublin time
    'daphne': ('daphne-data', 'Asia/Kolkata', daphne_data_dir, daphne_plots_dir),
    'DC': ('daphne-data', 'Asia/Kolkata', daphne_data_dir, daphne_plots_dir),
    'DM': ('daphne-data', 'Asia/Kolkata', daphne_data_dir, daphne_plots_dir),
    'DMC': ('daphne-data', 'Asia/Kolkata', daphne_data_dir, daphne_plots_dir),
    'DA': ('daphne-data', 'Asia/Kolkata', daphne_data_dir, daphne_plots_dir),
    'DAP': ('daphne-data', 'Asia/Kolkata', daphne_data_dir, daphne_plots_dir),
    'DX': ('daphne-data', 'Asia/Kolkata', daphne_data_dir, daphne_plots_dir),
    'apcaps': ('apcaps-data', 'Asia/Kolkata', apcaps_data_dir, apcaps_plots_dir),
    'AC': ('apcaps-data', 'Asia/Kolkata', apcaps_data_dir, apcaps_plots_dir),
    'peeps': ('peeps-data', 'Asia/Kolkata',  peeps_data_dir, peeps_plots_dir),
    'PE': ('peeps-data', 'Asia/Kolkata', peeps_data_dir, peeps_plots_dir),
    'XX': ('specknet-test', 'Europe/London', specknet_data_dir, specknet_plots_dir),
    'specknet': ('specknet-test', 'Europe/London', specknet_data_dir, specknet_plots_dir),
    'inhale': ('inhale-data', 'Europe/London', inhale_data_dir, inhale_plots_dir),
    'LIX': ('inhale-data', 'Europe/London', inhale_data_dir, inhale_plots_dir),
    'PH': ('philap-data', 'Asia/Kolkata', philap_data_dir, philap_plots_dir),
    'philap': ('philap-data', 'Asia/Kolkata', philap_data_dir, philap_plots_dir),
    'QI': ('qip-data', 'Europe/London', qip_data_dir, qip_plots_dir),
    'qip': ('qip-data', 'Europe/London', qip_data_dir, qip_plots_dir),
    'CU': (None, 'Europe/London', cundall_data_dir, cundall_plots_dir),
    'cundall': (None, 'Europe/London', cundall_data_dir, cundall_plots_dir),
    'MX': ('guadalajara-data', 'America/Mexico_City', gdl_data_dir, gdl_plots_dir),
    'leon': ('leon-data', 'America/Mexico_City', leon_data_dir, leon_plots_dir),
    'leon_uncalibrated': ('leon-data', 'America/Mexico_City', leon_uncalibrated, leon_plots_dir),
    'guadalajara_uncalibrated': ('guadalajara-data', 'America/Mexico_City', gdl_uncalibrated, gdl_plots_dir),
    'british-heart': ('british-heart-data', 'Europe/London', bhf_data_dir, bhf_plots_dir),
    'BH': ('british-heart-data', 'Europe/London', bhf_data_dir, bhf_plots_dir),
    'WI': ('windmill-data', 'Europe/Amsterdam', windmill_data_dir, windmill_plots_dir),
    'windmill': ('windmill-data', 'Europe/Amsterdam', windmill_data_dir, windmill_plots_dir)}

activity_colors = ["#ffffff", '#377eb8', "#ffb400", '#4daf4a', "#000000", "#275a82",
                   "#8bcbfe", "#79c677", "#327230", "#628c61", "#e3ff00"]

# No data is index -1, movement 9
activity_name = ["No data", "Sitting straight/standing", "Walking", "Lying down normal on back",
                 "Worn incorrectly", "Sitting bent forward", "Sitting bent backward",
                 "Lying down to the right", "Lying down to the left", "Lying down on stomach", "Movement"]

static_activities = [0, 2, 4, 5, 6, 7, 8]
sitting_activities = [0, 4, 5]
lying_activities = [2, 6, 7, 8]

particle_sizes_per_bin = ["0.38-0.54", "0.54-0.78", "0.78-1.0", "1.0-1.3", "1.3-1.6", "1.6-2.1", "2.1-3.0",
                          "3.0-4.0", "4.0-5.0", "5.0-6.5", "6.5-8.0", "8.0-10.0", "10.0-12.0", "12.0-14.0",
                          "14.0-16.0", "16.0-17.0"]

static_sensor_ids_daphne = ["0709495D1E817C9C", "D34EDAEB5B564570", "64208C63C6C702A5", "C2CE86F150B73489",
                            "212B4F030E110F1B", "EB35F0770283A33A", "7E81A98CC79E85D9", "E864778321F55A8F",
                            "595DA699CD9937A4", "3226265E938C7D1D", "7101DC0EB64C1984", "557C24E84D0675A1"]

calibration_columns_pm_old = [u'pm1', u'pm2_5', u'pm10', u'temperature', u'humidity', u'bin0',
                  u'bin1', u'bin2', u'bin3', u'bin4', u'bin5', u'bin6', u'bin7', u'bin8',
                  u'bin9', u'bin10', u'bin11', u'bin12', u'bin13', u'bin14', u'bin15']

#The first column should always be the measurement to calibrate
calibration_columns_pm = ['pm2_5', 'pm1','pm10', 'temperature', 'humidity']
calibration_columns_pm_H = ['pm2_5', 'humidity']

calibration_columns_pm10 = ['pm10', 'pm1','pm2_5', 'temperature', 'humidity']

calibration_columns_ox = ['ox_we', 'ox_ae', 'no2_we', 'no2_ae']#, 'temperature', 'humidity']

calibration_columns_no2 = ['no2_we', 'no2_ae', 'ox_we', 'ox_ae']#,'pm1','pm2_5','pm10', 'temperature', 'humidity']
