import pandas as pd
import numpy as np
import datetime as dt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import logging
import json

# data directory containing the raw NMIR files
NMIR_DATA_DIR = "data/NMIR"

# savepath for training job outputs
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Import the database for AIRACs
AIRAC_DF = pd.read_csv('data/AIRAC_dates.csv')
AIRAC_DF['start_date'] = pd.to_datetime(AIRAC_DF['start_date'])
AIRAC_DF['end_date'] = pd.to_datetime(AIRAC_DF['end_date'])

# Import the delay categorization for evaluation
DELAY_CATG_DF = pd.read_csv('data/delay_categorization.csv')

# Define all possible Regulation types
REGULATION_TYPES = ['C - ATC Capacity', 'W - Weather', 'S - ATC Staffing',
                     'G - Aerodrome Capacity', 'I - ATC Ind Action',
                     'M - Airspace Management', 'O - Other', 'P - Special Event',
                     'T - ATC Equipment', 'V - Environmental Issues',
                     'E - Aerodrome Services', 'R - ATC Routeings',
                     'A - Accident/Incident', 'N - Ind Action non-ATC']

def format_raw_NMIR_df(df):
    '''Function to format the raw NMIR dataframe and keep only the important columns.
    
        Args:
            df (DataFrame): raw NMIR dataframe
        
        Returns:
            df (DataFrame): formatted NMIR dataframe
    '''
    df['ACC'] = df['TVS Id'].str[0:4] #First four letters of 'TVS Id' is considered as ACC 
    df['Regulation Start Time'] = pd.to_datetime(df['Regulation Start Time'])
    df['Date'] = df['Regulation Start Time'].dt.date
    df['Datetime'] = pd.to_datetime(df['Date'])
    df['Regulation Activation Date'] = pd.to_datetime(df['Regulation Activation Date'])
    df = df.loc[df['ATFM Delay (min)']>0].reset_index(drop=True)

    columns_to_retain = ['ACC', 'Date', 'Datetime', 'Regulation Start Time',
                         'Regulation End Date', 'Regulation Activation Date',
                         'Regulation Duration (min)', 'Regulation Cancel Status',
                         'Regulation Cancel Date', 'Regulation Reason Name',
                         'ATFM Delay (min)', 'MP Delayed Traffic']
    df = df[columns_to_retain]                     
    return df

def get_airac(date):
    '''Function to get the airac cycle (1 to 13) for any date.
    
        Args:
            date (datetime object): the date for which the AIRAC cycle is required.
    
        Returns:
            airac (int): AIRAC (1 to 13) for the date.
    '''
    airac = AIRAC_DF.loc[(AIRAC_DF['start_date']<=date) & (AIRAC_DF['end_date']>date)]['AIRAC_cycle'].iloc[0]
    return airac

def get_regulation_count(day_df):
    '''Function to count the number of regulations for each type of regulation for a day.
    
        Args:
            day_df (pandas df): The dataframe from which the counts are to be made.
        
        Returns:
            reg_counts_list (list): count of each type of regulations for a day as a list
    '''
    reg_counts_list = []
    for rt in REGULATION_TYPES:
        try:
            reg_count = day_df['Regulation Reason Name'].value_counts()[rt]
        except KeyError:
            reg_count = 0
        reg_counts_list.append(reg_count)
    return reg_counts_list

def build_basic_features(day_df):
    '''Function to build a feature list for a day from the NMIR dataframe list of regulations for a day
    
        Args:
            day_df (pandas df): the dataframe containing the list of regulations (cut off by activation time ex: 6AM) for a day.
    
        Returns:
            features (list): the features for a day. 
    '''
    datetime_0hrs = day_df.iloc[0]['Datetime']
    count_reg_pub = day_df.shape[0]
    avg_reg_dur_pub = day_df['Regulation Duration (min)'].mean()
    d_op_activation_counts = day_df.loc[day_df['Regulation Activation Date']>datetime_0hrs].shape[0]
    count_num_ACC_pub = len(day_df['ACC'].unique().tolist())
    weekday = day_df.loc[0]['Datetime'].dayofweek
    airac = get_airac(day_df.loc[0]['Datetime'])
    reg_counts_list = get_regulation_count(day_df)
    features = [count_reg_pub, avg_reg_dur_pub, d_op_activation_counts, count_num_ACC_pub, weekday, airac] + reg_counts_list
    return features

def build_labels(day_df):
    '''Function to build the labels for a day from the NMIR dataframe list of regulations for a day.
    
        Args:
            day_df (pandas df): The dataframe containing the list of regulations (cut off by dayEndHrs ex: 24 represnting end of the day) for a day.
    
        Returns:
            labels (list): the labels ['ATFM Delay (min)', 'MP Delayed Traffic'] for a day.
    '''
    atfm_delay = day_df['ATFM Delay (min)'].sum()
    mp_delyed_traffic = day_df['MP Delayed Traffic'].sum()
    labels = [atfm_delay, mp_delyed_traffic]
    return labels

def transform_to_daywise_basic(raw_df, pub_cut_off_hrs=6, day_end_hrs=24, encode=False):
    '''Function to transform raw NMIR dataframe into a daywise dataframe with features and labels.
    
        Args:
            raw_df (pandas df): The raw NMIR dataframe.
        
            pubCutOffHrs (int): This number represents the hours (0 to 24) which is used to seperate a days'
                list of regulations as a snapshot to build the features. This is based on the 'activation time' column.
        
            dayEndHrs (int): This number represents the target time hour (0 to 24) which is used as target delays as labels for the day.
        
        Returns:
            daywise_df: A daywise dataframe with features and lables
    '''
    raw_df = raw_df.reset_index(drop=True)
    raw_df = format_raw_NMIR_df(raw_df)
    dates = raw_df.groupby(by ='Date',as_index=False).sum()['Date'].tolist() # get a list of all available dates in the df
    
    daywise_rowdata_list = []
    for d in dates:
        day_df = raw_df.loc[raw_df['Date']==d].reset_index(drop=True)
        day_begin_time = pd.to_datetime(d) # convert date to datetime to get 00:00 hrs time-stamp
        reg_cut_off_time = day_begin_time + dt.timedelta(hours=pub_cut_off_hrs)
        day_end_time = day_begin_time + dt.timedelta(hours=24)
        
        # select only regulations activated from the start of the day until the cutoff time
        day_df = day_df.loc[day_df['Regulation Activation Date']<=reg_cut_off_time].reset_index(drop=True)
        
        # build features from the filtered daily regulations list
        day_features = build_basic_features(day_df)
        day_labels = build_labels(day_df.loc[day_df['Regulation Start Time']<=day_end_time])
        daywise_rowdata_list.append([d] + day_features + day_labels)
    header = ['Date', 'CountRegPub', 'AvgRegDurPub', 'DopActivationCounts','CountNumACCPub',
              'WeekDay', 'AIRAC'] + REGULATION_TYPES + ['ATFM Delay (min)', 'MP Delayed Traffic']
    return pd.DataFrame(daywise_rowdata_list, columns=header)
    
def get_MAPE(y_act, y_pred):
    '''Function to calculate the Mean Absolute Percentage Error (MAPE).
        MAPE = (|y_act-y_pred| * 100) / y_act
    
        Args:
            y_act (1D numpy array): actual values.
        
            y_pred (1D numpy array): predicted values.
        
        Returns:
            MAPE (float): Mean Absolute Error Percentage
    '''
    abs_per_err = np.abs(y_pred-y_act) *100 / y_act 
    mape = abs_per_err.mean()
    return mape

def print_metrics(y_act_train, y_pred_train, y_act_test, y_pred_test, target):
    print('-----' + 'Results: ' + target + '-----')
    print('------Training Metrics------')
    print('R_squared:', r2_score(y_act_train, y_pred_train))
    print('Error % (abs):', get_MAPE(y_act_train, y_pred_train))
    print('MAE:', mean_absolute_error(y_act_train, y_pred_train))
    print('RMSE:', np.sqrt(mean_squared_error(y_act_train, y_pred_train)))
    print('------Testing Metrics------')
    print('R_squared:', r2_score(y_act_test, y_pred_test))
    print('Error % (abs):', get_MAPE(y_act_test, y_pred_test))
    print('MAE:', mean_absolute_error(y_act_test, y_pred_test))
    print('RMSE:', np.sqrt(mean_squared_error(y_act_test, y_pred_test)))

def save_metrics_detailed(y_act_train, y_pred_train, y_act_test, y_pred_test, target, job_dir):
    columns = ['category', 'train_days', 'test_days', 'train_MAPE', 'test_MAPE', 'train_R2', 'test_R2', 'train_RMSE', 'test_RMSE']
    data = []
    categories_list = DELAY_CATG_DF['category'].to_list()
    for category in categories_list:
        if target=="delay":
            lower_bound = DELAY_CATG_DF.loc[DELAY_CATG_DF['category']==category]['delay_low'].item()
            upper_bound = DELAY_CATG_DF.loc[DELAY_CATG_DF['category']==category]['delay_high'].item()
        else:
            lower_bound = DELAY_CATG_DF.loc[DELAY_CATG_DF['category']==category]['delayed_traffic_low'].item()
            upper_bound = DELAY_CATG_DF.loc[DELAY_CATG_DF['category']==category]['delayed_traffic_high'].item()
        if np.isnan(upper_bound):
                upper_bound = np.inf
        y_act_train_catg = y_act_train[(y_act_train >= lower_bound) & (y_act_train < upper_bound)]
        y_pred_train_catg = y_pred_train[(y_act_train >= lower_bound) & (y_act_train < upper_bound)]
        y_act_test_catg = y_act_test[(y_act_test >= lower_bound) & (y_act_test < upper_bound)]
        y_pred_test_catg = y_pred_test[(y_act_test >= lower_bound) & (y_act_test < upper_bound)]
        train_days = y_act_train_catg.shape[0]
        test_days = y_act_test_catg.shape[0]
        train_MAPE = get_MAPE(y_act_train_catg, y_pred_train_catg)
        test_MAPE = get_MAPE(y_act_test_catg, y_pred_test_catg)
        train_R2 = r2_score(y_act_train_catg, y_pred_train_catg)
        test_R2 = r2_score(y_act_test_catg, y_pred_test_catg)
        train_RMSE = np.sqrt(mean_squared_error(y_act_train_catg, y_pred_train_catg))
        test_RMSE = np.sqrt(mean_squared_error(y_act_test_catg, y_pred_test_catg))
        data.append([category, train_days, test_days, round(train_MAPE, 2), round(test_MAPE, 2),
                     round(train_R2, 2), round(test_R2, 2),
                     round(train_RMSE, 2), round(test_RMSE, 2)])
    metrics_df = pd.DataFrame(data, columns=columns)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, job_dir, target + "_" + "metrics.csv"), index=False)

def save_line_plots(y_act_train, y_pred_train, y_act_test, y_pred_test, target, job_dir):
    fig, ax = plt.subplots(2, 1, sharex = False, figsize=(15,8))
    fig.subplots_adjust(hspace=.4)
    ax[0].plot(range(0, len(y_act_train), 1), y_act_train, range(0, len(y_act_train), 1), y_pred_train)
    ax[0].set_title('Training set results: ' + target)
    ax[0].legend(['Actual','Prediction'])
    ax[0].set_xlabel('days (unordered)')
    ax[1].plot(range(0, len(y_act_test), 1), y_act_test, range(0, len(y_act_test), 1), y_pred_test)
    ax[1].set_title('Testing set results: ' + target)
    ax[1].legend(['Actual','Prediction'])
    ax[1].set_xlabel('days (unordered)')
    ax[1].set_ylabel('ATFM Delay (min)')
    if target == 'delay':
        ax[0].set_ylabel('delay (min)')
        ax[1].set_ylabel('delay (min)')
    else:
        ax[0].set_ylabel('delayed traffic (flights)')
        ax[1].set_ylabel('delayed traffic (flights)')
    plt.savefig(os.path.join(OUTPUT_DIR, job_dir, "lineplot.png"), bbox_inches='tight')

def save_scatter_plots(y_act_train, y_pred_train, y_act_test, y_pred_test, target, job_dir):
    fig, ax = plt.subplots(2, 1, sharex = True, figsize=(8,8))
    fig.subplots_adjust(hspace=.3)
    max_value = np.max(np.concatenate((y_act_train, y_pred_train, y_act_test, y_pred_test), axis=0))
    scatter_limit = int(max_value + 0.1 * max_value)
    ax[0].plot(range(0, scatter_limit, 1),range(0, scatter_limit, 1), color='red')
    ax[0].scatter(y_act_train, y_pred_train, alpha=0.5)
    ax[0].set_title('Training set results: ' + target)
    ax[0].legend(['Target','Prediction'])
    ax[0].set_xlabel('Actual ATFM Delay (min)')
    ax[1].plot(range(0, scatter_limit, 1),range(0, scatter_limit, 1), color='red')
    ax[1].scatter(y_act_test, y_pred_test, alpha=0.5)
    ax[1].set_title('Testing set results: ' + target)
    ax[1].legend(['Target','Prediction'])
    ax[1].set_xlabel('Actual ATFM Delay (min)')
    if target == 'delay':
        ax[0].set_ylabel('delay (min)')
        ax[1].set_ylabel('delay (min)')
    else:
        ax[0].set_ylabel('delayed traffic (flights)')
        ax[1].set_ylabel('delayed traffic (flights)')
    plt.savefig(os.path.join(OUTPUT_DIR, job_dir, "scatterplot.png"), bbox_inches='tight')

def save_predictions(y_act_train, y_pred_train, y_act_test, y_pred_test, target, job_dir):
    header = ['actual', 'prediction']
    y_act_train = np.array(y_act_train).reshape(-1,1)
    y_pred_train = np.array(y_pred_train).reshape(-1,1)
    y_act_test = np.array(y_act_test).reshape(-1,1)
    y_pred_test = np.array(y_pred_test).reshape(-1,1)
    train_result = pd.DataFrame(np.concatenate((y_act_train, y_pred_train), axis=1), columns = header)
    test_result = pd.DataFrame(np.concatenate((y_act_test, y_pred_test), axis=1), columns = header)
    train_result.to_csv(os.path.join(OUTPUT_DIR, job_dir, target + "_" + "train_results.csv"))
    test_result.to_csv(os.path.join(OUTPUT_DIR, job_dir, target + "_" + "test_results.csv"))

def register_job_log(job_dir, y_train, y_pred_train, y_test, y_pred_test):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, 'jobs_registry.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    train_MAPE = get_MAPE(y_train, y_pred_train)
    train_MAPE = str(round(train_MAPE, 2))
    test_MAPE = get_MAPE(y_test, y_pred_test)
    test_MAPE = str(round(test_MAPE, 2))
    log_msg = job_dir + " :: " + "train_MAPE" + " :: " + train_MAPE + " :: " + "test_MAPE" + " :: " + test_MAPE
    logger.info(log_msg)

def create_job_dir(job_dir):
    if not os.path.exists(os.path.join(OUTPUT_DIR, job_dir)):
        os.makedirs(os.path.join(OUTPUT_DIR, job_dir))

def save_training_file_info(train_filenames, job_dir):
    filename = os.path.join(OUTPUT_DIR, job_dir, 'training_files_info.json')
    with open(filename, 'w') as filehandler:
        json.dump({'training_files_used': train_filenames}, filehandler)