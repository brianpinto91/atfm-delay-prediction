import pandas as pd
import numpy as np
import datetime as dt

# Import the database for AIRACs
AIRAC_DF = pd.read_csv('data/AIRAC_dates.csv')
AIRAC_DF['start_date'] = pd.to_datetime(AIRAC_DF['start_date'])
AIRAC_DF['end_date'] = pd.to_datetime(AIRAC_DF['end_date'])

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
        except:
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

def print_metrics(y_act_train, y_pred_train, y_act_test, y_pred_test):
    pass

def print_metrics_detailed(y_act_train, y_pred_train, y_act_test, y_pred_test):
    pass

def save_line_plots(y_act, y_pred, title, axis_label):
    pass

def save_scatter_plots(y_act, y_pred, title, axis_label):
    pass

def save_predictions(y_act, y_pred, filename_prefix):
    pass
