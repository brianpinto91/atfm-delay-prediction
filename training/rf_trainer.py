import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import utils
from numpy.random import seed

RANDOM_STATE = 42
seed(RANDOM_STATE)

# Grid for best parameter search
PARAM_GRID = [{'n_estimators': [50, 70, 100, 130], 'max_features': [2,4,6,8,10], 'max_depth':[5,10,20,25,50]},
              {'bootstrap':[False],'n_estimators': [50, 70, 100, 130], 'max_features': [2,4,6,8,10],
              'max_depth':[5,10,20,25,50]}]

def get_args():
    """Function to parse arguments that are used to configure the training job
        
        Args:
            None: No explicit function arguments. Arguments are passed through command line
        
        Returns:
            args: parsed arguments
    """
    parser = argparse.ArgumentParser("training hyperparameters")
    
    parser.add_argument(
        '--target',
        type=str,
        default='delay',
        metavar='L',
        help="the label for training which can be either delay or delayed_traffic"
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.3,
        metavar='TS',
        help="fraction of dataset to be used for testing"
    )
    parser.add_argument(
        '--n_folds_cv',
        type=int,
        default=10,
        metavar='CV',
        help="number of folds for cross validation during grid search"
    )
    parser.add_argument(
        '--custom_parameters',
        action='store_true',
        default=False,
        help="fraction of dataset to be used for testing"
    )
    parser.add_argument(
        '--n_estimators',
        type=int,
        default=None,
        metavar='NE',
        help="number of estimators for the Random Forest Estimator"
    )
    parser.add_argument(
        '--max_features',
        type=int,
        default=None,
        metavar='MF',
        help="max number of features to be used from all the features for the Estimator"
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=None,
        metavar='MD',
        help="max depth for each Estimator"
    )
    parser.add_argument(
        '--bootstrap',
        action='store_true',
        default=False,
        help="whether to use bootstrap during training"
    )
    args = parser.parse_args()
    return args

def train_custom_rf_model(X_train, y_train, random_state, **rf_parameters):
    '''Function to train a Random Forest Regressor with defined hyperparameters
        passed as kwargs.

        Args:
            X_train (pd dataframe or np array): a 2D (training samples X features) dataframe or a numpy array
            y_train (pd dataframe or np array): a single column target value for training
            random_state (int): seed to keep the results consistent
            **rf_parameters (**kwargs): the hyperparameters for Random forest Regressor
        
        Returns:
            rf_model: A Random Forest Regressor model fit on the training data
    '''
    rf_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **rf_parameters)
    rf_model.fit(X_train, y_train)
    return rf_model

def perform_grid_search(X_train, y_train, n_folds_cv, random_state):
    '''Function to perform a k-fold grid search and get model with best parameters from the grid.

        Args:
            X_train (pd dataframe or np array): a 2D (training samples X features) dataframe or a numpy array
            y_train (pd dataframe or np array): a single column target value for training
            n_folds_cv (int): number of folds for cross validation during grid search
            random_state (int): seed to keep the results consistent
        
        Returns:
            rf_model: A Random Forest Regressor model fit on the training data with the best parameters
                obtained from grid search
    '''
    rf_model = RandomForestRegressor()
    grid_search = GridSearchCV(rf_model, PARAM_GRID, cv=n_folds_cv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    rf_model = grid_search.best_estimator_.fit(X_train, y_train)
    return rf_model

def train(target='delay', test_size=0.3, n_folds_cv=10, **rf_parameters):
    '''Function to train a RF model. The hyperparameters are tuned using gridsearch if enabled
    '''
    print("Reading the raw data.....", flush=True)
    raw_df_17 = pd.read_csv('data/NMIR/NMIR_2017.csv')
    raw_df_18 = pd.read_csv('data/NMIR/NMIR_2018.csv')
    raw_df = pd.concat((raw_df_17, raw_df_18), axis=0).reset_index(drop=True)
    print("Transforming the data.....", flush=True)
    daywise = utils.transform_to_daywise_basic(raw_df)
    X = daywise.drop(columns=['Date', 'ATFM Delay (min)', 'MP Delayed Traffic'])
    if target == 'delay':
        y = daywise['ATFM Delay (min)']
    else:
        y = daywise['MP Delayed Traffic']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    if rf_parameters:
        print("Training RF regressor with the passed hyper-parameters.....", flush=True)
        rf_model = train_custom_rf_model(X_train, y_train, random_state=RANDOM_STATE, **rf_parameters)
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        utils.print_metrics(y_train, y_pred_train, y_test, y_pred_test)
    else:
        print("Performing grid search to find the best hyper-parameters.....", flush=True)
        rf_model = perform_grid_search(X_train, y_train, n_folds_cv, random_state=RANDOM_STATE)
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        utils.print_metrics(y_train, y_pred_train, y_test, y_pred_test)

if __name__ == "__main__":
    args = vars(get_args())
    rf_parameters = {}
    for par in ['n_estimators', 'max_features', 'max_depth', 'bootstrap']:
        if args[par]:
            rf_parameters[par] = args[par]
    if args['custom_parameters']:
        train(target=args['target'], test_size=args['test_size'], **rf_parameters)
    else:
        train(target=args['target'], test_size=args['test_size'], n_folds_cv=args['n_folds_cv'])