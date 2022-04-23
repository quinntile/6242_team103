import pip
import sys
import os

import pandas as pd
from pycaret.regression import *


## Load the preprocessed data for austin housing prices

        
        


class Modeling:
    def __init__(self, training_path, unseendata_path):
        self.current_dir = os.path.dirname(__file__)
        self.training_data = pd.read_csv(self.current_dir[:-5] + training_path)
        self.unseen_data = pd.read_csv(self.current_dir[:-5] + unseendata_path)

        self.set_up = setup(data = self.training_data, target = 'SalePrice', 
                   normalize = True,
                   transformation= False, transformation_method = 'yeo-johnson', 
                   numeric_features=['median_income', 'YrBuilt', 'SqFtLot', 'SqFtTotLiving', 'mobility_us',
                    'mobility_county', 'stockSPY', 'mobility_abroad', 'unemployment',
                    'mobility_state'],
                   transform_target = True, remove_outliers= False, 
                   remove_multicollinearity = True,
                   ignore_low_variance = True, combine_rare_levels = True) 
        
    def run_modelling(self, n_iter = 50):
        self.lightgbm = create_model('lightgbm')
        self.tuned_lightgbm = tune_model(self.lightgbm, optimize = 'MAE', n_iter = n_iter)
        # plot_model(self.tuned_rf, plot = 'error')
        # plot_model(self.tuned_rf, plot = 'feature')
        print(self.tuned_lightgbm)
        
        self.unseen_predictions = predict_model(self.tuned_lightgbm, data= self.unseen_data[['median_income', 'YrBuilt', 'SqFtLot',
       'SqFtTotLiving', 'mobility_us', 'mobility_county', 'stockSPY',
       'mobility_abroad', 'unemployment', 'mobility_state']])
        self.final = pd.merge(self.unseen_data[['ZipCode', 'Month']],self.unseen_predictions, left_index=True, right_index=True)
        



# Example
if __name__ == '__main__':
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycaret==2.3.9'])
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.20'])
    ####### current_dir[:-5] + '/data/data_unseen.csv'
    current_dir = os.path.dirname(__file__)
    data = Modeling(training_path = '/data/training_seattle.csv', unseendata_path = '/data/data_unseen_seattle.csv')
    data.run_modelling(n_iter=1)
    data.final.to_csv(current_dir[:-5] + '/visualization/data_source/Seattle_price.csv')


