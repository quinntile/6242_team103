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
        self.set_up = setup(data = self.training_data, target = 'median_sale_price', 
                   normalize = True,
                   transformation= False, ignore_features = ['Unnamed: 0'],transformation_method = 'yeo-johnson', 
                   categorical_features = ['property_type'], 
                   numeric_features=['median_income', 'SPY', 'Unemployment', 'mobility_withUS',  
                   'mobility_withincounty', 'mobility_abroad', 'mobility_withinstate'],
                   transform_target = True, remove_outliers= False, 
                   remove_multicollinearity = True,
                   ignore_low_variance = True, combine_rare_levels = True) 
    def run_modelling(self, n_iter = 50):
        self.rf = create_model('rf')
        self.tuned_rf = tune_model(self.rf, optimize = 'MAE', n_iter = n_iter)
        # plot_model(self.tuned_rf, plot = 'error')
        # plot_model(self.tuned_rf, plot = 'feature')
        print(self.tuned_rf)
        
        self.unseen_predictions = predict_model(self.tuned_rf, data= self.unseen_data[['median_income', 'SPY', 'Unemployment',
       'mobility_withUS', 'property_type', 'mobility_withincounty',
       'mobility_abroad', 'mobility_withinstate']])
        self.final = pd.merge(self.unseen_data[['zipcode', 'Quarter']],self.unseen_predictions, left_index=True, right_index=True)
        



# Example
if __name__ == '__main__':
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pycaret==2.3.9'])
    # subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.20'])
    ####### current_dir[:-5] + '/data/data_unseen.csv'
    current_dir = os.path.dirname(__file__)
    data = Modeling(training_path = '/data/training_austin.csv', unseendata_path = '/data/data_unseen_austin.csv')
    data.run_modelling(n_iter=1) # number of iteration during grid search
    data.final.to_csv(current_dir[:-5] + '/data/prediction_result_austin.csv')


