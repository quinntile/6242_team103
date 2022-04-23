import os
import pandas as pd
import numpy as np


class PreProcess_austin:
    def __init__(self, filepath):
        
        self.current_dir = os.path.dirname(__file__)
        self.data = pd.read_csv(filepath)
        self.data.drop(columns=['Unnamed: 0',  "property_type_id", "school_rating"], inplace = True)
        self.data = self.data.astype({"Year": str, "zip_code":str})
        self.data.fillna(method = 'ffill', inplace=True)
        self.data.drop_duplicates(subset = ["period_begin", "zip_code", "property_type"], inplace=True)
        self.data['Unemployment'] = self.data['Unemployment'].str[:-1].astype(float)/100
        self.data['mobility_abroad'] = self.data['mobility_abroad'].str[:-1].astype(float)/100
        self.data['mobility_withincounty'] = self.data['mobility_withincounty'].str[:-1].astype(float)/100
        self.data['mobility_withinstate'] = self.data['mobility_withinstate'].str[:-1].astype(float)/100
        self.data['mobility_withUS'] = self.data['mobility_withUS'].str[:-1].astype(float)/100        
        ### filter the columns of original df
        self.data = self.data[['period_begin', 'zip_code', 'Year', 'median_income', 
        'SPY', 'Unemployment', 'mobility_withUS', 'property_type', 'mobility_withincounty', 
        'mobility_abroad', 'mobility_withinstate', 'median_sale_price']]
        

    def read_features(self):
        self.DGL_predict_austin = pd.read_csv(self.current_dir[:-5] + "/data/Data_mortgage_stocks/forecast_dgl_austin.csv")
        self.spy_predict_austin = pd.read_csv(self.current_dir[:-5] +"/data/Data_mortgage_stocks/forecast_spy_austin.csv")
        
        self.median_income_TX_predict = pd.read_csv(self.current_dir[:-5] +"/data/median_income/median_income_TX_predicted.csv")

        self.unemployment_austin_predict = pd.read_csv(self.current_dir[:-5] +"/data/unemployment_rate/unemployment_austin_predicted.csv")
       
        self.mobility_austin_county_predict = pd.read_csv(self.current_dir[:-5] +"/data/mobility/predict_mobility_austin_county.csv")
        self.mobility_austin_US_predict = pd.read_csv(self.current_dir[:-5] +"/data/mobility/predict_mobility_austin_US.csv")
        self.mobility_austin_abroad_predict = pd.read_csv(self.current_dir[:-5] +"/data/mobility/predict_mobility_austin_abroad.csv")
        self.mobility_austin_state_predict = pd.read_csv(self.current_dir[:-5] +"/data/mobility/predict_mobility_austin_state.csv")
        
        
        self.properties = self.data['property_type'].unique()
        self.quarters = self.spy_predict_austin['date'].unique()

    def clean_features(self):
        self.df_wo_zip_period = self.data.drop(columns=['zip_code', 'period_begin'])        
        self.training_austin = self.df_wo_zip_period[['median_income', 'SPY', 'Unemployment', 
        'mobility_withUS', 'property_type', 'mobility_withincounty', 'mobility_abroad', 
        'mobility_withinstate', 'median_sale_price']]
        self.median_income_TX_predict.drop_duplicates(['zipcode', 'year'], inplace=True)

        self.unemployment_austin_predict = self.unemployment_austin_predict[self.unemployment_austin_predict['year']==2022].drop_duplicates(['zipcode', 'year'])
        self.mobility_austin_county_predict = self.mobility_austin_county_predict[['Zip Code', '2022']].drop_duplicates(['Zip Code', '2022'])
        
        self.mobility_austin_US_predict = self.mobility_austin_US_predict[['Zip Code', '2022']].drop_duplicates(['Zip Code', '2022'])
        self.mobility_austin_abroad_predict = self.mobility_austin_abroad_predict[['Zip Code', '2022']].drop_duplicates(['Zip Code', '2022'])
        self.mobility_austin_state_predict = self.mobility_austin_state_predict[['Zip Code', '2022']].drop_duplicates(['Zip Code', '2022'])
        

    def join_unseen_features(self):
        self.data_unseen_austin = self.median_income_TX_predict[self.median_income_TX_predict['year']==2022].reset_index(drop=True)
        self.data_unseen_austin = self.data_unseen_austin.loc[self.data_unseen_austin.index.repeat(5)]
        self.data_unseen_austin = self.data_unseen_austin.reset_index(drop=True)
        self.data_unseen_austin['property_type'] = ''
        for idx, row in self.data_unseen_austin.iterrows():
            # if data_unseen.iloc[idx, 0] == data_unseen.iloc[idx+1, 0]:
            # print(idx)
            # print(idx % 5)
            # print(properties[idx % 5])
            self.data_unseen_austin.iloc[idx, 3] = self.properties[idx % 5]

        self.data_unseen_austin = self.data_unseen_austin.reset_index(drop=True)

        #data_unseen.head(20)
        self.data_unseen_austin = self.data_unseen_austin.loc[self.data_unseen_austin.index.repeat(3)]
        self.data_unseen_austin = self.data_unseen_austin.reset_index(drop=True)
        self.data_unseen_austin['Quarter'] = ''
        #data_unseen.head(20)


        # data_unseen.index
        for idx, row in self.data_unseen_austin.iterrows():
            # if data_unseen.iloc[idx, 0] == data_unseen.iloc[idx+1, 0]:
            
            self.data_unseen_austin.iloc[idx, 4] = self.quarters[idx % 3]
        # data_unseen['Quarter'] = data_unseen['Quarter'].values.astype('<M8[D]')
        #data_unseen.dtypes

        self.data_unseen_austin = self.data_unseen_austin.merge(self.spy_predict_austin, how='left', left_on='Quarter', right_on = 'date')
        self.data_unseen_austin = self.data_unseen_austin.drop(columns = 'date')
        self.data_unseen_austin = self.data_unseen_austin.reindex(columns = ['zipcode', 'year', 'Quarter','property_type', 'median_income', 'SPY'])
        self.data_unseen_austin = self.data_unseen_austin.merge(self.unemployment_austin_predict, how = 'left', left_on = 'zipcode', right_on = 'zipcode')
        self.data_unseen_austin.drop(columns = ['year_x', 'year_y'], inplace = True)    
        self.data_unseen_austin = self.data_unseen_austin.merge(self.mobility_austin_county_predict, how = 'left', left_on='zipcode', right_on = 'Zip Code')
        self.data_unseen_austin.drop(columns  = 'Zip Code', inplace = True)
        self.data_unseen_austin.rename(columns = {'2022': 'mobility_withincounty'}, inplace = True)

        self.data_unseen_austin = self.data_unseen_austin.merge(self.mobility_austin_US_predict, how = 'left', left_on='zipcode', right_on = 'Zip Code')
        self.data_unseen_austin.drop(columns  = 'Zip Code', inplace = True)
        self.data_unseen_austin.rename(columns = {'2022': 'mobility_austin_US_predict'}, inplace = True)

        self.data_unseen_austin = self.data_unseen_austin.merge(self.mobility_austin_abroad_predict, how = 'left', left_on='zipcode', right_on = 'Zip Code')
        self.data_unseen_austin.drop(columns  = 'Zip Code', inplace = True)
        self.data_unseen_austin.rename(columns = {'2022': 'mobility_austin_abroad_predict'}, inplace = True)


        self.data_unseen_austin = self.data_unseen_austin.merge(self.mobility_austin_state_predict, how = 'left', left_on='zipcode', right_on = 'Zip Code')
        self.data_unseen_austin.drop(columns  = 'Zip Code', inplace = True)
        self.data_unseen_austin.rename(columns = {'2022': 'mobility_austin_state_predict'}, inplace = True)
        self.data_unseen_austin.rename(columns = {'unemployment_rate':'Unemployment', 
        'mobility_austin_US_predict': 'mobility_withUS', 
        'mobility_austin_abroad_predict': 'mobility_abroad', 
        'mobility_austin_state_predict':'mobility_withinstate'}, inplace = True)
        self.data_unseen_austin = self.data_unseen_austin.reindex(columns = ['zipcode', 'Quarter', 'median_income', 'SPY', 'Unemployment', 'mobility_withUS',
            'property_type', 'mobility_withincounty', 'mobility_abroad',
            'mobility_withinstate'])
        return self.data_unseen_austin

    def get_cols(self):
        return self.data.columns

    def check_cardinality(self):
        print("Checking cardinality of all columns in the dataframe: \n")
        print(self.data.apply(pd.Series.nunique))

    def get_num_rows(self):
        return len(self.data)

    @staticmethod
    def save_to_csv(df):
        df.to_csv('data.csv')


if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    data = PreProcess_austin(current_dir[:-5] + '/data/Austin housing market data/clean_austin_price_reg_train.csv')
    # print(data.get_cols())
    data.read_features()
    #print(data.mobility_seattle_county_predict)
    data.clean_features()
    data_unseen = data.join_unseen_features()
    
    data_unseen.to_csv(current_dir[:-5] + '/data/data_unseen_austin.csv')
    data.training_austin.to_csv(current_dir[:-5] + '/data/training_austin.csv')

    # print(data.data_unseen_austin.head())
    