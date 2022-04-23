import os
import pandas as pd
import numpy as np


class PreProcess_seattle:
    def __init__(self, filepath):
        
        self.current_dir = os.path.dirname(__file__)
        self.data = pd.read_csv(filepath)
        self.data.drop(columns=['Unnamed: 0', "WaterSystem", "SewerSystem", "Access", "TrafficNoise", "AirportNoise"
        , "PowerLines", "WaterProblems", "YrRenovated"], inplace = True)
        self.data.drop_duplicates(subset = ["Major", "Minor", "DocumentDate"], inplace=True)
        ### 95% quantile for sqftlot and 99.98% quantile for saleprice.
        self.data = self.data[(self.data["SqFtLot"] <=47862.6) & (self.data['SalePrice'] <= 9944196.8)]
        self.data = self.data.reset_index(drop=True)


        self.data.fillna(method = 'ffill', inplace=True)
        self.data['unemployment'] = self.data['unemployment'].str[:-1].astype(float)/100
        self.data['mobility_abroad'] = self.data['mobility_abroad'].str[:-1].astype(float)/100
        self.data['mobility_county'] = self.data['mobility_county'].str[:-1].astype(float)/100
        self.data['mobility_state'] = self.data['mobility_state'].str[:-1].astype(float)/100
        self.data['mobility_us'] = self.data['mobility_us'].str[:-1].astype(float)/100
        self.training_seattle = self.data[['median_income', 'YrBuilt', 'SqFtLot', 'SqFtTotLiving', 
'mobility_us', 'mobility_county', 'stockSPY', 'mobility_abroad', 'unemployment' , 'mobility_state', 'SalePrice']]



        ### filter the columns of original df
        # self.data = self.data[['period_begin', 'zip_code', 'Year', 'median_income', 
        # 'SPY', 'Unemployment', 'mobility_withUS', 'property_type', 'mobility_withincounty', 
        # 'mobility_abroad', 'mobility_withinstate', 'median_sale_price']]
        

    def read_features(self):
        
        self.spy_predict_seattle= pd.read_csv(self.current_dir[:-5] +"/data/Data_mortgage_stocks/forecast_spy_seattle.csv")
        
        self.median_income_WA_predict = pd.read_csv(self.current_dir[:-5] +"/data/median_income/median_income_WA_predicted.csv")
        
        self.unemployment_seattle_predict = pd.read_csv(self.current_dir[:-5] +"/data/unemployment_rate/unemployment_seattle_predicted.csv")
        self.mobility_seattle_county_predict = pd.read_csv(self.current_dir[:-5] +"/data/mobility/mobility_withincounty_king_predicted.csv")
        self.mobility_seattle_state_predict = pd.read_csv(self.current_dir[:-5] +"/data/mobility/mobility_withinstate_king_predicted.csv")
        self.mobility_seattle_US_predict = pd.read_csv(self.current_dir[:-5] +"/data/mobility/mobility_withinUS_king_predicted.csv")
        self.mobility_seattle_abroad_predict = pd.read_csv(self.current_dir[:-5] +"/data/mobility/mobility_abroad_king_predicted.csv")
        

    def clean_features(self):
        self.median_income_WA_predict.drop_duplicates(["zipcode", "year"], inplace=True)
        self.unemployment_seattle_predict.drop_duplicates(["zipcode", "year"],inplace=True)             

        self.median_income_WA_predict = self.median_income_WA_predict[self.median_income_WA_predict['year']==2022]
        self.unemployment_seattle_predict = self.unemployment_seattle_predict[self.unemployment_seattle_predict['year']==2022]
        self.mobility_seattle_US_predict = self.mobility_seattle_US_predict[self.mobility_seattle_US_predict['year']==2022]
        self.mobility_seattle_county_predict = self.mobility_seattle_county_predict[self.mobility_seattle_county_predict['year']==2022]
        self.mobility_seattle_abroad_predict = self.mobility_seattle_abroad_predict[self.mobility_seattle_abroad_predict['year']==2022]
        self.mobility_seattle_state_predict = self.mobility_seattle_state_predict[self.mobility_seattle_state_predict['year']==2022]

        

    def join_unseen_features(self):
        self.data_unseen_seattle = self.data[['ZipCode', 'SqFtLot', 'SqFtTotLiving']].groupby(by = 'ZipCode').median().reset_index()
        self.data_unseen_seattle = self.data_unseen_seattle.loc[self.data_unseen_seattle.index.repeat(43)]
        self.data_unseen_seattle['YrBuilt'] = 0
        years = [1980 + i for i in range(43)]
        self.data_unseen_seattle = self.data_unseen_seattle.reset_index(drop=True)
        for idx, row in self.data_unseen_seattle.iterrows():
            # if data_unseen_seattle.iloc[idx, 0] == data_unseen_seattle.iloc[idx+1, 0]:
            # print(idx)
            # print(idx % 5)
            # print(properties[idx % 5])
            self.data_unseen_seattle.iloc[idx, 3] = years[idx % 43]
        #data_unseen_seattle.head(20)
        self.data_unseen_seattle = self.data_unseen_seattle.reset_index(drop=True)
        
        self.data_unseen_seattle = self.data_unseen_seattle.merge(self.median_income_WA_predict, how = 'left', left_on='ZipCode', right_on='zipcode')
        self.data_unseen_seattle.drop(columns = ['zipcode', 'year'], inplace = True)
        self.data_unseen_seattle = self.data_unseen_seattle.merge(self.unemployment_seattle_predict, how = 'left', left_on='ZipCode', right_on='zipcode')
        self.data_unseen_seattle.drop(columns = ['zipcode', 'year'], inplace = True)
        months = ['2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12']
        self.data_unseen_seattle = self.data_unseen_seattle.loc[self.data_unseen_seattle.index.repeat(9)]
        self.data_unseen_seattle = self.data_unseen_seattle.reset_index(drop=True)
        self.data_unseen_seattle['Month'] = ''
        for idx, row in self.data_unseen_seattle.iterrows():
            self.data_unseen_seattle.iloc[idx, 6] = months[idx % 9]

        self.data_unseen_seattle = self.data_unseen_seattle.merge(self.spy_predict_seattle, how = 'left', left_on='Month', right_on='ds')
        self.data_unseen_seattle.drop(columns = ['ds'], inplace=True)
        self.data_unseen_seattle.rename(columns = {'SPY' : 'stockSPY'}, inplace=True)

        
        self.data_unseen_seattle = self.data_unseen_seattle.merge(self.mobility_seattle_state_predict, how = 'left', left_on='ZipCode', right_on='zipcode')
        self.data_unseen_seattle.drop(columns = ['zipcode', 'year'], inplace=True)
        self.data_unseen_seattle.rename(columns = {'mobility' : 'mobility_state'}, inplace=True)

        self.data_unseen_seattle = self.data_unseen_seattle.merge(self.mobility_seattle_US_predict, how = 'left', left_on='ZipCode', right_on='zipcode')
        self.data_unseen_seattle.drop(columns = ['zipcode', 'year'], inplace=True)
        self.data_unseen_seattle.rename(columns = {'mobility' : 'mobility_US'}, inplace=True)

        self.data_unseen_seattle = self.data_unseen_seattle.merge(self.mobility_seattle_county_predict, how = 'left', left_on='ZipCode', right_on='zipcode')
        self.data_unseen_seattle.drop(columns = ['zipcode', 'year'], inplace=True)
        self.data_unseen_seattle.rename(columns = {'mobility' : 'mobility_county'}, inplace=True)

        self.data_unseen_seattle = self.data_unseen_seattle.merge(self.mobility_seattle_abroad_predict, how = 'left', left_on='ZipCode', right_on='zipcode')
        self.data_unseen_seattle.drop(columns = ['zipcode', 'year'], inplace=True)
        self.data_unseen_seattle.rename(columns = {'mobility' : 'mobility_abroad'}, inplace=True)
        self.data_unseen_seattle.rename(columns = {'unemployment_rate':'unemployment', 'mobility_US':'mobility_us'}, inplace = True)
        self.data_unseen_seattle = self.data_unseen_seattle.reindex(columns=['ZipCode', 'Month', 'median_income', 'YrBuilt', 'SqFtLot', 'SqFtTotLiving', 'mobility_us',
                                                    'mobility_county', 'stockSPY', 'mobility_abroad', 'unemployment',
                                                    'mobility_state'])

        return self.data_unseen_seattle

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
    data = PreProcess_seattle(current_dir[:-5] + '/data/seattle_house_price/clean_seattle_price_reg_train.csv')
    #print(data.get_num_rows())
    # print(data.get_cols())
    data.read_features()
    # print(data.mobility_seattle_county_predict)
    data.clean_features()
    data_unseen = data.join_unseen_features()
    
    
    data_unseen.to_csv(current_dir[:-5] + '/data/data_unseen_seattle.csv')
    data.training_seattle.to_csv(current_dir[:-5] + '/data/training_seattle.csv')

    # print(data.data_unseen_austin.head())
    