import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


class Modeling:
    def __init__(self, training_path):
        self.current_dir = os.getcwd()
        self.new_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        self.training_data = pd.read_csv(self.new_dir + training_path)

    def ts_learner(self, loc='Austin', date_col = 'period_begin', pred_col = 'median_list_price', property_type = 'Single Family Residential', periods = 365):
        data = self.training_data
        if loc == 'Austin':
            date_col = 'period_begin'
            pred_col = 'median_list_price'
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.loc[data['property_type'] == property_type, [date_col, pred_col]]
            data.rename(columns = {date_col: "ds", pred_col: "y"}, inplace = True)
        else:
            date_col = 'DocumentDate'
            pred_col = 'SalePrice'
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.loc[:, [date_col, pred_col]]
            data.rename(columns = {date_col: "ds", pred_col: "y"}, inplace = True)
            periods=365*2
        self.training_data = data
        m = Prophet()
        m.fit(data)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        forecast.ds = pd.to_datetime(forecast.ds)
        return data, forecast


if __name__ == '__main__':
    current_dir = os.getcwd()
    new_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_a = Modeling(training_path = '/data/Austin_housing_market_data/clean_austin_price_reg_train.csv')
    austin_SFH, forecast2 = data_a.ts_learner()
    forecast2.rename(columns = {'ds': 'ds_sfh', 'yhat': 'yhat_sfh', 'yhat_upper': 'yhat_sfh_upper', 'yhat_lower': 'yhat_sfh_lower'}, inplace = True)
    data_b = Modeling(training_path = '/data/Austin_housing_market_data/clean_austin_price_reg_train.csv')
    austin_TH, forecast3 = data_b.ts_learner(property_type = 'Townhouse')
    forecast3.rename(columns = {'ds': 'ds_th', 'yhat': 'yhat_th', 'yhat_upper': 'yhat_th_upper', 'yhat_lower': 'yhat_th_lower'}, inplace = True)
    data_c = Modeling(training_path = '/data/Austin_housing_market_data/clean_austin_price_reg_train.csv')
    austin_C, forecast4 = data_c.ts_learner(property_type = 'Condo/Co-op')
    forecast4.rename(columns = {'ds': 'ds_c', 'yhat': 'yhat_c', 'yhat_upper': 'yhat_c_upper', 'yhat_lower': 'yhat_c_lower'}, inplace = True)
    line1 = forecast2.loc[:, ['ds_sfh','yhat_sfh_lower','yhat_sfh_upper','yhat_sfh']].merge(forecast3.loc[:, ['ds_th','yhat_th_lower','yhat_th_upper','yhat_th']], how = 'outer', left_on='ds_sfh', right_on='ds_th')
    line2 = line1.merge(forecast4.loc[:, ['ds_c','yhat_c_lower','yhat_c_upper','yhat_c']], how = 'outer', left_on='ds_sfh', right_on='ds_c')
    line3 = line2.iloc[:, (line2.columns != 'ds_sfh') & (line2.columns != 'ds_th') & (line2.columns != 'ds_c')]
    line3[line3 < 0] = 0
    austin_SFH_md = austin_SFH.groupby('ds').mean().reset_index()
    austin_TH_md = austin_TH.groupby('ds').mean().reset_index()
    austin_C_md = austin_C.groupby('ds').mean().reset_index()
    austin_SFH_md.rename(columns = {'y':'y_SFH'}, inplace = True)
    austin_TH_md.rename(columns = {'y':'y_TH'}, inplace = True)
    austin_C_md.rename(columns = {'y':'y_C'}, inplace = True)
    line5 = line2.merge(austin_SFH_md, how = 'left', left_on='ds_sfh', right_on='ds')
    line6 = line5.merge(austin_TH_md, how = 'left', left_on='ds_th', right_on='ds')
    line7 = line6.merge(austin_C_md, how = 'left', left_on='ds_c', right_on='ds')
    a_r = pd.read_csv(new_dir + '/visualization/data_source/Austin_price.csv')
    a_r_sfh = a_r.loc[a_r['property_type'] == 'Single Family Residential', ['Quarter', 'Label']]
    a_r_sfh_avg = a_r_sfh.groupby('Quarter').mean().reset_index()
    a_r_sfh_avg.Quarter = pd.to_datetime(a_r_sfh_avg.Quarter)
    a_r_C = a_r.loc[a_r['property_type'] == 'Condo/Co-op', ['Quarter', 'Label']]
    a_r_C_avg = a_r_C.groupby('Quarter').mean().reset_index()
    a_r_C_avg.Quarter = pd.to_datetime(a_r_C_avg.Quarter)
    a_r_th = a_r.loc[a_r['property_type'] == 'Townhouse', ['Quarter', 'Label']]
    a_r_th_avg = a_r_th.groupby('Quarter').mean().reset_index()
    a_r_th_avg.Quarter = pd.to_datetime(a_r_th_avg.Quarter)
    dates = pd.date_range(start="2022-04-01",end="2022-12-31").to_frame()
    s_r = pd.read_csv(new_dir + '/visualization/data_source/Seattle_price.csv')
    s_r2 = s_r[['Month', 'Label']]
    s_r_avg = s_r2.groupby('Month').mean().reset_index()
    s_r_avg.Month = pd.to_datetime(s_r_avg.Month)
    dates['Label'] = s_r_avg.iloc[0, 1]
    dates['a_c'] = a_r_C_avg.iloc[0, 1]
    dates['a_th'] = a_r_th_avg.iloc[0, 1]
    dates['a_sfh'] = a_r_sfh_avg.iloc[0, 1]
    data_s = Modeling(training_path = '/data/seattle_house_price/clean_seattle_price_reg_train.csv')
    s, forecast5 = data_s.ts_learner(loc='Seattle')
    s_avg = s.groupby('ds').mean().reset_index()
    line8 = forecast5.merge(s_avg, how = 'left', left_on='ds', right_on='ds')
    line9 = line8.merge(dates, how = 'left', left_on='ds', right_on=0)
    line10 = line9.iloc[:, (line9.columns != 'ds') & (line9.columns != 'Month')]
    line12 = line7.merge(dates, how = 'left', left_on = 'ds_sfh', right_on = 0)

    # plots for austin

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    x = line2['ds_sfh']
    y = line3['yhat_sfh']
    y_u = line3['yhat_sfh_upper']
    y_l = line3['yhat_sfh_lower']
    y_th = line3['yhat_th']
    y_u_th = line3['yhat_th_upper']
    y_l_th = line3['yhat_th_lower']
    y_c = line3['yhat_c']
    y_u_c = line3['yhat_c_upper']
    y_l_c = line3['yhat_c_lower']
    y_reg_sfh = line12['a_sfh']
    y_reg_th = line12['a_th']
    y_reg_c = line12['a_c']
    # forecast.loc[:, ['ds','yhat_lower','yhat_upper','yhat']].set_index('ds').plot()
    ax.plot(x, y, color='dodgerblue', label = 'GAM Foreecarst for Single Family Home')
    ax.fill_between(x, y_u, y_l, color='dodgerblue', alpha=.1)
    ax.plot(x, y_th, color = 'orange', label = 'GAM Foreecarst for Townhouse')
    ax.fill_between(x, y_u_th, y_l_th, color='orange', alpha=.1)
    ax.plot(x, y_c, color = 'limegreen', label = 'GAM Foreecarst for Condo')
    ax.plot(x, y_reg_th, label = 'Regression Prediction for Townhouse', color = 'orangered', linestyle='dashed', marker='o',  linewidth=3)
    ax.plot(x, y_reg_c, label = 'Regression Prediction for Condo', color = 'lime', linestyle='dashed', marker='o',  linewidth=3)
    ax.plot(x, y_reg_sfh, label = 'Regression Prediction for Single Family Home', color = 'blue', linestyle='dashed', marker='o',  linewidth=3)
    ax.fill_between(x, y_u_c, y_l_c, color='limegreen', alpha=.1)
    plt.legend(loc = 'upper left')
    plt.title("10-Year Housing Forecast for Austin in 1 Year by Property Type", fontsize=20)
    plt.ylabel("Price (in $)")
    plt.savefig(new_dir + '/visualization/Austin_full_forecast.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    x = line7['ds_sfh']
    y = line3['yhat_sfh']
    y_u = line3['yhat_sfh_upper']
    y_l = line3['yhat_sfh_lower']
    y_avg = line7['y_SFH']
    y_reg = line12['a_sfh']
    ax.plot(x, y, label = 'GAM Forecast for Single Family Home', color = 'dodgerblue', linestyle='dashed')
    ax.plot(x, y_avg, color = 'blue', label = 'Actual Average price for Single Family Home')
    ax.plot(x, y_reg, label = 'Regression Prediction for Single Family Home', color = 'darkblue', linestyle='dashed', marker='o',  linewidth=5)
    ax.fill_between(x, y_u, y_l, color='dodgerblue', alpha=.1)
    plt.legend(loc = 'best')
    plt.title("10-Year Housing Forecast Performance for Austin in 1 Year (Single Family Home)", fontsize=20)
    plt.ylabel("Price (in $)")
    plt.savefig(new_dir + '/visualization/Austin_sfh_forecast.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    x = line7['ds_th']
    y = line3['yhat_th']
    y_u = line3['yhat_th_upper']
    y_l = line3['yhat_th_lower']
    y_avg = line7['y_TH']
    y_reg = line12['a_th']
    ax.plot(x, y, label = 'Forecast for Townhouse', color = 'orange', linestyle='dashed')
    ax.plot(x, y_avg, color = 'orangered', label = 'Actual Average price for Townhouse')
    ax.plot(x, y_reg, label = 'Regression Prediction for Townhouse', color = 'darkorange', linestyle='dashed', marker='o',  linewidth=5)
    ax.fill_between(x, y_u, y_l, color='orange', alpha=.1)
    plt.legend(loc = 'best')
    plt.title("10-Year Housing Forecast Performance for Austin in 1 Year (Townhouse)", fontsize=20)
    plt.ylabel("Price (in $)")
    plt.savefig(new_dir + '/visualization/Austin_th_forecast.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    x = line7['ds_c']
    y = line3['yhat_c']
    y_u = line3['yhat_c_upper']
    y_l = line3['yhat_c_lower']
    y_avg = line7['y_C']
    y_reg = line12['a_c']
    ax.plot(x, y, label = 'Forecast for Condo', color = 'lime', linestyle='dashed')
    ax.plot(x, y_avg, color = 'forestgreen', label = 'Actual Average price for Condo')
    ax.plot(x, y_reg, label = 'Regression Prediction for Condo', color = 'darkgreen', linestyle='dashed', marker='o',  linewidth=5)
    ax.fill_between(x, y_u, y_l, color='lime', alpha=.1)
    plt.legend(loc = 'best')
    plt.title("10-Year Housing Forecast Performance for Austin in 1 Year (Condo)", fontsize=20)
    plt.ylabel("Price (in $)")
    plt.savefig(new_dir + '/visualization/Austin_condo_forecast.png')

    # plots for Seattle
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    x = forecast5['ds']
    y = line10['yhat']
    y_u = line10['yhat_upper']
    y_l = line10['yhat_lower']
    y_avg = line8['y']
    y_reg = line9['Label']
    ax.plot(x, y, label = 'GAM Forecast for Single Family Home', color = 'dodgerblue', linestyle='dashed')
    ax.plot(x, y_avg, 'g-', label = 'Actual Average price for Single Family Home', alpha = 0.5)
    ax.plot(x, y_reg, label = 'Regression Prediction for Single Family Home', color = 'lime', linestyle='dashed', marker='o',  linewidth=5)
    ax.fill_between(x, y_u, y_l, color='dodgerblue', alpha=.1)
    plt.legend(loc = 'best')
    plt.title("10-Year Housing Forecast Performance for Seattle in 1 Year (Single Family Home)", fontsize=20)
    plt.ylabel("Price (in $)")
    plt.savefig(new_dir + '/visualization/Seattle_forecast.png')
