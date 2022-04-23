DESCRIPTION
------------

 * This file contains instructions on how to set up the working environment so that prediction 
of austin and seattle housing market data into the later half of Year 2022 can be performed. 
 * This package contains full functionality of preprocessing the data, including normalization, feature extraction, data cleaning, model creation, model hyperparameter tuning based on cross validation and model prediction.
 * Specifically, this package fits a random forest regressor for Austin housing market, and a light gradient boosting machine for Seattle individual houses. Both estimator has been validated using 10-fold cross validation and number of iterations during each grid search can be specified by the user.
 * In terms of time series forecasted features used in the regression for future:
   1) Austin mobility prediction - pycaret：naive forecaster
   2) SPY, DGL & Mortgage rate: prophet, GAM model
   3) median income for each zipcode: exponential smoothing
   4) seattle mobility (county, state, US, abroad): exponential smoothing
   5) unemployment rate: exponential smoothing

 * To submit bug reports and feature suggestions, or track changes, go to the github repo:
   https://github.com/xiaoyangxuoo/SeattleAustinHousePred.git


REQUIREMENTS
------------

This package has been developed and tested using python 3.7.9 and for dependencies,
see requirements.txt for reference:

 * pycaret==2.3.10
 * numpy==1.20.0
 * scikit-learn==0.23.2
 * pandas==1.3.5


INSTALLATION
------------
 * Open a terminal in your current working directory, and run the following commands:
    git clone https://github.com/xiaoyangxuoo/SeattleAustinHousePred.git

 * Open a terminal in your current working directory, and run the following commands:
    pip install -r requirements.txt


EXECUTION
------------

The complete workflow of this project is as follows:
PreProcessing --> Modelling --> visualization


Separately, the EXECUTION of each step is:
 * For preprocessing (reading cleaned data and join features), open the PreProcessing_XXX.py for each city, where XXX represents the city name.
 * Run the PreProcessing_XXX.py code in a local python kernel, where the fate of the processing step is two files, data_unseen_XXX.csv and training_XXX.csv, where XXX represents your city of choice. Inside the PreProcessing_XXX class, we have defined several methods:
 
    ## get_cols(): Output the current dataframe's columns to examine before being fed into a model.
    ## read_features(): read predicted features into one whole dataframe
   
    ## clean_features(): clean all the features to make them ready for processing by the pycaret modelling functions
    ## join_unseen_features(): use time series predicted features into the future to make a dataset called "data_unseen_XXX" where XXX represents the city of your choice.
    
 * Upon completion of the above functions in sequence, you can save the ready training data and unseen data into the folder data by using pd.to_csv() API, so that these files can be used for modelling. Example is as below: 
      data_unseen.to_csv(current_dir[:-5] + '/data/data_unseen_austin.csv')
      data.training_austin.to_csv(current_dir[:-5] + '/data/training_austin.csv')

 * Run the Regression_XXX.py code in the same local python kernel, where the fate of the modelling step is rediction_result_XXX, where XXX represents the city of your choice.Inside the Modelling class, we have defined several methods:
 
    ## run_modelling(): create the model (random forest regressor or lightgbm regressor, depending on the city of your choice), tune the model using 10-fold cross validation(n_iter represents the number of iterations the grid search algorithm uses when tuning the hyperparameters, can be specified by user),  and use the tuned model to predict the house prices into the future. In our use case, we predicted into the next 9 months (3 quarters) in the year of 2022. 

 * Upon completion of the above functions in sequence, you can save the predicted data into the folder data by using pd.to_csv() API, so that these files can be used for visualization. Example is as below: 
      data.final.to_csv(current_dir[:-5] + '/visualization/data_source/Austin_price.csv')
    


After completion of the above preprocessing and modelling steps, you can start exploring the housing market data in the form of an interactive visualization dashboard
Simply follow the below steps:

   * Open a terminal window

   * Navigate to the directory : visualization

   * Execute the command to start the server
      If you use Python 2:
      $ python -m SimpleHTTPServer 8000
      If you use Python 3
      $ python -m http.server 8000

   * Open a web browser at http://localhost:8000/

Done! Feel free to play with visualization of multiple zipcodes we predicted in the year of 2022!

 