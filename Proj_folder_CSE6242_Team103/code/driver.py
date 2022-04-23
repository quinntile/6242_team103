from PreProcessing_austin import *
from PreProcessing_seattle import *




current_dir = os.path.dirname(__file__)
austin_data = PreProcess_austin(current_dir[:-5] + '/data/Austin housing market data/clean_austin_price_reg_train.csv')
print("Data has the following columns: ",austin_data.get_cols(), "\n")
print("And we have ",austin_data.get_num_rows(), "rows. ")
print("The first 5 rows of the training data is like", austin_data.data.head())
print("Reading all features in the data folder......")
austin_data.read_features()
print("Done!")


austin_data.clean_features()
print("Joining all features into unseen_data......")
austin_data_unseen = austin_data.join_unseen_features()
print("Done!")

print("Saving processed data into folders......")
austin_data_unseen.to_csv(current_dir[:-5] + '/data/data_unseen_austin.csv')
austin_data.training_austin.to_csv(current_dir[:-5] + '/data/training_austin.csv')
print("Done!")


