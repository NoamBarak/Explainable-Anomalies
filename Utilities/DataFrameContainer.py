import pandas as pd
import os
from Utilities import Constants as constants

class DataFrameContainer:
    def __init__(self):
        file_path = constants.PROJECT_PATH + "\Data\house_prices_test.csv"
        df = pd.read_csv(file_path)
        selected_columns = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
                            'YearBuilt', 'YearRemodAdd', 'GarageCars', 'SalePrice']
        data = df[selected_columns]

        data = pd.DataFrame(data)
        self.data = data
        self.rows_amount = data.shape[0]
        self.cols_amount = data.shape[1]
        self.cols_names = data.columns
        self.entropy_cache = {}  # Initialize a cache for entropy calculations
        self.prob_cache = {}  # Initialize a cache for probability calculations

    def get_df(self):
        return self.df

    def calc_prob_of_val(self, col_name, val):
        """
        Calculate the probability of a specific value occurring in a specified column.
        """
        # Check if the probability for this column and value pair is already cached
        cache_key = (col_name, val)
        if cache_key in self.prob_cache:
            return self.prob_cache[cache_key]

        count_of_val = len(self.df[self.df[col_name] == val])
        prob = count_of_val / self.df_rows_amount

        self.prob_cache[cache_key] = prob   # Store the calculated probability in the cache

        return prob
