import pandas as pd
import math
from functools import lru_cache

class DataFrameContainer:
    def __init__(self):
        data = {
            'Column1': [1, 2],
            'Column2': [4, 5],
        }

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

    def calc_row_entropy(self, given_row):
        # Check if the entropy for this row is already cached
        row_key = tuple(given_row.items())
        if row_key in self.entropy_cache:
            return self.entropy_cache[row_key]

        res = 0
        for col in self.df:
            cur_prob = self.calc_prob_of_val(col, given_row[col])
            res = res + (cur_prob * math.log(cur_prob, 2))

        self.entropy_cache[row_key] = -res  # Store the calculated entropy in the cache
        return -res











