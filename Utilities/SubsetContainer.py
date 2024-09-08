import numpy as np
import math


class SubsetContainer:
    def __init__(self, subset, anomaly=None, features=None):
        self.subset = subset
        self.euclidian_distance = self.calc_euclidian_distance(anomaly, features)

    def get_subset(self):
        return self.subset

    def get_euclidian_distance(self):
        return self.euclidian_distance

    def set_euclidian_distance(self, euclidian_distance):
        self.euclidian_distance = euclidian_distance

    def calc_euclidian_distance(self, anomaly, features):
        """
         Euclidean distances between D_prime and a sample (s) based on the specified features (features).
        """
        if anomaly is None:
            return None
        features = list(features)
        euclidian_distance = (np.linalg.norm(self.subset[features] - anomaly[features])) / len(features)
        return euclidian_distance


    def calc_subset_entropy(self):
        subset_entropy = 0
        subset_size = len(self.subset)

        for _, row in self.subset.iterrows():
            # Convert row to a tuple for caching
            row_key = tuple(row.items())
            if row_key in self.entropy_cache:
                subset_entropy += self.entropy_cache[row_key]
            else:
                row_entropy = 0
                for col in self.df:
                    cur_prob = self.calc_prob_of_val(col, row[col])
                    if cur_prob > 0:
                        row_entropy += cur_prob * math.log(cur_prob, 2)
                row_entropy = -row_entropy
                self.entropy_cache[row_key] = row_entropy
                subset_entropy += row_entropy

        # Return the average entropy for the subset
        return subset_entropy / subset_size if subset_size > 0 else 0








