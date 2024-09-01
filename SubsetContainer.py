import numpy as np

class SubsetContainer:
    def __init__(self, subset, anomaly=None, features=None):
        self.subset = subset
        self.sim_value = self.calc_similarity_value(anomaly, features)

    def get_subset(self):
        return self.subset

    def get_sim_value(self):
        return self.sim_value

    def set_sim_value(self, sim_value):
        self.sim_value = sim_value

    def calc_similarity_value(self, anomaly, features):
        """
         Euclidean distances between D_prime and a sample (s) based on the specified features (features).
        """
        if anomaly is None:
            return None
        features = list(features)
        sim_value = (np.linalg.norm(self.subset[features] - anomaly[features])) / len(features)
        return sim_value











