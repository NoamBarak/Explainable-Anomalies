from Utilities.SubsetContainer import SubsetContainer
from Utilities import Constants as util
import random
import numpy as np
from datetime import datetime

EPSILON = 0.3
class MultiArmedBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)  # Number of times each arm has been pulled
        self.values = np.zeros(n_arms)  # Estimated value of each arm

    def select_arm(self):
        """Select an arm based on epsilon-greedy strategy."""
        if random.random() > EPSILON:
            # Exploit: Choose the arm with the highest estimated value
            return np.argmax(self.values)
        else:
            # Explore: Randomly select an arm
            return random.randint(0, self.n_arms - 1)

    def update(self, chosen_arm, reward):
        """Update the estimated value of the chosen arm."""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = value + (reward - value) / n
        self.values[chosen_arm] = new_value


def get_sub_dfs(df, anomaly, top_n=10, num_iterations=1000):
    """
    Generate and return the top N sub-DataFrames from the input DataFrame by
    using a Multi-Armed Bandit approach to explore subsets of rows and columns.
    The top N subsets with the lowest similarity method values are kept in memory.

    Args:
        df (pd.DataFrame): The input DataFrame.
        anomaly (pd.Series): The anomaly instance to compare against.
        top_n (int): The number of top subsets to return.
        num_iterations (int): The number of iterations to run the bandit algorithm.
        epsilon (float): The probability of exploring a random arm (epsilon-greedy strategy).

    Returns:
        list: A list of the top N SubsetContainers with the lowest similarity values.
    """

    rows = list(range(df.shape[0]))  # List of row indices
    cols = list(df.columns)  # List of column names
    num_arms = len(rows) * len(cols)  # Total number of possible row-column combinations
    bandit = MultiArmedBandit(num_arms)
    top_subsets = []  # List to store the top N subsets

    for i in range(num_iterations):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"Iteration {i + 1}/{num_iterations}, Current Time: {current_time}")

        # Select an arm (row-column subset) using the epsilon-greedy strategy
        chosen_arm = bandit.select_arm()

        # Derive row and column combinations from the chosen arm index
        r = chosen_arm // len(cols) + util.MIN_ROWS_AMOUNT  # Rows
        c = chosen_arm % len(cols) + util.MIN_COLS_AMOUNT  # Columns

        # Randomly sample the subset of rows and columns
        row_comb = random.sample(rows, min(r, len(rows)))
        col_comb = random.sample(cols, min(c, len(cols)))

        # Create the sub-DataFrame
        sub_df = df.iloc[list(row_comb), list(df.columns.get_indexer(col_comb))]
        subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, features=col_comb)

        # Check if all selected features are identical to the anomaly's corresponding feature values
        if all((sub_df[col] == anomaly[col]).all() for col in col_comb):
            continue  # Skip this subset

        # Calculate the Euclidean distance
        euclidian_distance = subset_container.get_euclidian_distance()
        reward = -euclidian_distance  # Reward is the negative distance

        # Update the Multi-Armed Bandit with the reward
        bandit.update(chosen_arm, reward)

        # If we have fewer than top_n subsets, just add the new one
        if len(top_subsets) < top_n:
            top_subsets.append((euclidian_distance, subset_container))
            top_subsets.sort(key=lambda x: x[0])  # Sort by similarity value
        else:
            # If the new subset has a lower similarity value, replace the worst one
            if euclidian_distance < top_subsets[-1][0]:
                top_subsets[-1] = (euclidian_distance, subset_container)
                top_subsets.sort(key=lambda x: x[0])  # Re-sort the list

    return [subset for _, subset in top_subsets]
