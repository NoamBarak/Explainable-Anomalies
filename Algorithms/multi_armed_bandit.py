from Utilities.SubsetContainer import SubsetContainer
from Utilities import Constants as util
import random
from datetime import datetime
from collections import defaultdict



def get_sub_dfs(df, anomaly, top_n=10, num_samples=1000, epsilon=0.1):
    """
    Generate and return the top N sub-DataFrames from the input DataFrame using
    a Multi-Armed Bandit approach.

    Args:
        df (pd.DataFrame): The input DataFrame.
        anomaly (pd.Series): The anomaly instance to compare against.
        top_n (int): The number of top subsets to return.
        num_samples (int): The number of bandit iterations.
        epsilon (float): The exploration rate for the epsilon-greedy strategy.

    Returns:
        list: A list of the top N SubsetContainers with the lowest similarity values.
    """

    rows = list(range(df.shape[0]))  # List of row indices
    cols = list(df.columns)  # List of column names
    top_subsets = []  # List to store the top N subsets
    seen_combinations = set()  # Set to keep track of processed combinations

    # Initialize the rewards for each possible combination
    arm_rewards = defaultdict(lambda: 0)

    for i in range(num_samples):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"Sample {i + 1}/{num_samples}, Current Time: {current_time}")

        # Epsilon-Greedy Strategy
        if random.random() < epsilon:
            # Exploration: Randomly sample a new combination
            r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
            c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))

            row_comb = tuple(sorted(random.sample(rows, r)))
            col_comb = tuple(sorted(random.sample(cols, c)))
        else:
            # Exploitation: Select the best-known arm (min reward)
            best_arm = max(arm_rewards, key=arm_rewards.get, default=None)
            if best_arm:
                row_comb, col_comb = best_arm
            else:
                # If no arms have been tried yet, default to random sampling
                r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
                c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))

                row_comb = tuple(sorted(random.sample(rows, r)))
                col_comb = tuple(sorted(random.sample(cols, c)))

        # Skip already seen combinations
        if (row_comb, col_comb) in seen_combinations:
            continue

        # Mark the combination as seen
        seen_combinations.add((row_comb, col_comb))

        # Create the sub-DataFrame
        sub_df = df.iloc[list(row_comb), list(df.columns.get_indexer(col_comb))]
        subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, features=col_comb)

        # Check if all selected features are identical to the anomaly's corresponding feature values
        if all((sub_df[col] == anomaly[col]).all() for col in col_comb):
            continue  # Skip this subset

        # Calculate the Euclidean distance (reward is inverse of distance)
        euclidean_distance = subset_container.get_euclidian_distance()
        reward = -euclidean_distance  # Inverse to make it a maximization problem

        # Update the reward for the selected arm
        arm_rewards[(row_comb, col_comb)] = reward

        # If we have fewer than top_n subsets, just add the new one
        if len(top_subsets) < top_n:
            top_subsets.append((euclidean_distance, subset_container))
            top_subsets.sort(key=lambda x: x[0])  # Sort by similarity value
        else:
            # If the new subset has a lower similarity value, replace the worst one
            if euclidean_distance < top_subsets[-1][0]:
                top_subsets[-1] = (euclidean_distance, subset_container)
                top_subsets.sort(key=lambda x: x[0])  # Re-sort the list

    return [subset for _, subset in top_subsets]