from Utilities.SubsetContainer import SubsetContainer
import pandas as pd
from Utilities import Constants as util
import random
from datetime import datetime


def get_sub_dfs(df, anomaly, top_n=10, num_samples=1000):
    """
    Generate and return the top N sub-DataFrames from the input DataFrame by
    randomly sampling subsets of rows and columns. The top N subsets with
    the lowest similarity method values are kept in memory.

    Args:
        df (pd.DataFrame): The input DataFrame.
        anomaly (pd.Series): The anomaly instance to compare against.
        top_n (int): The number of top subsets to return.
        num_samples (int): The number of Monte Carlo samples to generate.

    Returns:
        list: A list of the top N SubsetContainers with the lowest similarity values.
    """

    rows = list(range(df.shape[0]))  # List of row indices
    cols = list(df.columns)  # List of column names
    top_subsets = []  # List to store the top N subsets

    for i in range(num_samples):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"Sample {i + 1}/{num_samples}, Current Time: {current_time}")

        # Randomly select the number of rows and columns to sample
        r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
        c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))

        # Randomly sample a combination of rows and columns
        row_comb = random.sample(rows, r)
        col_comb = random.sample(cols, c)

        # Create the sub-DataFrame
        sub_df = df.iloc[list(row_comb), list(df.columns.get_indexer(col_comb))]
        subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, features=col_comb)

        # Check if all selected features are identical to the anomaly's corresponding feature values
        if all((sub_df[col] == anomaly[col]).all() for col in col_comb):
            continue  # Skip this subset

        euclidian_distance = subset_container.get_euclidian_distance()

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
