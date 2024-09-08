from DataFrameContainer import DataFrameContainer
import brute_force
import pandas as pd
import csv
import torch
import Utilities as util

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(torch.cuda.current_device()))


top_10_subsets = {
    "Brute Force": None,
    "Monte Carlo": None,
    "Genetic": None,
    "Multi Arm-Bandit": None
}

def print_top_subsets(top_10_subsets, output_filename):
    with open(output_filename, 'w') as file:
        for i, subset_container in enumerate(top_10_subsets):
            # Assuming the SubsetContainer has 'euclidian_distance' and 'subset' attributes
            sub_euclidian_dis = subset_container.euclidian_distance
            sub_df = subset_container.subset

            # Create the output text
            output_text = (f"Subset {i+1}:\n"
                           f"Similarity Value: {sub_euclidian_dis}\n"
                           "Subset DataFrame:\n"
                           f"{sub_df}\n"
                           + "-" * 50 + "\n")

            # Print to console
            print(output_text)

            # Write to file
            file.write(output_text)


dataframe_container = DataFrameContainer()
data = dataframe_container.data
cols_names = dataframe_container.cols_names
anomaly = data.iloc[2]
print(f"Anomaly:{anomaly}\n\n")
top_10_subsets["Brute Force"] = brute_force.get_sub_dfs(data, anomaly)
print_top_subsets(top_10_subsets["Brute Force"], "brute_force_top_subsets")