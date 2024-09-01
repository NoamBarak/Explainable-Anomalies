from DataFrameContainer import DataFrameContainer
import brute_force
import pandas as pd
import csv

def write_to_file(subsets, output_filename, cols_names):
    all_data = []

    for idx, subset_info in enumerate(subsets):
        subset = subset_info.subset
        sim_value = subset_info.sim_value
        subset = subset.reindex(columns=cols_names, fill_value='')  # Ensure all columns exist
        subset['Similarity'] = sim_value

        all_data.append(subset)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Write the combined dataframe to CSV
    combined_df.to_csv(output_filename, index=False, header=True)



subsets = {
    "Brute Force": None,
    "Genetic": None,
    "Multi Arm-Bandit":None
}

subsets_files_names= {
    "Brute Force": "brute_force_subsets.csv",
    "Genetic": "genetic_subsets.csv",
    "Multi Arm-Bandit":"mab_subsets.csv"
}



dataframe_container = DataFrameContainer()
data = dataframe_container.data
cols_names = dataframe_container.cols_names
anomaly = data.iloc[1]
print(f"Anomaly:{anomaly}\n\n")
subsets["Brute Force"] = brute_force.get_sub_dfs(data, anomaly)
write_to_file(subsets["Brute Force"], subsets_files_names["Brute Force"], cols_names)
for subset in subsets["Brute Force"]:
    print(f"Subset:{subset.get_subset()}\nSim:{subset.get_sim_value()}\n\n")
