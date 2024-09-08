from Utilities.DataFrameContainer import DataFrameContainer
from Utilities import Constants as constants
from Algorithms import brute_force, monte_carlo, multi_armed_bandit as mab

# Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.get_device_name(torch.cuda.current_device()))

best_subsets = {
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

best_subsets["Brute Force"] = brute_force.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT)
num_samples = dataframe_container.rows_amount * dataframe_container.cols_amount
best_subsets["Monte Carlo"] = monte_carlo.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT, num_samples=num_samples)
best_subsets["Multi Arm-Bandit"] = mab.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT, num_iterations=num_samples)

print_top_subsets(best_subsets["Brute Force"], "Results/brute_force_top_subsets")
print_top_subsets(best_subsets["Monte Carlo"], "Results/monte_carlo_top_subsets")
print_top_subsets(best_subsets["Multi Arm-Bandit"], "Results/mab_top_subsets")