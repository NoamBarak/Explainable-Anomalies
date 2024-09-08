MIN_ROWS_AMOUNT = 3
MAX_ROWS_AMOUNT = 5

MIN_COLS_AMOUNT = 4
MAX_COLS_AMOUNT = 7

subsets_files_names = {
    "Brute Force": "brute_force_subsets.csv",
    "Genetic": "genetic_subsets.csv",
    "Multi Arm-Bandit":"mab_subsets.csv"
}

def write_single_subset_to_file(subset_info, file, cols_names):
    """
    Write a single subset DataFrame to a CSV file.
    """
    subset = subset_info.subset
    sim_value = subset_info.sim_value
    subset = subset.reindex(columns=cols_names, fill_value='')  # Ensure all columns exist
    subset['Similarity'] = sim_value

    # Write the subset to the file
    subset.to_csv(file, index=False, header=False)