from itertools import combinations
from SubsetContainer import SubsetContainer

def get_sub_dfs(df, anomaly):
    """
        Generate and return all possible sub-DataFrames from the input DataFrame by
        considering all combinations of rows and columns.
    """
    sub_dfs = []
    rows = list(range(df.shape[0]))  # List of row indices
    cols = list(df.columns)  # List of column names

    # Generate all combinations of rows
    for r in range(1, len(rows) + 1):
        for row_comb in combinations(rows, r):
            # Generate all combinations of columns
            for c in range(1, len(cols) + 1):
                for col_comb in combinations(cols, c):
                    sub_df = df.iloc[list(row_comb), list(df.columns.get_indexer(col_comb))]    # Create sub-DataFrame
                    subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, features=col_comb)
                    sub_dfs.append(subset_container)

    return sub_dfs

