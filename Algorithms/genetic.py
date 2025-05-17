import random
import torch
from Utilities.SubsetContainer import SubsetContainer
import Utilities.Constants as constants
from Utilities import Constants as util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_population(df, anomaly, population_size):
    """
    Create an initial population of subsets (each wrapped in SubsetContainer)
    with exactly util.MIN_ROWS_AMOUNT rows.
    """
    population = []
    rows = list(range(df.shape[0]))
    cols = list(df.columns)

    for _ in range(population_size):
        # Determine number of rows (and columns for sim_features) to select
        if len(rows) < util.MIN_ROWS_AMOUNT:
            row_count = len(rows)
        else:
            row_count = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
        if len(cols) < util.MIN_COLS_AMOUNT:
            col_count = len(cols)
        else:
            col_count = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))

        # Randomly select the specified number of rows
        selected_rows = random.sample(rows, row_count)
        sub_df = df.iloc[selected_rows]  # subset DataFrame with those rows (all columns)
        subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, sim_features_amount=col_count)
        fitness = subset_container.get_euclidian_distance()
        population.append((fitness, subset_container))

    # Sort initial population by fitness (Euclidean distance)
    return sorted(population, key=lambda x: x[0])

def mutate(subset_container, df, anomaly):
    """
    Mutate a subset by replacing one row with a new row, keeping subset size constant.
    """
    subset_df = subset_container.subset.copy()
    sim_features_amount = len(subset_container.sim_features)

    # Determine the set of row indices not currently in the subset (candidates for addition)
    current_indices = set(subset_df.index)
    all_indices = set(df.index)
    outside_indices = list(all_indices - current_indices)

    # With 50% chance, perform a mutation if a new row is available
    if outside_indices and random.random() < 0.5:
        # Remove one random row from the subset
        drop_idx = random.choice(list(current_indices))
        subset_df = subset_df.drop(index=drop_idx)
        # Add one random row from the outside pool
        add_idx = random.choice(outside_indices)
        subset_df.loc[add_idx] = df.loc[add_idx]

    # Wrap the mutated (or unmodified) subset in a new SubsetContainer
    mutated_container = SubsetContainer(subset=subset_df, anomaly=anomaly, sim_features_amount=sim_features_amount)
    fitness = mutated_container.get_euclidian_distance()
    return (fitness, mutated_container)

def crossover(parent1, parent2, df, anomaly):
    """
    Crossover two parent subsets to produce a child subset (3 rows).
    Ensures the child has exactly util.MIN_ROWS_AMOUNT rows by combining parents' row indices.
    """
    # Extract the row index sets of the parents
    parent1_indices = set(parent1.subset.index)
    parent2_indices = set(parent2.subset.index)

    # If parents are identical in row composition, return a mutation of one (to introduce variation)
    if parent1_indices == parent2_indices:
        return mutate(parent1, df, anomaly)

    # Determine common and total unique indices from both parents
    common_indices = parent1_indices & parent2_indices
    union_indices = parent1_indices | parent2_indices
    child_indices = set()

    # Case 1: If there are 2 common rows (since target is 3), use them and add one more unique row
    if len(common_indices) == util.MIN_ROWS_AMOUNT - 1:  # 2 common rows in this context
        child_indices |= common_indices  # include the two common rows
        # Add one of the remaining unique rows (from either parent) to make 3
        remaining = list(union_indices - common_indices)
        if remaining:
            child_indices.add(random.choice(remaining))
    # Case 2: If there is exactly 1 common row, include it and one unique from each parent
    elif len(common_indices) == 1:
        child_indices |= common_indices  # include the single common row
        unique1 = list(parent1_indices - common_indices)
        unique2 = list(parent2_indices - common_indices)
        if unique1:
            child_indices.add(random.choice(unique1))
        if unique2 and len(child_indices) < util.MIN_ROWS_AMOUNT:
            child_indices.add(random.choice(unique2))
        # If still less than 3 (e.g., one parent had no unique rows left), fill from remaining union
        while len(child_indices) < util.MIN_ROWS_AMOUNT:
            remaining = list(union_indices - child_indices)
            if not remaining:
                break
            child_indices.add(random.choice(remaining))
    # Case 3: No common rows, take at least one from each parent and then one more from either
    else:  # len(common_indices) == 0 in this scenario (no overlap)
        # Take one random row from each parent
        pick1 = random.choice(list(parent1_indices))
        pick2 = random.choice(list(parent2_indices))
        child_indices.add(pick1)
        child_indices.add(pick2)
        # Add a third row from the remaining pool of unique indices
        remaining = list(union_indices - child_indices)
        if remaining and len(child_indices) < util.MIN_ROWS_AMOUNT:
            child_indices.add(random.choice(remaining))

    # Defensive check: ensure we have exactly 3 indices (in case of any unforeseen scenario)
    if len(child_indices) > util.MAX_ROWS_AMOUNT:
        child_indices = set(list(child_indices)[:util.MAX_ROWS_AMOUNT])
    elif len(child_indices) < util.MIN_ROWS_AMOUNT:
        # If we have fewer than 3 (which is unlikely), fill with random choices from union_indices
        for idx in list(union_indices):
            if len(child_indices) < util.MIN_ROWS_AMOUNT:
                child_indices.add(idx)
            else:
                break

    # Create the child subset DataFrame from the selected indices (include all columns)
    child_df = df.loc[sorted(child_indices)]
    child_container = SubsetContainer(subset=child_df, anomaly=anomaly, sim_features_amount=len(parent1.sim_features))
    fitness = child_container.get_euclidian_distance()
    return (fitness, child_container)

def get_sub_dfs(df, anomaly, top_n=constants.SUBSETS_AMOUNT, population_size=50, generations=100):
    """
    Run the genetic algorithm to find counter-example subsets.
    Returns a list of SubsetContainer instances (each with 3 rows) sorted by Euclidean distance.
    """
    # Initial population of subsets
    population = initialize_population(df, anomaly, population_size)

    for _ in range(generations):
        # Select the top-performing half of the population to be parents (elitism)
        population = sorted(population, key=lambda x: x[0])[: max(1, population_size // 2)]
        new_population = []

        # Generate new offspring via crossover, and occasionally mutate one parent
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            # Perform crossover to produce a child
            child = crossover(parent1[1], parent2[1], df, anomaly)
            new_population.append(child)
            # Perform mutation on one parent with a small probability (to introduce new variation)
            if random.random() < 0.2:
                mutant = mutate(parent1[1], df, anomaly)
                new_population.append(mutant)

        # Prepare for next generation: take the top individuals from new_population as the new population
        population = sorted(new_population, key=lambda x: x[0])
        if len(population) > population_size:
            population = population[:population_size]
        # (Duplicates are implicitly allowed; population size remains constant to avoid sampling errors)

    # Sort final population by fitness and return the top N SubsetContainer results
    population = sorted(population, key=lambda x: x[0])
    return [subset for _, subset in population[:top_n]]
