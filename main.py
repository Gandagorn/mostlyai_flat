import os
import random
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

from mostlyai import qa

from pipeline.training import get_synthetic_data
from pipeline.postprocessing import select_rows_with_ipf_and_refinement
from pipeline.utils import calculate_accuracy

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the flat data generation pipeline.")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the input training CSV file.'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with a smaller dataset and fewer iterations.'
    )
    args = parser.parse_args()

    # --- Setup & Configuration ---
    seed_value = 42
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    # Prepare directories and filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pool_dir = "pool_data"
    results_dir = "results"
    os.makedirs(pool_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    pool_file_name = os.path.join(pool_dir, f"flat_intermediate_pre_trained_pool_{timestamp}.csv")
    output_filename = os.path.join(results_dir, f"flat_result_{timestamp}.csv")

    # Load data
    train_df = pd.read_csv(args.data_path)
    start_time = datetime.now()

    model_params = dict(
        model_name='MOSTLY_AI/Large',
        batch_size=1024,
        gradient_accumulation_steps=1,
        enable_flexible_generation=False,
        value_protection=False,
        max_epochs=17,
        max_training_time=60,  # 1 hour
        train_iterations=2,
        train_size=len(train_df),
        sample_size=2_500_000,
    )

    postprocessing_params = dict(
        ipf_top_pairs=5000,
        refinement_iterations=500,
        refinement_top_pairs=500,
        refinement_top_triples=25,
        trimming_data_multiplier=1.25,
        trimming_swapsize=180,
        max_trimming_time=90,
        max_refinement_time=60,
    )

    if args.test:
        print("Testing mode enabled: Using smaller dataset and fewer iterations.")
        model_params["train_size"] = 2000
        model_params["sample_size"] = 10000
        model_params["max_training_time"] = 2
        postprocessing_params["refinement_iterations"] = 50
        postprocessing_params["ipf_top_pairs"] = 500
        postprocessing_params["refinement_top_pairs"] = 50
        postprocessing_params["max_trimming_time"] = 3
        postprocessing_params["max_refinement_time"] = 2

    # Prepare training data based on size parameter
    train_df = train_df.copy(deep=True).iloc[:model_params["train_size"]]
    print(f"Training with {len(train_df)} Train Samples")

    print("--- STEP 1: Generating Synthetic Data Pool ---")
    synthetic_data = get_synthetic_data(train_df=train_df, model_params=model_params)
    synthetic_data.to_csv(pool_file_name, index=False)
    print(f"Synthetic data pool saved to {pool_file_name}")

    print("--- STEP 2: Selecting Best Subset via Post-processing ---")
    synthetic_pool = pd.read_csv(pool_file_name)
    chosen_indices = select_rows_with_ipf_and_refinement(
        train_df=train_df,
        synthetic_data=synthetic_pool,
        **postprocessing_params,
    )
    subset_df = (synthetic_pool
                 .iloc[chosen_indices]
                 .reset_index(drop=True))
    # Match dtypes of training data
    for c in train_df.columns:
        subset_df[c] = subset_df[c].astype(train_df[c].dtype)

    print("--- STEP 3: Final Evaluation ---")
    print("Using Local Validation Metrics")
    before = calculate_accuracy(
        original_data=train_df,
        synthetic_data=synthetic_pool.sample(len(train_df)),
    )
    after = calculate_accuracy(
        original_data=train_df,
        synthetic_data=subset_df,
    )

    print("Accuracy of Initial Pool (Before):")
    for key, value in before.items():
        print(f"  - {key}: {value}")

    print("Accuracy of Refined Subset (After):")
    for key, value in after.items():
        print(f"  - {key}: {value}")

    print("Quick Checks on Final Subset:")
    print(f"  - Size: {len(subset_df)}")
    print(f"  - All unique indices: {subset_df.index.is_unique}")
    print(f"  - All unique data points: {len(subset_df.drop_duplicates()) == len(subset_df)}")

    print("--- STEP 4: Storing Final Result ---")
    subset_df.to_csv(output_filename, index=False)
    print(f"Final output saved to {output_filename}")

    duration = (datetime.now() - start_time).total_seconds() / 3600
    print(f"Pipeline completed successfully in {duration:.2f} hours.")