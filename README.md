#  Iterative Subset Selection from Oversampled SYNTHetic data (ISSOSYNTH)

This approach consists of a multi-stage pipeline that first generates a large, diverse pool of synthetic data and then uses a post-processing strategy to select an optimal subset.

## Pipeline
Here we present the approach to use ISSOSYNTH to create an optimized synthetic dataset from a source dataset.

### 1\. Synthetic Data Pool Generation

To create a rich and diverse foundation for an optimized synthetic dataset, the pipeline first generates a large pool of synthetic data. This is achieved by training a generative TabularARGN model from MOSTLY AI.

### 2\. Post-processing for Subset Selection

Once the data pool is generated, a multi-step selection algorithm carefully chooses the final records.

1.  **IPF Selection**: An initial, oversized subset is selected from the pool using **Iterative Proportional Fitting (IPF)**. This step focuses on accurately matching the bivariate (2-column) distributions of the most statistically significant column pairs.
2.  **Greedy Trimming**: The oversized subset is intelligently trimmed down to the required target size. This is achieved by iteratively identifying and removing the rows that contribute the most to the statistical error (L1 distance) when compared to the original data.
3.  **Iterative Refinement**: The final record set undergoes a refinement process. The algorithm iteratively swaps rows from the subset with "better" rows from the larger data pool to further minimize the univariate, bivariate, and trivariate statistical errors. This step fine-tunes the dataset, polishing it for final submission.

## Key Features

This pipeline was designed to be robust, efficient, and adaptable, making it a strong foundation for various synthetic data generation tasks.

  * **High Generalizability**: The two-stage architecture (generation and post-processing) is highly modular. The post-processing scripts can be easily reused for other projects or datasets with minimal changes. The parameters exposed in `main.py` (e.g., `ipf_top_pairs`, `refinement_iterations`) can tune the trade-off between compute time and quality.

  * **Model Agnostic**: The refinement pipeline works with **any source of synthetic data**. While this project uses the `mostlyai` SDK for the initial pool generation, you could substitute it with any generative model (e.g., GANs, VAEs, or even weaker, faster models). The strength of the final output comes from the post-processing, which can polish and upgrade the quality of any base synthetic dataset.

  * **Performance-Optimized**: By converting data into binned, integer-based representations, the algorithms operate with very low memory overhead. The core computations rely on efficient NumPy operations, avoiding the need to hold multiple large DataFrames in memory.

## Setup & Usage

The entire pipeline is orchestrated by the `run.sh` script, which handles environment setup, dependency installation, and execution.

### 1\. Grant Execute Permissions

First, make the `run.sh` script executable:

```bash
chmod +x run.sh
```

### 2\. Run the Pipeline

Execute the script, providing the full path to the training data CSV file as an argument.

```bash
./run.sh /path/to/your/flat-training.csv
```

The script will:

1.  Check for `uv` and install it if not present.
2.  Create a local virtual environment in a `.venv` directory.
3.  Install all required Python packages from `requirements.txt`.
4.  Activate the virtual environment.
5.  Adapt the parameters for training and post-processing in the `main.py` script (or use the `select_rows_with_ipf_and_refinement` in your own code).
5.  Run the main pipeline script (`main.py`) with the provided data path.

All dependencies are pinned in `requirements.txt` for reproducibility. This file was generated from `requirements.in` using `uv pip compile`.

## Output

The pipeline generates two primary outputs:

  - **Intermediate Data Pool**: A large CSV file containing all generated data points is saved in the `pool_data/` directory.
  - **Final Output**: The final, refined synthetic dataset is saved in the `results/` directory with a timestamped filename, e.g., `flat_result_20250702_1955.csv`.

## License

This project is licensed under the **MIT License**.