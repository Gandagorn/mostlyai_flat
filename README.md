# The MOSTLY AI Prize: Flat Data Challenge Submission

This repository contains the source code for a submission to "The MOSTLY AI Prize" flat data challenge.

This solution consists of a multi-stage pipeline that first generates a large, diverse pool of synthetic data and then uses a post-processing strategy to select an optimal subset.

## Methodology


### 1\. Synthetic Data Pool Generation

To create a rich and diverse foundation for the final dataset, the pipeline first generates a large pool of synthetic data. This is achieved by training a generative model from MOSTLY AI.

Instead of relying on a single training run, the pipeline iteratively trains the model multiple times. Each training cycle produces a new, distinct generator that contributes its own unique synthetic samples to the overall data pool. To further improve the quality of the generated data, the model is also trained on engineered features that explicitly represent patterns of missing values.

### 2\. Post-processing for Subset Selection

Once the data pool is generated, a multi-step selection algorithm carefully chooses the final 100,000 records. This is where the "magic" happens, refining the raw generated data into a high-quality, competitive submission.

1.  **IPF Selection**: An initial, oversized subset is selected from the pool using **Iterative Proportional Fitting (IPF)**. This step focuses on accurately matching the bivariate (2-column) distributions of the most statistically significant column pairs.
2.  **Greedy Trimming**: The oversized subset is intelligently trimmed down to the required target size (100,000 records). This is achieved by iteratively identifying and removing the rows that contribute the most to the statistical error (L1 distance) when compared to the original data.
3.  **Iterative Refinement**: The final 100,000-record set undergoes a refinement process. The algorithm iteratively swaps rows from the subset with "better" rows from the larger data pool to further minimize the univariate, bivariate, and trivariate statistical errors. This step fine-tunes the dataset, polishing it for final submission.

## Key Features

This pipeline was designed to be robust, efficient, and adaptable, making it a strong foundation for various synthetic data generation tasks.

  * **High Generalizability**: The two-stage architecture (generation and post-processing) is highly modular. The post-processing scripts can be easily reused for other projects or datasets with minimal changes. The parameters exposed in `main.py` (e.g., `ipf_top_pairs`, `refinement_iterations`) can tune the trade-off between compute time and quality.

  * **Model Agnostic**: The refinement pipeline works with **any source of synthetic data**. While this project uses the `mostlyai` SDK for the initial pool generation, you could substitute it with any generative model (e.g., GANs, VAEs, or even weaker, faster models). The strength of the final output comes from the post-processing, which can polish and upgrade the quality of any base synthetic dataset.

  * **Performance-Optimized**: The entire post-processing pipeline is engineered for efficiency:

      * **Low Memory Footprint**: By converting data into binned, integer-based representations, the algorithms operate with very low memory overhead. The core computations rely on efficient NumPy operations, avoiding the need to hold multiple large DataFrames in memory.
      * **CPU-Friendly Refinement**: While the initial model training benefits from a GPU, the post-processing does not need a GPU.
      * **Parallelizable**: The core refinement logic—evaluating row swaps can be parallelized for .

  * **Enhanced Privacy Guarantees**: By selecting from a vast, pre-generated pool based on aggregate statistical distributions (up to the trivariate level), the risk of replicated individual records or their sensitive combinations is minimal. The final dataset copies the *statistical patterns* of the original data, not the data points themselves, ensuring privacy-safe output.

## System Requirements

  - **Python**: 3.10+
  - **GPU Environment**: This submission should run with a GPU in an **`g5.2xlarge`** instance.
  - **AWS AMI**: The `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)` was used for testing this solution.
  - **NVIDIA Driver**: Before running, ensure the NVIDIA driver is correctly installed by executing `nvidia-smi`.

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

1.  Check for `uv` (a fast Python package installer) and install it if not present.
2.  Create a local virtual environment in a `.venv` directory.
3.  Install all required Python packages from `requirements.txt`.
4.  Activate the virtual environment.
5.  Run the main pipeline script (`main.py`) with the provided data path.

## Output

The pipeline generates two primary outputs:

  - **Intermediate Data Pool**: A large CSV file containing all generated data points is saved in the `pool_data/` directory.
  - **Final Submission File**: The final, refined synthetic dataset is saved in the `results/` directory with a timestamped filename, e.g., `flat_result_20250702_1955.csv`. This is the file that should be used for evaluation.

## License

This project is licensed under the **MIT License**.