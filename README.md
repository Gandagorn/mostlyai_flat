# The MOSTLY AI Prize: Flat Data Generation Pipeline

This repository contains a solution for generating synthetic flat data. 

## Requirements

All required Python packages are listed in the `requirements.txt` file. The `run.sh` script will automatically create a virtual environment and install them using `uv`.

- Python 3.10+
- `uv` (will be installed by the run script if not present)

## Usage

The entire pipeline can be executed using the provided shell script. It handles environment setup, dependency installation, and running the main Python script.

1.  **Make the script executable:**

    ```bash
    chmod +x run.sh
    ```

2.  **Run the script:**
    Provide the full path to the training data CSV file as the first argument.

    ```bash
    ./run.sh /path/to/your/training-data.csv
    ```

    For example:

    ```bash
    ./run.sh ./data/flat-training.csv
    ```

The final synthetic dataset will be saved as `flat_result_{timestamp}.csv` in the `results/` directory.

## Pipeline Overview

1.  **Training**: A generative model is trained on the input data. To increase the diversity of the generated data, multiple training iterations can be configured, with the results concatenated into a large data pool.
2.  **Post-processing (IPF + Refinement)**:
    * **IPF Selection**: An initial, larger-than-needed subset is selected from the pool using Iterative Proportional Fitting (IPF) on the most significant column pairs. This ensures the bivariate distributions are well-represented.
    * **Trimming**: The IPF-selected subset is trimmed down to the final target size using a refinement algorithm that removes rows that contribute most to statistical error.
    * **Refinement**: The final set is further refined by swapping rows with others from the pool to minimize the combined univariate, bivariate, and trivariate statistical error.