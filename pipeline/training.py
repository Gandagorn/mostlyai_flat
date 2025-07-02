import pandas as pd
from mostlyai.sdk import MostlyAI
from .utils import calculate_accuracy


def train_generator(
        data: pd.DataFrame,
        model_params,
        mostly,
):
    """Trains a Mostly AI generator on the provided data.

    This function configures and initiates a training run using the Mostly AI
    SDK. It creates an oversampled dataset to improve model performance.

    Args:
        data: The input DataFrame to train the model on.
        model_params: A dictionary containing model and training configuration,
                      such as model_name, batch_size, max_epochs, etc.
        mostly: An initialized MostlyAI SDK client instance.

    Returns:
        A trained Mostly AI generator object.
    """
    oversample_data = pd.concat([data, data], ignore_index=True)
    g = mostly.train(config={
        'tables': [{
            'name': 'Baseline',
            'data': oversample_data,
            'tabular_model_configuration': {
                'model': model_params["model_name"],
                'batch_size': model_params["batch_size"],
                'gradient_accumulation_steps': model_params["gradient_accumulation_steps"],
                'enable_flexible_generation': model_params["enable_flexible_generation"],
                'value_protection': model_params["value_protection"],
                'max_epochs': model_params["max_epochs"],
                'max_training_time': model_params["max_training_time"],
            }
        }]
    })
    return g


def get_synthetic_data(train_df, model_params):
    """Generates a pool of synthetic data by training and sampling multiple models.

    This function orchestrates the data generation process. It can run
    multiple training iterations. In each iteration, it trains a new generator
    and samples from it. The results from all iterations are concatenated to
    form a large, diverse pool of synthetic data. It also adds helper features
    for missing values to improve performance.

    Args:
        train_df: The original training DataFrame.
        model_params: A dictionary containing parameters for training and
                      sampling, such as train_iterations, sample_size, etc.

    Returns:
        A DataFrame containing the combined pool of generated synthetic data.
    """
    mostly = MostlyAI(local=True)
    synthetic_data_list = []
    iterations = model_params["train_iterations"]

    train_df_extended = train_df.copy(deep=True)
    na_columns = train_df_extended.columns[train_df_extended.isna().sum(axis=0) > 0].to_list()
    print(f"Add NA features for: {na_columns}")
    for na_col in na_columns:
        na_tmp_col = f"is_na_{na_col}"
        train_df_extended[na_tmp_col] = train_df_extended[na_col].isna().astype(str)

    for i in range(iterations):
        print(f"Starting Training Iteration {i+1}/{iterations}")

        g = train_generator(
            data=train_df_extended,
            model_params=model_params,
            mostly=mostly
        )

        print(f"Generating data with generator from iteration {i+1}")
        sd = mostly.generate(
            g,
            config={
                'tables': [{
                    'name': 'Baseline',
                    'configuration': {
                        "sample_size": model_params["sample_size"] // iterations,
                        "sampling_temperature": 1.05
                    }
                }]
            }
        )
        synthetic_data_it = sd.data()

        accuracy_result = calculate_accuracy(train_df, synthetic_data_it.sample(model_params['train_size']))
        print(f"Accuracy for iteration {i+1}: {accuracy_result}")
        synthetic_data_list.append(synthetic_data_it)

    print("Combining data from all iterations into the final pool.")
    synthetic_data = pd.concat(synthetic_data_list)
    synthetic_data = synthetic_data[train_df.columns]
    accuracy = calculate_accuracy(train_df, synthetic_data.sample(model_params['train_size']))
    print(f"Final combined data pool accuracy: {accuracy}")
    return synthetic_data