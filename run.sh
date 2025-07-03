#!/bin/bash
set -e

TRAINING_DATA_PATH=$1

if [ -z "$TRAINING_DATA_PATH" ]; then
  echo "Error: Please provide the path to the training data CSV as the first argument."
  echo "Usage: ./run.sh /path/to/your/data.csv"
  exit 1
fi

if command -v uv &> /dev/null; then
    echo "uv is already installed."
else
    echo "uv not found, installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.local/bin:$PATH"
    echo "uv has been installed successfully."
fi

if [ -d ".venv" ]; then
    echo "--- Removing existing virtual environment to ensure clean start ---"
    rm -rf .venv
fi

echo "--- Creating Python virtual environment using uv ---"
uv venv

echo "--- Installing dependencies from requirements.txt ---"
uv pip install -r requirements.txt

echo "--- Activating the virtual environment ---"
source .venv/bin/activate

echo "--- Running the main prediction script ---"
shift
python main.py --data_path "$TRAINING_DATA_PATH" "$@"

echo "--- Script finished successfully ---"