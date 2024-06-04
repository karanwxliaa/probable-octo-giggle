#!/bin/bash

#set -x

echo "You provided the arguments:" "$@"
echo "You provided $# arguments"

# Define the path for the virtual environment
VENV_PATH="./venv"

# # Create the virtual environment if it doesn't exist
# if [ ! -d "$VENV_PATH" ]; then
#     echo "Creating virtual environment at $VENV_PATH"
#     python3 -m venv $VENV_PATH
# fi

echo "Activating Virtual Environment"
# Activate the virtual environment
source $VENV_PATH/bin/activate

# python3 main.py --gpuid $2 --model_type "$3" --model_name "$4" 
python3 main.py --gpuid "$1" --model_type "$2" --model_name "$3" 

echo "Deactivating Virtual Environment"
deactivate