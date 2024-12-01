#!/bin/csh

set execute_pyfile_name = 'main'

# check the current directory
set current_dir = `pwd`
echo "current directory: $current_dir"

# Install Python virtual environment (create it if the venv directory does not exist)
if (! -d "$current_dir/venv") then
    echo "creating virtual environment."
    python -m venv "$current_dir/venv"
endif

# Activate the virtual environment
source "$current_dir/venv/bin/activate.csh"
echo "virtual environment activated."

# Check if requirements.txt exists and install the libraries
if (-f "$current_dir/requirements.txt") then
    echo "installing libraries from requirements.txt."
    pip install -r "$current_dir/requirements.txt"
else
    echo "requirements.txt file not found."
endif