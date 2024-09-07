#!/bin/bash

# Define installation directory (update as needed)
INSTALL_DIR=$HOME/python3.11

# Download Python 3.11.8
PYTHON_VERSION=3.11.8
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz

# Extract and install Python 3.11.8 locally
tar -xf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION
./configure --prefix=$INSTALL_DIR --enable-optimizations
make -j$(nproc)
make altinstall

# Add the local Python installation to the PATH
export PATH=$INSTALL_DIR/bin:$PATH

# Verify the installation
python3.11 --version

# Install pip for Python 3.11
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Verify pip installation
pip3.11 --version

# Navigate to the directory where requirements.txt is located (update path as needed)
# cd /path/to/your/project

# Install dependencies from requirements.txt into a virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt