#!/bin/bash

# Copy the SSH key to the .ssh directory
cp ~/nicko-sandbox/saved_keys/id_rsa ~/.ssh/
cp ~/nicko-sandbox/.bashrc ~/.bashrc

# Set global Git email
git config --global user.email "ncorriveau13@gmail.com"

# Set global Git name
git config --global user.name "Nicko Corriveau"