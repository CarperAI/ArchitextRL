#!/bin/bash

echo "Running build.sh..."
source $HOME/.nvm/nvm.sh
echo "Installing nodejs..."
nvm install node
echo "Building..."
cd frontend
npm install
npm run build