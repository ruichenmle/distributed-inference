#!/bin/bash

S3_MODEL_PATH="s3://mosaicml-68c98fa5-0b21-4c7b-b40b-c4482db8832a/sut-extraction/ruichen/sut-final-model/"
S3_FILE_PATH="s3://mosaicml-68c98fa5-0b21-4c7b-b40b-c4482db8832a/sut-extraction/ruichen/sut-combined-data/new_test_fixed.tsv"
LOCAL_MODEL_PATH="/mnt/ruichen/distributed-inference/origin-model/"
LOCAL_FILE_PATH="/mnt/ruichen/distributed-inference/new_test_fixed.tsv"

DOWNLOAD_SUCCESS=false

# Check if the AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI not found. Installing..."
    sudo apt-get install unzip
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    sudo ./aws/install
    echo "AWS CLI installed."
else
    echo "AWS CLI already exists."
fi

# Print AWS CLI version
aws --version

# Check if the folder or file already exists locally
if [ -d "$LOCAL_MODEL_PATH" ] || [ -f "$LOCAL_MODEL_PATH" ]; then
    echo "Folder or file already exists."
    DOWNLOAD_SUCCESS=true
else
    # Download the folder or file from S3
    source /secrets/secrets.env
    

    aws s3 cp "$S3_MODEL_PATH" "$LOCAL_MODEL_PATH" --recursive
    aws s3 cp "$S3_FILE_PATH" "$LOCAL_FILE_PATH" 

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download completed successfully."
        DOWNLOAD_SUCCESS=true
    else
        echo "Download failed."
    fi
fi

pwd
ls
# Execute additional commands
pip install -r requirements.txt

# if $DOWNLOAD_SUCCESS; then
#     echo "Running additional commands..."
#     # Command 1
#     # Command 2
#     # ...
# fi

# Continue with the rest of your script...
