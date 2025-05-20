#!/bin/bash

# AWS S3 Upload Script for NIFTI Files
#
# Description:
# This script uploads all T2 image (.nii.gz) files from a local directory to an AWS S3 bucket.
# It prompts for AWS credentials without hardcoding them, uploads the files,
# and optionally verifies the upload by listing files in the S3 bucket.
#
# Usage:
#   ./t2_transfer2AWS_Ariels_051925.sh
#
# Author: Roy Seo
# Date: May 19, 2025

# Local directory containing .nii.gz files
LOCAL_DIR="/ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_ADNI1234_nifti_4_Ariel"

# S3 bucket destination
S3_BUCKET="s3://adni.nrdg.uw.edu"

# Check if local directory exists
if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: Local directory does not exist: $LOCAL_DIR"
    exit 1
fi

# Count total number of .nii.gz files
total_files=$(find "$LOCAL_DIR" -name "*.nii.gz" | wc -l)
echo "Found $total_files .nii.gz files to upload."

if [ "$total_files" -eq 0 ]; then
    echo "No .nii.gz files found in $LOCAL_DIR. Exiting."
    exit 1
fi

# Prompt for AWS credentials
echo "Please enter your AWS credentials (they will not be stored in the script)"
read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -sp "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo
read -p "AWS Default Region [us-west-2]: " AWS_DEFAULT_REGION
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2}

# Temporarily export AWS credentials for this session
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION

# Verify AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed. Please install it before running this script."
    exit 1
fi

# Verify AWS credentials
echo "Verifying AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "Error: Failed to authenticate with AWS. Please check your credentials and try again."
    exit 1
fi
echo "AWS credentials verified successfully."

# Create a timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="s3_upload_${timestamp}.log"

# Function to upload files
upload_files() {
    echo "Starting upload of .nii.gz files to $S3_BUCKET..."
    echo "This may take some time depending on your internet connection and the size of the files."
    
    # Use sync command to upload only new or modified files
    aws s3 sync "$LOCAL_DIR" "$S3_BUCKET" --exclude "*" --include "*.nii.gz" --no-progress | tee "$log_file"
    
    # Check if the upload was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Upload completed successfully!"
        return 0
    else
        echo "Error occurred during upload. Check the log file: $log_file"
        return 1
    fi
}

# Function to verify upload
verify_upload() {
    echo "Verifying upload..."
    
    # List objects in the S3 bucket
    aws s3 ls "$S3_BUCKET" --recursive | grep ".nii.gz" > "s3_files_${timestamp}.txt"
    
    # Count files in S3 bucket
    s3_file_count=$(wc -l < "s3_files_${timestamp}.txt")
    
    echo "Files in S3 bucket: $s3_file_count"
    echo "Local .nii.gz files: $total_files"
    
    if [ "$s3_file_count" -ge "$total_files" ]; then
        echo "Verification successful! All files appear to be uploaded."
        return 0
    else
        echo "Verification shows potential missing files. Please check 's3_files_${timestamp}.txt'"
        return 1
    fi
}

# Main execution
echo "=================================="
echo "AWS S3 Upload Script"
echo "Local directory: $LOCAL_DIR"
echo "S3 bucket: $S3_BUCKET"
echo "=================================="

# Ask for confirmation before proceeding
read -p "Do you want to proceed with the upload? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 0
fi

# Perform the upload
upload_files

# Ask if verification is desired
read -p "Do you want to verify the upload? (y/n): " verify
if [[ "$verify" =~ ^[Yy]$ ]]; then
    verify_upload
fi

# Clean up environment variables
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_DEFAULT_REGION

echo "Script execution completed."