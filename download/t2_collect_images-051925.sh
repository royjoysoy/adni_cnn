#!/bin/bash

# T2 Image Collection Script
#
# Description:
# This script searches for T2 MRI images (NIfTI format .nii.gz files) in specified source directories
# and moves them to a destination directory. It identifies T2 images by checking if the filename
# contains "T2", "t2", "FLAIR", or "flair" and ends with .nii.gz. The script creates the destination 
# directory if it doesn't exist and prints the total number of files moved at the end.
#
# Usage:
#   ./t2_collect_051925.sh
#
# Source directories:
#   /ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_adni1234_5-10-2025-nifti_zip1
#   /ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_adni1234_5-10-2025-nifti_zip2
#
# Destination directory:
#   /ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_ADNI1234_nifti_4_Ariel
#
# Author: Roy Seo
# Date: May 19, 2025

# Source directories
SOURCE_DIR1="/ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_adni1234_5-10-2025-nifti_zip1"
SOURCE_DIR2="/ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_adni1234_5-10-2025-nifti_zip2"

# Destination directory
DEST_DIR="/ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_ADNI1234_nifti_4_Ariel"

# Counter for moved files
count=0

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    echo "Creating destination directory: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

echo "Searching for T2 images in source directories..."

# Function to move T2 images and count them
move_t2_images() {
    local src_dir="$1"
    local files_found=0
    
    echo "Searching in: $src_dir"
    
    # Find T2 images (case insensitive search for T2 or FLAIR in filename)
    for file in $(find "$src_dir" -type f -name "*.nii.gz" | grep -i -E '(T2|t2|FLAIR|flair)'); do
        echo "Moving: $(basename "$file")"
        # Move file to destination
        cp "$file" "$DEST_DIR"
        
        # Check if the move was successful
        if [ $? -eq 0 ]; then
            ((count++))
            ((files_found++))
        else
            echo "Error moving file: $file"
        fi
    done
    
    echo "Found $files_found T2 images in $src_dir"
}

# Process each source directory
move_t2_images "$SOURCE_DIR1"
move_t2_images "$SOURCE_DIR2"

# Print summary
echo "=================================="
echo "Operation complete!"
echo "Total T2 images moved: $count"
echo "Destination: $DEST_DIR"
echo "=================================="