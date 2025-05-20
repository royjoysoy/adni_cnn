#!/usr/bin/env python3
"""
ADNI T2 Image Metadata Extractor

This script extracts Image IDs from .nii.gz filenames 
in "/ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_ADNI1234_nifti_4_Ariel", 
filters a metadata file ("/mnt/home/royseo/Downloads/ADNI1_2_3_4_T2_5-10-2025_5_10_2025.csv")
to include only rows with matching Image Data IDs, and saves the filtered data 
to a new Excel file.
things to modify for the future usage:  nifti_dir, output_filepath, metadata_filepath

Author: Roy Seo
Date: May 19, 2025
"""

import os
import re
import pandas as pd
from pathlib import Path

def extract_image_ids(directory):
    """
    Extract Image IDs from .nii.gz filenames in the specified directory.
    The Image ID is assumed to start with 'I' and appears right before '.nii.gz'.
    """
    image_ids = []
    pattern = re.compile(r'(I\d+)\.nii\.gz$')
    
    print(f"Scanning directory: {directory}")
    
    # Get all .nii.gz files in the directory
    nifti_files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]
    print(f"Found {len(nifti_files)} .nii.gz files")
    
    # Extract Image IDs using regex
    for filename in nifti_files:
        match = pattern.search(filename)
        if match:
            image_id = match.group(1)
            image_ids.append(image_id)
    
    # Remove any potential duplicates
    unique_ids = list(set(image_ids))
    print(f"Extracted {len(unique_ids)} unique Image IDs")
    
    return unique_ids

def load_metadata(filepath):
    """
    Load metadata from CSV or Excel file based on file extension.
    """
    print(f"Loading metadata from: {filepath}")
    
    # Determine file type and load accordingly
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Metadata file must be CSV or Excel format")
    
    print(f"Loaded metadata with {len(df)} rows and {len(df.columns)} columns")
    return df

def filter_and_save_metadata(image_ids, metadata_filepath, output_filepath):
    """
    Filter metadata to include only rows with matching Image Data IDs
    and save to a new Excel file.
    """
    # Load metadata
    df = load_metadata(metadata_filepath)
    
    # Check if 'Image Data ID' column exists
    if 'Image Data ID' not in df.columns:
        print("Warning: 'Image Data ID' column not found in metadata.")
        possible_columns = [col for col in df.columns if 'id' in col.lower() or 'image' in col.lower()]
        if possible_columns:
            print(f"Possible ID columns found: {possible_columns}")
        
        # Ask user for the correct column name
        id_column = input("Please enter the correct column name for Image IDs: ")
    else:
        id_column = 'Image Data ID'
    
    # Filter metadata to include only rows with matching Image Data IDs
    filtered_df = df[df[id_column].isin(image_ids)]
    
    print(f"Filtered metadata from {len(df)} to {len(filtered_df)} rows")
    
    # Create directory for output file if it doesn't exist
    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save filtered data to new CSV file
    # Changed from Excel to CSV to match your output filename
    if output_filepath.endswith('.csv'):
        filtered_df.to_csv(output_filepath, index=False)
    else:
        filtered_df.to_excel(output_filepath, index=False)
    
    print(f"Saved filtered metadata to: {output_filepath}")
    
    return filtered_df

def main():
    # Directory containing .nii.gz files
    nifti_dir = "/ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_ADNI1234_nifti_4_Ariel"
    
    # Output file path (CSV format)
    output_filepath = "/ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/ADNI1_2_3_4_T2_530files_info_051925.csv"
    
    # Extract Image IDs from filenames
    image_ids = extract_image_ids(nifti_dir)
    
    # Fix: Set the metadata filepath directly instead of using input()
    metadata_filepath = "/mnt/home/royseo/Downloads/ADNI1_2_3_4_T2_5-10_2025_5_10_2025.csv"
    
    # Filter metadata and save to new CSV file
    filtered_df = filter_and_save_metadata(image_ids, metadata_filepath, output_filepath)
    
    print("\nSummary:")
    print(f"- Extracted {len(image_ids)} unique Image IDs from .nii.gz files")
    print(f"- Filtered metadata file to {len(filtered_df)} matching entries")
    print(f"- Saved filtered data to: {output_filepath}")

if __name__ == "__main__":
    main()