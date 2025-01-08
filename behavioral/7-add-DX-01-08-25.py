"""
Script: 7-adding-DX_01_08_2025.py
Purpose: Add diagnostic (DX) information to ADNI MRI dataset by matching image and subject IDs

This script merges diagnostic information from ADNIMERGE into a dataset containing ADNI MRI information.
It matches records based on:
- Image Data ID (from input file) with IMAGEUID (from ADNIMERGE)
- Subject (from input file) with PTID (from ADNIMERGE)

Input Files:
- 4-adni_1234_28002_dx_age_sex_acqdate.csv: Contains MRI data with Image Data ID and Subject
- ADNIMERGE_08Jan2025.csv: Contains diagnostic information

Output File:
- 7-add-DX.csv: Original data with added DX column

Features:
- Performs left merge to preserve all original entries
- Handles float to integer conversion for IMAGEUID matching
- Provides warnings for unmatched entries
- Displays distribution of diagnostic categories
- Includes error handling for file operations

Usage:
Place both input CSV files in the same directory as this script and run:
python 7-adding-DX_01_08_2025.py

Date Created: January 8, 2025
"""

import pandas as pd
import numpy as np

def clean_image_id(id_val):
    # Convert to string and remove any 'I' prefix
    id_str = str(id_val).strip()
    if id_str.startswith('I'):
        id_str = id_str[1:]
    # Remove any decimal points and zeros after decimal
    if '.' in id_str:
        id_str = id_str.split('.')[0]
    return id_str

def add_dx_column():
    # Read the input CSV files
    print("Reading input files...")
    adni_data = pd.read_csv('4-adni_1234_28002_dx_age_sex_acqdate.csv')
    adnimerge = pd.read_csv('ADNIMERGE_08Jan2025.csv', low_memory=False)
    
    # Create a copy of the original data
    result_df = adni_data.copy()
    
    # Clean and standardize the Image IDs in both dataframes
    print("Standardizing Image IDs...")
    result_df['Image Data ID'] = result_df['Image Data ID'].apply(clean_image_id)
    adnimerge['IMAGEUID'] = adnimerge['IMAGEUID'].apply(clean_image_id)
    
    # Remove any leading/trailing whitespace from subject IDs
    result_df['Subject'] = result_df['Subject'].str.strip()
    adnimerge['PTID'] = adnimerge['PTID'].str.strip()
    
    # Print sample of processed data
    print("\nProcessed data sample:")
    print("\nADNI Data (processed):")
    print(result_df[['Image Data ID', 'Subject']].head())
    print("\nADNIMERGE (processed):")
    print(adnimerge[['IMAGEUID', 'PTID', 'DX']].head())
    
    # Check for exact matches before merging
    print("\nChecking for exact matches...")
    sample_ids = result_df['Image Data ID'].head()
    print("First few Image Data IDs and their matches in ADNIMERGE:")
    for img_id in sample_ids:
        matches = adnimerge[adnimerge['IMAGEUID'] == img_id]
        print(f"ID {img_id}: {len(matches)} matches found")
    
    # Perform the merge
    print("\nMerging datasets...")
    merged_df = pd.merge(
        result_df,
        adnimerge[['IMAGEUID', 'PTID', 'DX']],
        left_on=['Image Data ID', 'Subject'],
        right_on=['IMAGEUID', 'PTID'],
        how='left'
    )
    
    # Drop duplicate columns from the merge
    merged_df = merged_df.drop(['IMAGEUID', 'PTID'], axis=1)
    
    # Check for any unmatched entries
    unmatched = merged_df[merged_df['DX'].isna()]
    if len(unmatched) > 0:
        print(f"\nWarning: {len(unmatched)} entries could not be matched with a diagnosis")
        print("First few unmatched entries:")
        print(unmatched[['Image Data ID', 'Subject']].head())
        
        # Additional debugging information for unmatched entries
        print("\nChecking first unmatched ID in ADNIMERGE:")
        first_unmatched_id = unmatched['Image Data ID'].iloc[0]
        first_unmatched_subject = unmatched['Subject'].iloc[0]
        print(f"Looking for Image ID: {first_unmatched_id}")
        print(f"Looking for Subject: {first_unmatched_subject}")
        print("Matching rows in ADNIMERGE:")
        print(adnimerge[
            (adnimerge['IMAGEUID'] == first_unmatched_id) |
            (adnimerge['PTID'] == first_unmatched_subject)
        ])
    
    # Save the result
    print("\nSaving results...")
    merged_df.to_csv('7-add-DX.csv', index=False)
    print(f"Successfully saved output to 7-add-DX.csv")
    
    # Print summary statistics
    print("\nSummary of DX distribution:")
    print(merged_df['DX'].value_counts(dropna=False))

if __name__ == "__main__":
    try:
        add_dx_column()
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}")
    except Exception as e:
        print(f"Error occurred: {e}")