"""
match_adni_subjects.py

This script combines and reorganizes ADNI subject data by:
1. Taking a CSV file downloaded from ADNI ('raw_28002_ADNI_1_2_3_4_11_14_24_1_08_2025.csv')
2. Adding the full image filenames as the first column by matching Image Data IDs 
   from 'subj_list_28001_raw_prac.log'
3. Adding '_warped_brain' before '.nii' and changing extension to '.nii.gz'
4. Reordering the resulting dataset to match the order of subjects in the log file
5. Creating a new CSV file 'adni_1234_28002_df.csv' with the combined data
"""
import pandas as pd
import re

def extract_image_id(filename):
    """Extract the Image Data ID from filename."""
    match = re.search(r'I\d+\.nii$', filename)
    if match:
        return match.group(0).replace('.nii', '')
    return None

def modify_filename(filename):
    """Add '_warped_brain' before '.nii' and change extension to '.nii.gz'"""
    if filename:
        return filename.replace('.nii', '_warped_brain.nii.gz')
    return ''

def create_matched_dataset():
    # Read the log file
    with open('/Users/test_terminal/Desktop/adni_cnn/img_reg_nor/subj_list_28001_raw_prac.log', 'r') as f:
        log_files = f.read().splitlines()
    
    # Create a dictionary of Image IDs to full filenames
    image_id_to_filename = {}
    for filename in log_files:
        image_id = extract_image_id(filename)
        if image_id:
            image_id_to_filename[image_id] = filename
    
    # Read the CSV file
    df = pd.read_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/raw_28002_ADNI_1_2_3_4_11_14_24_1_08_2025.csv')
    
    # Create a new column for the full filename and modify it
    df['Full_Filename'] = df['Image Data ID'].map(lambda x: image_id_to_filename.get(x, ''))
    df['Full_Filename'] = df['Full_Filename'].apply(modify_filename)
    
    # Reorder columns to put Full_Filename first
    cols = ['Full_Filename'] + [col for col in df.columns if col != 'Full_Filename']
    df = df[cols]
    
    # Create a dictionary for ordering based on log file
    order_dict = {modify_filename(filename): idx for idx, filename in enumerate(log_files)}
    
    # Sort the dataframe based on the log file order
    df['sort_order'] = df['Full_Filename'].map(lambda x: order_dict.get(x, float('inf')))
    df = df.sort_values('sort_order')
    df = df.drop('sort_order', axis=1)
    
    # Save to new CSV file
    df.to_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/4-adni_1234_28002_dx_age_sex_acqdate.csv', index=False)
    
    # Print some statistics
    print(f"Total rows in output CSV: {len(df)}")
    print(f"Number of matched files: {len(df[df['Full_Filename'] != ''])}")
    print(f"Number of unmatched files: {len(df[df['Full_Filename'] == ''])}")
    print(f"Number of unmatched files: {(df[df['Full_Filename'] == ''])}")
if __name__ == "__main__":
    create_matched_dataset()