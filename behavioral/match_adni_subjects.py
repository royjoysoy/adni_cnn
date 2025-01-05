"""
match_adni_subjects.py

This script combines and reorganizes ADNI subject data by:
1. Taking a CSV file downloaded from ADNI ('ADNI_1_2_3_4_11_14_24_1_02_2025.csv')
2. Adding the full image filenames as the first column by matching Image Data IDs 
   from 'subj_list_28001_raw_prac.log'
3. Reordering the resulting dataset to match the order of subjects in the log file
4. Creating a new CSV file 'adni_1234_28002_df.csv' with the combined data

l

Input files required:
- ADNI_1_2_3_4_11_14_24_1_02_2025.csv: Subject information from ADNI database
- subj_list_28001_raw_prac.log: List of image filenames

Output:
- adni_1234_28002_df.csv: Combined dataset with image filenames and subject information

Later, the output file is going to use a csv file that has image file path.
"""
import pandas as pd
import re

def extract_image_id(filename):
    """Extract the Image Data ID from filename."""
    match = re.search(r'I\d+\.nii$', filename)
    if match:
        return match.group(0).replace('.nii', '')
    return None

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
    df = pd.read_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/ADNI_1_2_3_4_11_14_24_1_02_2025.csv')
    
    # Create a new column for the full filename
    df['Full_Filename'] = df['Image Data ID'].map(lambda x: image_id_to_filename.get(x, ''))
    
    # Reorder columns to put Full_Filename first
    cols = ['Full_Filename'] + [col for col in df.columns if col != 'Full_Filename']
    df = df[cols]
    
    # Create a dictionary for ordering based on log file
    order_dict = {filename: idx for idx, filename in enumerate(log_files)}
    
    # Sort the dataframe based on the log file order
    # Note: Rows with filenames not in log file will go to the end
    df['sort_order'] = df['Full_Filename'].map(lambda x: order_dict.get(x, float('inf')))
    df = df.sort_values('sort_order')
    df = df.drop('sort_order', axis=1)
    
    # Save to new CSV file
    df.to_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/adni_1234_28002_dx_age_sex_acqdate.csv', index=False)
    
    # Print some statistics
    print(f"Total rows in output CSV: {len(df)}")
    print(f"Number of matched files: {len(df[df['Full_Filename'] != ''])}")
    print(f"Number of unmatched files: {len(df[df['Full_Filename'] == ''])}")

if __name__ == "__main__":
    create_matched_dataset()