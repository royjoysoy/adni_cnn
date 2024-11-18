import os
import pandas as pd

# Get current directory and file paths
current_dir = os.getcwd()
file1 = os.path.join(current_dir, '1_output_MERGED_df_modified_AND_idaSearch_11_18_24.csv')

# Read CSV file
df1 = pd.read_csv(file1)

# Find duplicated Image IDs
duplicates = df1[df1['Image ID'].duplicated(keep=False)]

# Display duplicated rows
print("Duplicated Image IDs:")
print(duplicates[['Image ID']].to_string())
print(f"\nTotal number of duplicates: {len(duplicates)}")