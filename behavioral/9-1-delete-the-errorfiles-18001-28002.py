# 1-22-2025 Roy Seo
# 에러난 것 behavioral (demographic) csv file row에서 지우기
#  
# 3_linear_18001-28002
# 3_nonlinear_18001-28002

import pandas as pd

def delete_rows_from_csv(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    print("Initial data info:")
    print(f"Total rows before cleaning: {len(df)}")
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    print(f"Rows after removing empty rows: {len(df)}")
    
    ids_to_remove = [
        # ... (이전 ID 리스트 유지)
    ]
    
    # Remove rows where Image Data ID matches any ID in our list
    df_cleaned = df[~df['Image Data ID'].isin(ids_to_remove)]
    
    # Save the cleaned DataFrame
    df_cleaned.to_csv(output_csv, index=False)
    
    print(f"\nFinal Summary:")
    print(f"Original number of rows (after removing empty rows): {len(df)}")
    print(f"Number of rows removed by ID matching: {len(df) - len(df_cleaned)}")
    print(f"Number of rows in cleaned file: {len(df_cleaned)}")
    print(f"Cleaned file saved as: {output_csv}")



def delete_rows_from_csv(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    print("File information:")
    print("Number of rows:", len(df))
    print("Columns:", df.columns.tolist())
    
    # Print few example rows of Image Data ID column
    print("\nFirst few Image Data IDs:")
    print(df['Image Data ID'].head(10))
    
    ids_to_remove = [
        # First group (11 rows)
        1051042,  # 082_S_4224_2013-10-23_S72333_I1051042
        65418,    # 094_S_1241_2007-02-22_S27009_I65418
        65419,    # 094_S_1241_2007-02-22_S27009_I65419
        65421,    # 094_S_1241_2007-02-22_S27009_I65421
        666335,   # 127_S_1032_2012-12-04_S17650_I666335
        467257,   # 127_S_1032_2014-12-10_S24281_I467257
        467258,   # 127_S_1032_2014-12-10_S24281_I467258
        797080,   # 127_S_4198_2016-09-20_S50044_I797080
        89974,    # 133_S_1170_2006-12-29_S24673_I89974
        89975,    # 133_S_1170_2006-12-29_S24673_I89975
        89976,    # 133_S_1170_2006-12-29_S24673_I89976
        46668,    # 137_S_0686_2006-07-03_S16048_I46668
        66436,    # 137_S_0686_2007-02-12_S26334_I66436
        137267,   # 137_S_0796_2009-02-12_S63090_I137267
        43060,    # 137_S_0973_2006-11-15_S22528_I43060
        109973,   # 137_S_0973_2008-06-03_S50915_I109973
        43071,    # 137_S_1041_2006-11-09_S22310_I43071
        134940,   # 137_S_1041_2008-12-18_S61000_I134940
    ]
    
    # Check if any of these IDs exist in the data
    existing_ids = df[df['Image Data ID'].isin(ids_to_remove)]['Image Data ID'].tolist()
    print("\nFound these IDs in the data:")
    print(existing_ids)
    
    # Remove rows
    df_cleaned = df[~df['Image Data ID'].isin(ids_to_remove)]
    
    # Save the cleaned DataFrame
    df_cleaned.to_csv(output_csv, index=False)
    
    print(f"\nSummary:")
    print(f"Original number of rows: {len(df)}")
    print(f"Number of rows removed: {len(df) - len(df_cleaned)}")
    print(f"Number of rows in cleaned file: {len(df_cleaned)}")

# Usage
input_csv = '1_3_linear_nonlinear18001-28002_n_8829.csv'
output_csv = '9-1-delete-the-errorfiles.csv'
delete_rows_from_csv(input_csv, output_csv)



def check_file_info(input_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    print("Detailed File Information:")
    print("-" * 50)
    print(f"Total rows: {len(df)}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for any duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    # Check for any NULL values
    print("\nNull values in each column:")
    print(df.isnull().sum())
    
    return df

# Usage
input_csv = '1_3_linear_nonlinear18001-28002_n_8829.csv'
df = check_file_info(input_csv)