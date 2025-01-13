# 1-13-2025 Roy Seo
# 에러난 것 behavioral (demographic) csv file row에서 지우기
#  
# 1_linear_1-6000;
# 2_linear_6001-1800;
# 1_linear_1-6000;
# 2_nonlinear_6001-18000;

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
        377040,  # from 016_S_0702_2013-05-09_S18903_I377040
        327057,  # from 021_S_0337_2012-04-24_S14825_I327057
        327059,  # from 021_S_0337_2012-04-24_S14825_I327059
        327062,  # from 021_S_0337_2012-04-24_S14825_I327062
        327074,  # from 021_S_0337_2012-04-24_S14825_I327074
        389140,  # from 021_S_0337_2013-05-07_S18878_I389140
        389138,  # from 021_S_0337_2013-05-07_S18879_I389138
        432139,  # from 021_S_0337_2014-04-24_S21696_I432139
        432140,  # from 021_S_0337_2014-04-24_S21696_I432140
        510022,  # from 021_S_0337_2015-05-07_S25882_I510022
        510051,  # from 021_S_0337_2015-05-07_S25882_I510051
        
        # Second group (12 rows)
        467254,  # from 021_S_0984_2014-12-09_524277_I467254
        173330,  # from 023_S_0625_2007-07-16_535036_173330
        190886,  # from 023_S_0625_2008-01-11_544497_190886
        190882,  # from 023_S_0625_2008-01-11_544498_190882
        124075,  # from 023_S_0625_2008-09-29_556786_I124075
        124080,  # from 023_S_0625_2008-09-29_556787_I124080
        334521,  # from 027_S_0644_2012-07-02_S15633_I334521
        358248,  # from 027_S_4964_2012-10-06_S17005_I358248
        79604,   # from 031_S_0294_2007-09-25_540106_I79604
        67110,   # from 031_S_0618_2006-06-06_S15271_I67110
        72188,   # from 031_S_0618_2007-07-02_536607_I72188
        67706,   # from 033_S_0513_2007-05-31_S33069_I67706
        
        # Third group (1 row)
        1051043  # from 033_S_1016_2016-11-01_S51667_I1051043
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
input_csv = '8-3_1_2_linear_nonlinear_1-6000.csv'
output_csv = '9-delete-the-errorfiles.csv'
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
input_csv = '8-3_1_2_linear_nonlinear_1-6000.csv'
df = check_file_info(input_csv)