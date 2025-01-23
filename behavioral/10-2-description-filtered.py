import pandas as pd

# Read CSV file
df = pd.read_csv('8-2-DX_filled2.csv')

# Filter rows containing specific Description
filtered_df = df[df['Description'] == 'MPR; GradWarp; B1 Correction; N3']

# Save filtered dataframe to new CSV
filtered_df.to_csv('10-2-description-filtered.csv', index=False)