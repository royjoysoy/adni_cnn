# 01-23-2025 Roy Seo Korea
'''
Load the CSV file into a DataFrame using pandas.
Group the data by the Subject (3rd column) and Visit (7th column).
Count the occurrences of each unique Description (8th column) within each Subject and Visit group.
Identify the most and least frequently occurring Description in each group.
csv file: 8-2-DX_filled2
output: print in the terminal and '10-1-description-stat.txt' 
'''

import pandas as pd

# Read CSV file 
df = pd.read_csv('8-2-DX_filled2.csv')

# Count unique subjects
n_subjects = df['Subject'].nunique()
print(f"\nNumber of unique subjects: {n_subjects}\n")

# Count occurrences of each unique Description
desc_counts = df['Description'].value_counts()

# Print and save all counts
with open('10-1-description-stat.txt', 'w') as f:
   f.write("Description Frequency Statistics\n")
   f.write("-" * 30 + "\n\n")
   
   f.write(f"Number of unique subjects: {n_subjects}\n\n")
   f.write("Description counts:\n")
   
   for desc, count in desc_counts.items():
       line = f"{desc}: {count} times\n"
       print(line, end='')
       f.write(line)