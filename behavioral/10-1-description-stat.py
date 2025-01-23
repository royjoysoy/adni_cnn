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