"""
USAGE: python3 5-adni_subjects_stats.py 4-adni_1234_28002_dx_age_sex_acqdate.csv
       python3 5-adni_subjects_stats.py 1_1_input_df_modified_prac.csv


This script analyzes longitudinal visit data from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset.
It processes the file '4-adni_1234_28002_dx_age_sex_acqdate.csv' containing visit records.

Performs the following analyses:

1. Subject Count Analysis:
   - Counts total unique subjects in the dataset

2. Visit Pattern Analysis:
   - Calculates visits per subject
   - Determines average and maximum number of visits
   - Visualizes visit distribution

3. Demographic Analysis:
   - Analyzes sex distribution among subjects
   - Creates pie chart visualization

4. Study Duration Analysis:
   - Calculates study duration for each subject
   - Determines average and maximum follow-up periods
   - Visualizes duration distribution

5. Diagnosis Transition Analysis:
   - Tracks diagnosis changes over visits
   - Calculates percentage of subjects with diagnosis changes
   - Creates transition matrix visualization

Input:
- command-line arguments CSV file: eaxmple: '4-adni_1234_28002_dx_age_sex_acqdate.csv' 
- Required columns: Subject, Sex, Group, Acq Date

Output:
- Printed statistics
- Generated visualizations:
  * visits_distribution.png
  * sex_distribution.png
  * duration_distribution.png
  * diagnosis_transitions.png

Dependencies:
- pandas
- matplotlib
- seaborn
- numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import argparse 

def get_year(date_str):
    try:
        # First try mm/dd/yy format
        return datetime.strptime(date_str, '%m/%d/%y').year
    except ValueError:
        try:
            # Then try mm/dd/yyyy format
            return datetime.strptime(date_str, '%m/%d/%Y').year
        except ValueError:
            try:
                # Then try yyyy-mm-dd format
                return datetime.strptime(date_str, '%Y-%m-%d').year
            except ValueError:
                print(f"Problematic date: {date_str}")
                return None
            
def analyze_adni_data(file_path):
#def analyze_adni_data(file_path='4-adni_1234_28002_dx_age_sex_acqdate.csv'):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    print("Initial data check:")
    print("Columns in the dataset:", df.columns.tolist())
    print("\nSample rows:")
    print(df[['Subject', 'Group', 'Sex', 'Acq Date']].head())
    print("\nUnique values in Group column:", df['Group'].unique())
    
    # 1. Count unique subjects
    n_subjects = df['Subject'].nunique()
    print(f"\n1. Total number of unique subjects: {n_subjects}")
    
    # 2. Analyze visits
    visits_per_subject = df.groupby('Subject')['Acq Date'].nunique()
    avg_visits = visits_per_subject.mean()
    max_visits = visits_per_subject.max()
    print(f"\n2. Visit Statistics:")
    print(f"Average visits per subject: {avg_visits:.2f}")
    print(f"Maximum visits for a subject: {max_visits}")
    
    # Visualize visits distribution
    plt.figure(figsize=(10, 6))
    plt.hist(visits_per_subject, bins=20)
    plt.title('Distribution of Number of Visits per Subject')
    plt.xlabel('Number of Visits')
    plt.ylabel('Number of Subjects')
    plt.savefig('visits_distribution.png')
    plt.close()
    
    # 3. Analyze sex distribution
    sex_dist = df.groupby('Sex')['Subject'].nunique()
    print(f"\n3. Sex Distribution:")
    print(sex_dist)
    
    # Visualize sex distribution
    plt.figure(figsize=(8, 8))
    plt.pie(sex_dist, labels=sex_dist.index, autopct='%1.1f%%')
    plt.title('Sex Distribution of Subjects')
    plt.savefig('sex_distribution.png')
    plt.close()
    
    # 4. Analyze study duration for each subject
    # First, let's check the date format
    print("\nSample dates before processing:")
    print(df['Acq Date'].head())
    
    df['Year'] = df['Acq Date'].apply(get_year)
    
    # Remove any None values
    df = df.dropna(subset=['Year'])
    
    duration_per_subject = df.groupby('Subject').agg({
        'Year': lambda x: max(x) - min(x)
    })
    
    avg_duration = duration_per_subject['Year'].mean()
    print(f"\n4. Study Duration Statistics:")
    print(f"Average study duration: {avg_duration:.2f} years")
    print(f"Maximum study duration: {duration_per_subject['Year'].max()} years")
    
    # Visualize duration distribution
    plt.figure(figsize=(10, 6))
    plt.hist(duration_per_subject['Year'], bins=20)
    plt.title('Distribution of Study Duration per Subject')
    plt.xlabel('Duration (years)')
    plt.ylabel('Number of Subjects')
    plt.savefig('duration_distribution.png')
    plt.close()
    
    # 5. Analyze diagnosis changes - with more detailed output
    diagnosis_changes = df.groupby(['Subject'])['Group'].agg(list)
    
    print("\n5. Diagnosis Changes Analysis:")
    print("\nFirst 10 subjects' diagnosis sequences:")
    print(diagnosis_changes.head(10))
    
    print("\nSubjects with diagnosis changes:")
    for subject, dx_list in diagnosis_changes.items():
        if len(set(dx_list)) > 1:  # Check if there are different diagnoses
            print(f"{subject}: {dx_list}")
    
    changing_diagnosis = diagnosis_changes.apply(lambda x: len(set(x)) > 1)
    n_changed = changing_diagnosis.sum()
    
    print(f"\nSummary Statistics:")
    print(f"Number of subjects with changing diagnosis: {n_changed}")
    print(f"Percentage of subjects with changing diagnosis: {(n_changed/len(diagnosis_changes))*100:.2f}%")
    
    # Instead of creating an empty heatmap, only create if there are transitions
    if n_changed > 0:
        transitions = []
        for subject_diagnoses in diagnosis_changes:
            for i in range(len(subject_diagnoses)-1):
                if subject_diagnoses[i] != subject_diagnoses[i+1]:
                    transitions.append((subject_diagnoses[i], subject_diagnoses[i+1]))
        
        if transitions:
            transition_df = pd.DataFrame(transitions, columns=['From', 'To'])
            transition_matrix = pd.crosstab(transition_df['From'], transition_df['To'])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='YlOrRd')
            plt.title('Diagnosis Transition Matrix')
            plt.savefig('diagnosis_transitions.png')
            plt.close()
            
            print("\nTransition Matrix:")
            print(transition_matrix)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze ADNI visit data from a CSV file.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run analysis with provided input file
    analyze_adni_data(args.input_file)