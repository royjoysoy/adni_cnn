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
  * 5-1-visits_distribution.png
  * 5-2-sex_distribution.png
  * 5-3-duration_distribution.png
  * 5-4-diagnosis_transitions.png

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

# Hanyang Color Theme
COLORS = {
    'primary': {
        'blue': '#0E4A84',    # PANTONE 541C (C91 M59 Y0 K32)
        'silver': '#898C8E'    # PANTONE 877C (K55)
    },
    'secondary': {
        'yellow_green': '#7DB928',  # PANTONE 375C
        'orange': '#F08100',        # PANTONE 144C
        'gold': '#88774F'          # PANTONE 871C
    },
    'occasional': {
        'mint': '#6CCA98',        # PANTONE 346C
        'coral': '#FF8672'        # PANTONE 170C
    }
}

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

def set_style():
    """Set the style for all plots"""
    plt.style.use('seaborn')
    sns.set_palette([COLORS['primary']['blue'], COLORS['primary']['silver']])

def analyze_adni_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"\nTotal number of rows (excluding header): {len(df)}")
    print("\nInitial data check:")
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
    
    # Original visits distribution plot
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(visits_per_subject, bins=20, edgecolor='black', linewidth=1)
    plt.title('Distribution of Number of Visits per Subject')
    plt.xlabel('Number of Visits')
    plt.ylabel('Number of Subjects')
    
    # Add frequency numbers on top of each bar
    for i in range(len(counts)):
        if counts[i] != 0:  # Only add label if bar height is not zero
            center = (bins[i] + bins[i+1])/2  # Calculate center of the bar
            plt.text(center, counts[i], f'{int(counts[i])}', 
                    ha='center', va='bottom')
    
    plt.savefig('5-1-visits_distribution_original.png')
    plt.show()
    plt.close()
    
    # New visits distribution with Hanyang theme
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3, zorder=0)  # Add grid first with zorder=0
    
    # Define specific bins to match your desired visualization
    bins = np.arange(0, 14, 2)  # Creates bins [0, 2, 4, 6, 8, 10, 12]
    
    counts, bins, patches = plt.hist(visits_per_subject, bins=bins, 
                              color=COLORS['primary']['blue'], 
                              edgecolor='black', linewidth=1, alpha=0.8,
                              zorder=2)  # Add histogram with higher zorder
    plt.title('Distribution of Number of Visits per Subject (Hanyang Theme)')
    plt.xlabel('Number of Visits')
    plt.ylabel('Number of Subjects')
    
    # Add frequency numbers on top of each bar - corrected center calculation
    bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    for count, center in zip(counts, bin_centers):
        if count != 0:  # Only add label if bar height is not zero
            plt.text(center, count, f'{int(count)}', 
                    ha='center', va='bottom',
                    zorder=3)  # Text should be on top of everything
    
    # Set x-axis ticks to match your visualization
    plt.xticks(bins)
    
    # Set y-axis limits
    plt.ylim(0, 300)
    
    plt.savefig('5-1-visits_distribution_hanyang.png')
    plt.close()
    
    # 3. Analyze sex distribution
    sex_dist = df.groupby('Sex')['Subject'].nunique()
    print(f"\n3. Sex Distribution:")
    print(sex_dist)
    
    # Function to create percentage labels with counts
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return f'n={val}\n({pct:.1f}%)'
        return my_autopct
    
    # Original sex distribution plot
    plt.figure(figsize=(8, 8))
    plt.pie(sex_dist, labels=sex_dist.index, 
            autopct=make_autopct(sex_dist.values),
            textprops={'weight': 'bold'})
    plt.title('Sex Distribution of Subjects')
    plt.savefig('5-2-sex_distribution_original.png')
    plt.close()
    
    # New sex distribution with Hanyang theme
    plt.figure(figsize=(8, 8))
    colors = [COLORS['primary']['blue'], COLORS['primary']['silver']]
    text_colors = [COLORS['primary']['silver'], COLORS['primary']['blue']]  # Reversed colors for text
    
    # Updated pie chart with text colors and black edges
    wedges, texts, autotexts = plt.pie(sex_dist, labels=sex_dist.index, colors=colors,
                                      autopct=make_autopct(sex_dist.values),
                                      textprops={'weight': 'bold'},
                                      wedgeprops={'edgecolor': 'black', 'linewidth': 1},  # Add black borders
                                      startangle=90)
    
    # Set the colors for each text element
    for i, autotext in enumerate(autotexts):
        autotext.set_color(text_colors[i])
    for i, text in enumerate(texts):
        text.set_color(text_colors[i])
        
    plt.title('Sex Distribution of Subjects (Hanyang Theme)')
    plt.savefig('5-2-sex_distribution_hanyang.png')
    plt.close()
    
    # 4. Analyze study duration
    print("\nSample dates before processing:")
    print(df['Acq Date'].head())
    
    df['Year'] = df['Acq Date'].apply(get_year)
    df = df.dropna(subset=['Year'])
    
    duration_per_subject = df.groupby('Subject').agg({
        'Year': lambda x: max(x) - min(x)
    })
    
    avg_duration = duration_per_subject['Year'].mean()
    print(f"\n4. Study Duration Statistics:")
    print(f"Average study duration: {avg_duration:.2f} years")
    print(f"Maximum study duration: {duration_per_subject['Year'].max()} years")
    
    # Original duration distribution
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(duration_per_subject['Year'], bins=20, 
                              edgecolor='black', linewidth=1)
    plt.title('Distribution of Study Duration per Subject')
    plt.xlabel('Duration (years)')
    plt.ylabel('Number of Subjects')
    
    # Add frequency numbers on top of each bar
    for i in range(len(counts)):
        if counts[i] != 0:  # Only add label if bar height is not zero
            center = (bins[i] + bins[i+1])/2  # Calculate center of the bar
            plt.text(center, counts[i], f'{int(counts[i])}', 
                    ha='center', va='bottom')
    
    plt.savefig('5-3-duration_distribution_original.png')
    plt.close()
    
    # New duration distribution with Hanyang theme
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(duration_per_subject['Year'], bins=15, 
                              color=COLORS['secondary']['yellow_green'],
                              edgecolor='black', linewidth=1, alpha=0.8)
    plt.title('Distribution of Study Duration per Subject')
    plt.xlabel('Duration (years)')
    plt.ylabel('Number of Subjects')
    plt.grid(True, alpha=0.3)
    
    # Add frequency numbers on top of each bar
    for i in range(len(counts)):
        if counts[i] != 0:  # Only add label if bar height is not zero
            center = (bins[i] + bins[i+1])/2  # Calculate center of the bar
            plt.text(center, counts[i], f'{int(counts[i])}', 
                    ha='center', va='bottom')
    
    plt.savefig('5-3-duration_distribution_hanyang.png')
    plt.close()
    
    # 5. Analyze diagnosis changes
    diagnosis_changes = df.groupby(['Subject'])['Group'].agg(list)
    
    print("\n5. Diagnosis Changes Analysis:")
    print("\nFirst 10 subjects' diagnosis sequences:")
    print(diagnosis_changes.head(10))
    
    # Add new Group distribution analysis with Hanyang theme
    group_dist = df.groupby('Subject')['Group'].first().value_counts()
    print("\n6. Group Distribution (First visit diagnosis):")
    print(group_dist)
    
    plt.figure(figsize=(8, 8))
    colors = [COLORS['occasional']['mint'], 
             COLORS['occasional']['coral'],
             COLORS['secondary']['gold']]
    wedges, texts, autotexts = plt.pie(group_dist, labels=group_dist.index, colors=colors,
            autopct=make_autopct(group_dist.values),
            textprops={'weight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1},  # Add black borders
            startangle=90)
    plt.title('Distribution of Diagnostic Groups')
    plt.savefig('5-4-group_distribution_hanyang.png')
    plt.close()
    
    # Continue with original diagnosis transition analysis
    print("\nSubjects with diagnosis changes:")
    for subject, dx_list in diagnosis_changes.items():
        if len(set(dx_list)) > 1:
            print(f"{subject}: {dx_list}")
    
    changing_diagnosis = diagnosis_changes.apply(lambda x: len(set(x)) > 1)
    n_changed = changing_diagnosis.sum()
    
    print(f"\nSummary Statistics:")
    print(f"Number of subjects with changing diagnosis: {n_changed}")
    print(f"Percentage of subjects with changing diagnosis: {(n_changed/len(diagnosis_changes))*100:.2f}%")
    
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
            plt.savefig('5-5-diagnosis_transitions.png')
            plt.close()
            
            print("\nTransition Matrix:")
            print(transition_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze ADNI visit data from a CSV file.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    
    args = parser.parse_args()
    analyze_adni_data(args.input_file)