"""
USAGE: python3 5-1-adni_subjects_stats-10-3-lin-eval-set.py 10-3-description-filtered-stats-removedrepeat.csv
    

This script analyzes longitudinal visit data from the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset.
It processes the file '10-3-description-filtered-stats-removedrepeat.csv' containing visit records.

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
- command-line arguments CSV file: eaxmple: '10-3-description-filtered-stats-removedrepeat.csv' 
- Required columns: Subject, Sex, Group-RS:Initial, Acq Date

Output:
- Printed statistics
- Generated visualizations:
  * 5-1-1-visits_distribution.png
  * 5-2-1-sex_distribution.png
  * 5-3-1-duration_distribution.png
  * 5-4-1-diagnosis_transitions.png

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
def apply_hanyang_style(ax):
    """Apply Hanyang styling to plot elements"""
    ax.title.set_color(COLORS['primary']['blue'])
    ax.xaxis.label.set_color(COLORS['primary']['blue'])
    ax.yaxis.label.set_color(COLORS['primary']['blue'])
    ax.tick_params(colors=COLORS['primary']['blue'])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['primary']['blue'])

def set_plot_style():
    """Set the default plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')  # Using specific seaborn style
    sns.set_palette([COLORS['primary']['blue'], COLORS['primary']['silver']])

def get_year(date_str):
    """Convert date string to year, handling both string and float inputs"""
    if pd.isna(date_str):
        return None  # Quietly return None for NaN values
        
    # Convert to string if it's not already
    date_str = str(date_str)
    
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
                # Only print problematic dates that aren't NaN
                if date_str.lower() != 'nan':
                    print(f"Problematic date: {date_str}")
                return None

# When using in analyze_adni_data:
    # First, check how many NaN values exist
    print(f"\nNumber of missing dates: {df['Acq Date'].isna().sum()}")
    
    # Then convert to years
    df['Year'] = df['Acq Date'].astype(str).apply(get_year)

def make_autopct(values):
    """Create percentage labels with counts for pie charts"""
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return f'n={val}\n({pct:.1f}%)'
    return my_autopct

def plot_visit_distribution(visits_per_subject):
    """Create an improved visualization of visit distribution"""
    set_plot_style()  # Set style at the start
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    max_visits = visits_per_subject.max()
    bins = np.arange(0, max_visits + 2, 1)
    
    counts, bins, patches = plt.hist(visits_per_subject, 
                                   bins=bins,
                                   color=COLORS['primary']['blue'],
                                   edgecolor='black',
                                   linewidth=1,
                                   alpha=0.8,
                                   zorder=2)
    
    for i, count in enumerate(counts):
        if count > 0:
            plt.text(bins[i] + 0.5, count + 1,
                    f'{int(count)}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    zorder=3)
    
    plt.title('Distribution of Visits per Subject', fontsize=14, pad=20)
    plt.xlabel('Number of Visits', fontsize=12)
    plt.ylabel('Number of Subjects', fontsize=12)
    
    plt.xticks(bins[:-1] + 0.5, bins[:-1].astype(int))
    
    max_count = max(counts)
    plt.ylim(0, max_count * 1.15)
    
    stats_text = (f'Total Subjects: {len(visits_per_subject)}\n'
                 f'Average Visits: {visits_per_subject.mean():.2f}\n'
                 f'Maximum Visits: {int(max_visits)}')
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    apply_hanyang_style(ax)
    plt.tight_layout()
    plt.savefig('5-1-1-visits_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_adni_data(file_path):
    """Main function to analyze ADNI data"""
    set_plot_style()  # Set style at the start
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    print(f"\nTotal number of rows (excluding header): {len(df)}")
    print("\nInitial data check:")
    print("Columns in the dataset:", df.columns.tolist())
    print("\nSample rows:")
    print(df[['Subject', 'DX_fill', 'Sex', 'Acq Date']].head())
    print("\nUnique values in DX_fill column:", df['DX_fill'].unique())
    
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
    
    # Create visit distribution plots
    plot_visit_distribution(visits_per_subject)
    
    # 3. Analyze sex distribution
    sex_dist = df.groupby('Sex')['Subject'].nunique()
    print(f"\n3. Sex Distribution:")
    print(sex_dist)
    
    # Create sex distribution plots with Hanyang theme
    plt.figure(figsize=(8, 8))
    colors = [COLORS['primary']['blue'], COLORS['primary']['silver']]
    text_colors = [COLORS['primary']['silver'], COLORS['primary']['blue']]
    
    edges, texts, autotexts = plt.pie(sex_dist, labels=sex_dist.index, colors=colors,
                                      autopct=make_autopct(sex_dist.values),
                                      textprops={'weight': 'bold', 'fontsize': 15},
                                      wedgeprops={'edgecolor': 'black', 'linewidth': 1},
                                      startangle=90)
    
    for i, autotext in enumerate(autotexts):
        autotext.set_color(text_colors[i])
        autotext.set_fontsize(15)
    for i, text in enumerate(texts):
        text.set_color(text_colors[i])
        text.set_fontsize(15)
        
    plt.title('Sex Distribution of Subjects')
    plt.savefig('5-2-1-sex_distribution.png')
    plt.close()
    
    # 4. Analyze study duration
    print("\nSample dates before processing:")
    print(df['Acq Date'].head())
    
    # Convert dates to years (moved here from global scope)
    df['Year'] = df['Acq Date'].astype(str).apply(get_year)
    df = df.dropna(subset=['Year'])
    
    duration_per_subject = df.groupby('Subject').agg({
        'Year': lambda x: max(x) - min(x)
    })
    
    avg_duration = duration_per_subject['Year'].mean()
    print(f"\n4. Study Duration Statistics:")
    print(f"Average study duration: {avg_duration:.2f} years")
    print(f"Maximum study duration: {duration_per_subject['Year'].max()} years")
    
    # Create duration distribution plot with Hanyang theme
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.grid(True, alpha=0.3, color=COLORS['primary']['silver'])
    counts, bins, _ = plt.hist(duration_per_subject['Year'], bins=15, 
                              color=COLORS['secondary']['yellow_green'],
                              edgecolor='black', 
                              linewidth=1, 
                              alpha=0.8)
    
    plt.title('Distribution of Study Duration per Subject', color=COLORS['primary']['blue'])
    plt.xlabel('Duration (years)', color=COLORS['primary']['blue'], fontsize=15)
    plt.ylabel('Number of Subjects', color=COLORS['primary']['blue'], fontsize=15)
    plt.tick_params(colors=COLORS['primary']['blue'], labelsize=12)
    
    
    for i in range(len(counts)):
        if counts[i] != 0:
            center = (bins[i] + bins[i+1])/2
            plt.text(center, counts[i], f'{int(counts[i])}', 
                    ha='center', va='bottom', fontsize=12)
    
    plt.savefig('5-3-1-duration_distribution.png')
    plt.close()

    # 5. Analyze diagnosis changes
    # First, ensure DX_fill column exists and handle potential missing values
    if 'DX_fill' not in df.columns:
        print("Warning: 'DX_fill' column not found in dataset")
        return
    
    # Get first diagnosis for each subject
    group_dist = df.groupby('Subject')['DX_fill'].first().value_counts()
    
    # Check if we have any valid diagnoses
    if len(group_dist) == 0:
        print("Warning: No valid diagnosis data found")
        return
    
    print("\n5. Group Distribution (First visit diagnosis):")
    print(group_dist)
    
    # Create group distribution plot
    if len(group_dist) > 0:
        plt.figure(figsize=(8, 8))
        colors = [COLORS['occasional']['mint'], 
                 COLORS['occasional']['coral'],
                 COLORS['secondary']['gold']]
        
        # Ensure we don't have more categories than colors
        if len(group_dist) > len(colors):
            colors = colors * (len(group_dist) // len(colors) + 1)
        
        plt.pie(group_dist, labels=group_dist.index, colors=colors[:len(group_dist)],
                autopct=make_autopct(group_dist.values),
                textprops={'weight': 'bold', 'fontsize': 15},
                wedgeprops={'edgecolor': 'black', 'linewidth': 1},
                startangle=90)
        plt.title('Distribution of Diagnostic Groups', fontsize=18)
        plt.savefig('5-4-1-group_distribution.png')
        plt.close()
    
    # Analyze diagnosis transitions
    diagnosis_changes = df.groupby('Subject')['DX_fill'].agg(list)
    changing_diagnosis = diagnosis_changes.apply(lambda x: len(set(x)) > 1)
    n_changed = changing_diagnosis.sum()
    
    print("\n6. Diagnosis Changes Analysis:")
    print("\nFirst 10 subjects' diagnosis sequences:")
    print(diagnosis_changes.head(10))
    
    print(f"\nSummary Statistics:")
    print(f"Number of subjects with changing diagnosis: {n_changed}")
    print(f"Percentage of subjects with changing diagnosis: {(n_changed/len(diagnosis_changes))*100:.2f}%")

    # Create transition matrix visualization
    if n_changed > 0:
        transitions = []
        for subject_diagnoses in diagnosis_changes:
            for i in range(len(subject_diagnoses)-1):
                if subject_diagnoses[i] != subject_diagnoses[i+1]:
                    transitions.append((subject_diagnoses[i], subject_diagnoses[i+1]))
        
        if transitions:
            transition_df = pd.DataFrame(transitions, columns=['From', 'To'])
            diagnosis_order = ['CN', 'MCI', 'Dementia']
            transition_matrix = pd.crosstab(transition_df['From'], transition_df['To'])
            transition_matrix = transition_matrix.reindex(index=diagnosis_order, columns=diagnosis_order)
            transition_matrix = transition_matrix.fillna(0).astype(int)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            cmap = sns.light_palette(COLORS['primary']['blue'], as_cmap=True)
            
            sns.heatmap(transition_matrix, 
                       annot=True, 
                       fmt='d', 
                       cmap=cmap,
                       cbar_kws={'label': 'Number of Transitions'}, 
                       square=True)
            
            plt.title('Diagnosis Transition Matrix', pad=20, fontsize=18)
            plt.xlabel('To', labelpad=10, fontsize=15)
            plt.ylabel('From', labelpad=10, fontsize=15)
            
            apply_hanyang_style(ax)
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            plt.savefig('5-5-1-diagnosis_transitions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\nTransition Matrix:")
            print(transition_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze ADNI visit data from a CSV file.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    args = parser.parse_args()
    analyze_adni_data(args.input_file)