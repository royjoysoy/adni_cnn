
#'ADNI1234_subj_n1189.txt'의 파일에있는 subjects 1189 (즉 ADNI1234에서 가져온 28002개의 이미지가 총 1189 subjects것이다)
# 1189명의 subjects를 여기서  'All_Subjects_DXSUM_07Jan2025.csv' 골라낸다.
# 몇개가 골라졌는지 출력
# 필터링 테이터에서 누락된 피험자 ID가 있는지 출력
# 진단코드가 변한 사람이 있는지 출력


import pandas as pd

def process_subjects_data(subject_list_file, csv_file, output_file='7-filtered_subjects_DXSUM_01_08-2025.csv'):
    """
    Process subjects data by filtering and sorting according to a provided subject list.
    
    Parameters:
    subject_list_file (str): File containing the list of subject IDs
    csv_file (str): Input CSV file containing all subjects data
    output_file (str): Output file name for filtered and sorted data
    """
    # Read the subject IDs list
    with open(subject_list_file, 'r') as f:
        subject_ids = [line.strip() for line in f.readlines()]
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter the dataframe to include only the subjects in our list
    filtered_df = df[df['PTID'].isin(subject_ids)]
    
    # Create a categorical type for PTID with our specific order
    ordered_ptid = pd.CategoricalDtype(categories=subject_ids, ordered=True)
    filtered_df['PTID'] = filtered_df['PTID'].astype(ordered_ptid)
    
    # Sort by PTID to match our list order
    sorted_df = filtered_df.sort_values('PTID')
    
    # Save to CSV
    sorted_df.to_csv(output_file, index=False)
    
    # Print some summary statistics
    print(f"Total number of subjects in input list: {len(subject_ids)}")
    print(f"Number of subjects found in CSV: {len(sorted_df['PTID'].unique())}")
    print(f"Total number of rows in output: {len(sorted_df)}")
    
    # Check for any subjects that weren't found
    found_subjects = set(sorted_df['PTID'].unique())
    missing_subjects = set(subject_ids) - found_subjects
    if missing_subjects:
        print("\nWarning: The following subjects were not found in the CSV:")
        for subject in sorted(missing_subjects):
            print(subject)


import pandas as pd

def check_diagnosis_changes(df):
    """
    Check for subjects whose diagnosis changed across visits.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing subject data
    
    Returns:
    list: List of tuples containing (PTID, diagnosis changes info)
    """
    changes = []
    for ptid in df['PTID'].unique():
        subject_data = df[df['PTID'] == ptid].sort_values('EXAMDATE')
        
        if len(subject_data['DIAGNOSIS'].unique()) > 1:
            # Get the diagnosis history
            diagnosis_history = subject_data[['EXAMDATE', 'DIAGNOSIS']].values.tolist()
            changes.append((ptid, diagnosis_history))
    
    return changes

def process_subjects_data(subject_list_file, csv_file, output_file='filtered_subjects.csv'):
    """
    Process subjects data by filtering and sorting according to a provided subject list.
    
    Parameters:
    subject_list_file (str): File containing the list of subject IDs
    csv_file (str): Input CSV file containing all subjects data
    output_file (str): Output file name for filtered and sorted data
    """
    # Read the subject IDs list
    with open(subject_list_file, 'r') as f:
        subject_ids = [line.strip() for line in f.readlines()]
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter the dataframe to include only the subjects in our list
    filtered_df = df[df['PTID'].isin(subject_ids)]
    
    # Create a categorical type for PTID with our specific order
    ordered_ptid = pd.CategoricalDtype(categories=subject_ids, ordered=True)
    filtered_df['PTID'] = filtered_df['PTID'].astype(ordered_ptid)
    
    # Sort by PTID to match our list order
    sorted_df = filtered_df.sort_values(['PTID', 'EXAMDATE'])
    
    # Check for diagnosis changes
    print("\nChecking for diagnosis changes...")
    diagnosis_changes = check_diagnosis_changes(sorted_df)
    
    if diagnosis_changes:
        print(f"\nNumber of subjects with diagnosis changes: {len(diagnosis_changes)}")
        print("\nSubjects with diagnosis changes:")
        print("--------------------------------")
        for ptid, history in diagnosis_changes:
            print(f"\nSubject {ptid}:")
            for date, diagnosis in history:
                # Convert diagnosis codes to more readable format
                diagnosis_label = {
                    1: "CN (Cognitively Normal)",
                    2: "MCI (Mild Cognitive Impairment)",
                    3: "AD (Alzheimer's Disease)",
                }.get(diagnosis, f"Other/Unknown ({diagnosis})")
                
                print(f"  {date}: {diagnosis_label}")
    else:
        print("No subjects found with diagnosis changes.")
    
    # Save to CSV
    sorted_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total number of subjects in input list: {len(subject_ids)}")
    print(f"Number of subjects found in CSV: {len(sorted_df['PTID'].unique())}")
    print(f"Total number of rows in output: {len(sorted_df)}")
    print(f"\nNumber of subjects with diagnosis changes: {len(diagnosis_changes)}")
    
    # Check for any subjects that weren't found
    found_subjects = set(sorted_df['PTID'].unique())
    missing_subjects = set(subject_ids) - found_subjects
    if missing_subjects:
        print("\nWarning: The following subjects were not found in the CSV:")
        for subject in sorted(missing_subjects):
            print(subject)

# Example usage:
if __name__ == "__main__":
    subject_list_file = "ADNI1234_subj_n1189.txt"  # Your input subject list file
    csv_file = "All_Subjects_DXSUM_07Jan2025.csv"  # Your input CSV file
    output_file = "6-filtered_subjects.csv"  # Output file name
    
    process_subjects_data(subject_list_file, csv_file, output_file)