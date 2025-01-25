'''
01-25-25 Roy Seo Korea
Split 3 datasets (train, test, eval) for deep learning from 2409 images belonging to 649 subjects.
Two methods:
1. By Subject IDs: the same subject ID remain in the same group (train 7: test 2: eval 1)
2. Random Split: Split the 2409 images into train, test, and eval sets completely randomly (train 7: test 2: eval 1)

* Input: 0-copy-10-3-description-filtered-stats-removedrepeat.csv (columns for Subject and Image).
* Output: 1-dataset-spliting-01-25-25.py
* Optionally save the results as separate CSV files.
Please provide clear and well-commented code.
 
Minor:
3. Input file에서 1st column: Full Filename 에서 예: 002_S_0413_2006-05-02_S13893_I45116_warped_brain.nii.gz 
replace "_warped_brain.nii.gz" with "T1_to_MNI_nonlin.nii.gz"  

Usage: 
python 1-dataset-spliting-01-25-25.py 0-copy-10-3-description-filtered-stats-removedrepeat.csv --output_dir "/Users/test_terminal/Desktop/adni_cnn/lin_eval_dataset_split"

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

def clean_filename(filename):
    """Replace 'warped_brain.nii.gz' with 'T1_to_MNI_nonlin.nii.gz'"""
    return filename.replace('warped_brain.nii.gz', 'T1_to_MNI_nonlin.nii.gz')

def split_by_subject(df, train_ratio=0.7, test_ratio=0.2, eval_ratio=0.1, random_state=42):
    """Split dataset by subject IDs."""
    assert abs(train_ratio + test_ratio + eval_ratio - 1.0) < 1e-10
    
    subjects = df['Subject'].unique()
    np.random.seed(random_state)
    np.random.shuffle(subjects)
    
    n_subjects = len(subjects)
    train_idx = int(n_subjects * train_ratio)
    test_idx = int(n_subjects * (train_ratio + test_ratio))
    
    train_subjects = subjects[:train_idx]
    test_subjects = subjects[train_idx:test_idx]
    eval_subjects = subjects[test_idx:]
    
    train_df = df[df['Subject'].isin(train_subjects)]
    test_df = df[df['Subject'].isin(test_subjects)]
    eval_df = df[df['Subject'].isin(eval_subjects)]
    
    return train_df, test_df, eval_df

def random_split(df, train_ratio=0.7, test_ratio=0.2, eval_ratio=0.1, random_state=42):
    """Random split regardless of subject IDs."""
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio,
        random_state=random_state
    )
    
    relative_test_ratio = test_ratio / (test_ratio + eval_ratio)
    test_df, eval_df = train_test_split(
        temp_df,
        train_size=relative_test_ratio,
        random_state=random_state
    )
    
    return train_df, test_df, eval_df

def save_splits(train_df, test_df, eval_df, output_dir, prefix):
    """Save split DataFrames to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean filenames
    for df in [train_df, test_df, eval_df]:
        df['Full_Filename'] = df['Full_Filename'].apply(clean_filename)
    
    train_df.to_csv(output_dir / f"1-{prefix}_train.csv", index=False)
    test_df.to_csv(output_dir / f"1-{prefix}_test.csv", index=False)
    eval_df.to_csv(output_dir / f"1-{prefix}_eval.csv", index=False)

def main(input_file, output_dir="split_outputs"):
    df = pd.read_csv(input_file)
    
    required_columns = ['Subject', 'Image Data ID', 'Full_Filename']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    train_subj, test_subj, eval_subj = split_by_subject(df)
    save_splits(train_subj, test_subj, eval_subj, output_dir, "subject_split")
    
    train_rand, test_rand, eval_rand = random_split(df)
    save_splits(train_rand, test_rand, eval_rand, output_dir, "random_split")
    
    print("\nSubject-based split statistics:")
    print(f"Train set: {len(train_subj)} images, {len(train_subj['Subject'].unique())} subjects")
    print(f"Test set: {len(test_subj)} images, {len(test_subj['Subject'].unique())} subjects")
    print(f"Eval set: {len(eval_subj)} images, {len(eval_subj['Subject'].unique())} subjects")
    
    print("\nRandom split statistics:")
    print(f"Train set: {len(train_rand)} images")
    print(f"Test set: {len(test_rand)} images")
    print(f"Eval set: {len(eval_rand)} images")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into train, test, and eval sets")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("--output_dir", default="split_outputs", help="Output directory for split files")
    
    args = parser.parse_args()
    main(args.input_file, args.output_dir)