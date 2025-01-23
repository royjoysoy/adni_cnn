# 마지막 라인 매뉴얼리 없애라
# fsl_anat_subj_list_2409_raw.log 이름바꿔서 저장하였음 /img_reg_nor/fsl_anat_subj_list_2409_raw.log

import pandas as pd

# Read CSV file and keep first column
df = pd.read_csv('10-3-description-filtered-stats-removedrepeat.csv')
df = df[['Full_Filename']]

# Replace warped_brain.nii.gz with .nii
df['Full_Filename'] = df['Full_Filename'].str.replace('_warped_brain.nii.gz', '.nii')

# Save without header
df.to_csv('10-4-make-subject.log', index=False, header=False)