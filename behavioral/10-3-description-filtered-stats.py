'''
0. PRECHEK: 10-2-description-filtered.py 에서 MPR; GradWarp; B1 Correction; N3 처리 된것을 필터링한 
            output:'10-2-description-filtered.csv'에서 같은 'Subject' 같은 'Visit' time point, 같은 'Acq Date'가 있는지 확인. 
            있어서 지웠음. 중복된 경우 첫번째것을 버리고 두번째것을 킵하고 싶었는데 아마 두번째 것이 버려진거 같음 (이부분: df_unique = df.drop_duplicates(subset=['Subject', 'Visit', 'Acq Date'], keep='first'))
1. Subject 열에서 동일한 Visit이 여러 번 나타나는 Subject들을 확인하고, 해당 정보 출력.
2. 고유한 Subject의 수를 계산하여 출력.
3. DX_fill 그룹(예: 'CN', 'Dementia', 'MCI')별로 데이터 통계를 계산하고 출력.
   * 그룹별로 행의 개수를 계산.
4. Age 열에 대한 통계를 계산하여 출력.
   * 최소값, 최대값, 평균값, 중앙값, 표준편차를 계산.
5. Sex 열에 대해 통계를 계산하여 출력.
   * 각 성별('M', 'F')의 개수를 계산.
'''


import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv('10-2-description-filtered.csv')

# 0. PRE-CHECK: Find subjects with duplicate visits
subject_visit_pairs = df.groupby(['Subject', 'Visit']).size()
duplicate_visits = subject_visit_pairs[subject_visit_pairs > 1]

print("0. Subjects with duplicate visits in same timepoint:")
print(f"Number of cases: {len(duplicate_visits)}")
print(duplicate_visits)
print("\n")

# Read and process data
df = pd.read_csv('10-2-description-filtered.csv')
df_unique = df.drop_duplicates(subset=['Subject', 'Visit', 'Acq Date'], keep='first')
df_unique.to_csv('10-3-description-filtered-stats-removedrepeat.csv', index=False)

# Analysis by unique subjects
subject_first = df_unique.sort_values('Acq Date').groupby('Subject').first()
unique_subjects = df_unique['Subject'].nunique()

# Count duplicates and multiple visits
duplicate_visits = df_unique.groupby(['Subject', 'Visit']).size()
duplicate_visits = duplicate_visits[duplicate_visits > 1]
multi_visits = df_unique.groupby('Subject')['Visit'].nunique()
multi_visits = multi_visits[multi_visits > 1]

# Calculate statistics
dx_counts = subject_first['DX_fill'].value_counts()
sex_counts = subject_first['Sex'].value_counts()
age_first = subject_first['Age'].describe()
age_all = df_unique['Age'].describe()

# Print results
print("0. Subjects with duplicate visits:")
print(f"Cases: {len(duplicate_visits)}\n{duplicate_visits}\n")

print("1. Subjects with multiple visits:")
print(f"Count: {len(multi_visits)}\n{multi_visits}\n")

print(f"2. Unique subjects: {unique_subjects}\n")

print("3. DX_fill counts (unique subjects):")
print(f"{dx_counts}\n")

print("4a. Age statistics (first visit):")
print(f"Min: {age_first['min']:.1f}")
print(f"Max: {age_first['max']:.1f}")
print(f"Mean: {age_first['mean']:.1f}")
print(f"Median: {age_first['50%']:.1f}")
print(f"Std: {age_first['std']:.1f}\n")

print("4b. Age statistics (all visits):")
print(f"Min: {age_all['min']:.1f}")
print(f"Max: {age_all['max']:.1f}")
print(f"Mean: {age_all['mean']:.1f}")
print(f"Median: {age_all['50%']:.1f}")
print(f"Std: {age_all['std']:.1f}\n")

print("5. Sex distribution (unique subjects):")
print(sex_counts)