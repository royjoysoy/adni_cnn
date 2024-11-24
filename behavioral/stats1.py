import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('3_ouput_MERGED_df_modified_AND_idaSearch_11_18_24_AND_idaSearch_11_18_24_unmatched.csv')

# 점수 컬럼명
score_columns = ['MMSE Total Score', 'NPI-Q Total Score', 'GDSCALE Total Score', 'FAQ Total Score']

# Age 그룹 생성
df['Age_Group'] = pd.cut(df['Age'], 
                        bins=[0, 60, 70, 80, 90, 100],
                        labels=['<60', '60-70', '70-80', '80-90', '90+'])

# 1. 나이별 분포 (Box Plot)
plt.figure(figsize=(15, 10))
for i, score in enumerate(score_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x='Age_Group', y=score)
    plt.title(f'{score} Distribution by Age')
    plt.xticks(rotation=45)
    plt.ylabel('Score')
plt.tight_layout()
plt.show()

# 2. 진단 그룹별 분포 (Violin Plot)
plt.figure(figsize=(15, 10))
for i, score in enumerate(score_columns, 1):
    plt.subplot(2, 2, i)
    sns.violinplot(data=df, x='Group', y=score)
    plt.title(f'{score} Distribution by Diagnostic Group')
    plt.ylabel('Score')
plt.tight_layout()
plt.show()

# 3. 나이와 진단 그룹을 모두 고려한 분포 (Box Plot with hue)
plt.figure(figsize=(20, 15))
for i, score in enumerate(score_columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x='Age_Group', y=score, hue='Group')
    plt.title(f'{score} Distribution by Age and Diagnostic Group')
    plt.xticks(rotation=45)
    plt.ylabel('Score')
plt.tight_layout()
plt.show()

# 기술통계량도 추가
print("\nDescriptive Statistics by Age Group:")
print(df.groupby('Age_Group')[score_columns].describe())

print("\nDescriptive Statistics by Diagnostic Group:")
print(df.groupby('Group')[score_columns].describe())