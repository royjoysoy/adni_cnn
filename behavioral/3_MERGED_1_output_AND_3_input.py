import pandas as pd
import os

# Get current directory and file paths
current_dir = os.getcwd()
file1 = os.path.join(current_dir, '1_output_MERGED_df_modified_AND_idaSearch_11_18_24.csv')
file2 = os.path.join(current_dir, '3_input_idaSearch_11_18_24_unmatched.csv')

# Read CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 채울 컬럼들 정의
columns_to_fill = ['Subject ID', 'Image ID', 'Age', 'Sex', 'Weight', 
                   'APOE A1', 'APOE A2', 'Global CDR', 'NPI-Q Total Score', 
                   'MMSE Total Score', 'GDSCALE Total Score', 'FAQ Total Score', 
                   'Structure', 'Image Type']

# df2에서 필요한 정보 매핑 생성
fill_data = df2.set_index('Unmatched_Image_ID')

# df1의 빈 칸을 df2의 데이터로 채우기
filled_count = 0
for idx, row in df1.iterrows():
    if pd.isna(row[columns_to_fill]).any():  # 빈 칸이 있는 행 확인
        if row['Image ID'] in fill_data.index:  # 매칭되는 데이터가 있는지 확인
            match_data = fill_data.loc[row['Image ID']]
            for col in columns_to_fill:
                if pd.isna(df1.loc[idx, col]):  # 빈 칸인 경우에만 채우기
                    df1.loc[idx, col] = match_data[col]
            filled_count += 1

# 결과 저장
output_dir = os.path.expanduser('~/Desktop/adni_cnn/behavioral')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '3_output_MERGED_1_output_AND_3_input.csv')
df1.to_csv(output_path, index=False)

print(f"Number of rows with filled data: {filled_count}")
print(f"Merged file saved to: {output_path}")