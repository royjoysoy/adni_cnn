import pandas as pd
import os

# CSV 파일들을 읽어옵니다
ida_search = pd.read_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/1_2_input_idaSearch_11_18_24.csv')
df_modified = pd.read_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/1_1_input_df_modified.csv')

# IMAGE_ID에서 'I'를 제거하고 숫자만 추출하는 함수
def extract_number(image_id):
    return ''.join(filter(str.isdigit, str(image_id)))

# df_modified의 IMAGE_ID에서 숫자만 추출하여 임시 칼럼 생성
df_modified['temp_image_number'] = df_modified['IMAGE_ID'].apply(extract_number)

# ida_search의 Image ID를 문자열로 변환하고 임시 칼럼 생성
ida_search['temp_image_number'] = ida_search['Image ID'].astype(str)

# 매칭할 칼럼들
columns_to_add = [
    'Subject ID', 
    'Image ID', 
    'Age',
    'Sex', 
    'Weight', 
    'APOE A1', 
    'APOE A2',  
    'Global CDR', 
    'NPI-Q Total Score', 
    'MMSE Total Score', 
    'GDSCALE Total Score',
    'FAQ Total Score',
    'Structure',
    'Image Type'
]

# 두 데이터프레임을 PTID/Subject ID와 이미지 번호로 매칭
merged_df = pd.merge(
    df_modified,
    ida_search[['temp_image_number'] + columns_to_add],
    left_on=['PTID', 'temp_image_number'],
    right_on=['Subject ID', 'temp_image_number'],
    how='left'
)

# 임시 칼럼 제거
merged_df = merged_df.drop(['temp_image_number'], axis=1)

# 저장할 디렉토리 경로
save_dir = os.path.expanduser('~/Desktop/adni_cnn/behavioral')

# 디렉토리가 없으면 생성
os.makedirs(save_dir, exist_ok=True)

# 결과를 CSV 파일로 저장
output_path = os.path.join(save_dir, '1_output_MERGED_df_modified_AND_idaSearch_11_18_24.csv')
merged_df.to_csv(output_path, index=False)

print(f"파일이 성공적으로 저장되었습니다: {output_path}")
print(f"생성된 파일의 크기: {merged_df.shape}")

# 매칭 결과 확인을 위한 추가 정보 출력
print(f"\n원본 파일 크기:")
print(f"df_modified.csv: {df_modified.shape}")
print(f"idaSearch_11_18_2024.csv: {ida_search.shape}")
print(f"\n매칭된 행의 수: {merged_df.dropna(subset=['Subject ID']).shape[0]}")