import pandas as pd
import os

# CSV 파일들을 읽어옵니다
ida_search = pd.read_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/idaSearch_11_18_2024.csv')
ida_search_unmatched = pd.read_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/idaSearch_11_18_2024_unmatched.csv')
df_modified = pd.read_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/df_modified.csv')

# IMAGE_ID에서 'I'를 제거하고 숫자만 추출하는 함수
def extract_number(image_id):
    return ''.join(filter(str.isdigit, str(image_id)))

# df_modified의 IMAGE_ID에서 숫자만 추출하여 임시 칼럼 생성
df_modified['temp_image_number'] = df_modified['IMAGE_ID'].apply(extract_number)

# ida_search와 ida_search_unmatched의 Image ID를 문자열로 변환하고 임시 칼럼 생성
ida_search['temp_image_number'] = ida_search['Image ID'].astype(str)
ida_search_unmatched['temp_image_number'] = ida_search_unmatched['Image ID'].astype(str)

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

# 먼저 기존 데이터로 매칭
merged_df = pd.merge(
    df_modified,
    ida_search[['temp_image_number'] + columns_to_add],
    left_on=['PTID', 'temp_image_number'],
    right_on=['Subject ID', 'temp_image_number'],
    how='left'
)

# 데이터 소스 칼럼 추가
merged_df['Data_Source'] = 'original'

# unmatched 데이터로 두 번째 매칭
unmatched_merge = pd.merge(
    df_modified[merged_df['Subject ID'].isna()],  # 첫 번째 매칭에서 매칭되지 않은 행만 선택
    ida_search_unmatched[['temp_image_number'] + columns_to_add],
    left_on=['PTID', 'temp_image_number'],
    right_on=['Subject ID', 'temp_image_number'],
    how='left'
)

# unmatched 데이터의 소스 표시
unmatched_merge['Data_Source'] = 'from_unmatched'

# 두 데이터프레임 합치기
# 먼저 기존 매칭된 행들 선택
matched_rows = merged_df[merged_df['Subject ID'].notna()]

# unmatched에서 매칭된 행들 선택
unmatched_matched_rows = unmatched_merge[unmatched_merge['Subject ID'].notna()]

# 여전히 매칭되지 않은 행들 선택 (원본에서 NaN인 행들)
still_unmatched = merged_df[merged_df['Subject ID'].isna()]
still_unmatched['Data_Source'] = None  # 매칭되지 않은 행들의 Data_Source는 None으로 설정

# 모든 데이터프레임 합치기
final_df = pd.concat([matched_rows, unmatched_matched_rows, still_unmatched], axis=0)

# 임시 칼럼 제거
final_df = final_df.drop(['temp_image_number'], axis=1)

# 인덱스 재설정
final_df = final_df.reset_index(drop=True)

# 저장할 디렉토리 경로
save_dir = os.path.expanduser('~/Desktop/adni_cnn/behavioral')

# 디렉토리가 없으면 생성
os.makedirs(save_dir, exist_ok=True)

# 결과를 CSV 파일로 저장
output_path = os.path.join(save_dir, 'merged_df_modified_idaSearch_11_18_2024.csv')
final_df.to_csv(output_path, index=False)

print(f"파일이 성공적으로 저장되었습니다: {output_path}")
print(f"생성된 파일의 크기: {final_df.shape}")

# 매칭 결과 확인을 위한 추가 정보 출력
print(f"\n원본 파일 크기:")
print(f"df_modified.csv: {df_modified.shape}")
print(f"idaSearch_11_18_2024.csv: {ida_search.shape}")
print(f"idaSearch_11_18_2024_unmatched.csv: {ida_search_unmatched.shape}")
print(f"\n매칭 결과:")
print(f"기존 데이터에서 매칭된 행 수: {len(matched_rows)}")
print(f"unmatched 데이터에서 매칭된 행 수: {len(unmatched_matched_rows)}")
print(f"여전히 매칭되지 않은 행 수: {len(still_unmatched)}")