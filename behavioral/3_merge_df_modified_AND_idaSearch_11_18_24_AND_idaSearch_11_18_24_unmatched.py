import pandas as pd

# 파일 경로 설정
file1 = "/Users/test_terminal/Desktop/adni_cnn/behavioral/1_1_input_df_modified.csv"
file2 = "/Users/test_terminal/Desktop/adni_cnn/behavioral/1_2_input_idaSearch_11_18_24.csv"
file3 = "/Users/test_terminal/Desktop/adni_cnn/behavioral/3_input_idaSearch_11_18_24_unmatched.csv"
output_file = "/Users/test_terminal/Desktop/adni_cnn/behavioral/1_ouput_MERGED_df_modified_AND_idaSearch_11_18_24_chatGPTversion.csv"

# CSV 파일 읽기
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# 'IMAGE_ID'에서 첫 글자 'I' 제거
df1['IMAGE_ID_modified'] = df1['IMAGE_ID'].str[1:]  # 첫 글자 'I' 제거
df1['IMAGE_ID_modified'] = df1['IMAGE_ID_modified'].astype(str)

# 'Image ID'를 문자열로 변환
df2['Image ID'] = df2['Image ID'].astype(str)
df3['Image ID'] = df3['Image ID'].astype(str)

# merge 수행
merged_df = pd.merge(
    df1,
    df2,
    left_on="IMAGE_ID_modified",
    right_on="Image ID",
    how="left"
)

# merge되지 않은 subject 식별
unmerged_subjects = merged_df[merged_df['Image ID'].isnull()]
unmerged_ids = unmerged_subjects['IMAGE_ID_modified'].dropna().tolist()

# unmerged IDs 출력 (앞글자 'I' 제거된 상태)
print(f"Unmerged Image IDs: {', '.join(unmerged_ids)}")

# unmerged IDs를 3_input_idaSearch_11_18_24_unmatched.csv에서 다시 merge 시도
unmerged_ids_df = pd.DataFrame({'IMAGE_ID_modified': unmerged_ids})
second_merge_df = pd.merge(
    unmerged_ids_df,
    df3,
    left_on="IMAGE_ID_modified",
    right_on="Image ID",
    how="left"
)

# 추가 merge 결과를 기존 merge 결과와 결합
final_merge_df = pd.merge(
    merged_df,
    second_merge_df,
    on="IMAGE_ID_modified",
    how="left",
    suffixes=('', '_from_unmatched')
)

# 최종 결과 저장
final_merge_df.to_csv(output_file, index=False)
print(f"Merged file saved as '{output_file}'")

# 최종 merge되지 않은 항목 확인
final_unmerged = final_merge_df[final_merge_df['Image ID'].isnull() & final_merge_df['Image ID_from_unmatched'].isnull()]
final_unmerged_ids = final_unmerged['IMAGE_ID_modified'].dropna().tolist()
print(f"Final unmerged Image IDs: {', '.join(final_unmerged_ids)}")
