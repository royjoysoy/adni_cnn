import pandas as pd

# CSV 파일들을 읽어옵니다
ida_search = pd.read_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/1_2_input_idaSearch_11_18_24.csv')
df_modified = pd.read_csv('/Users/test_terminal/Desktop/adni_cnn/behavioral/1_1_input_df_modified.csv')

# IMAGE_ID에서 'I'를 제거하고 숫자만 추출하는 함수
def extract_number(image_id):
    return ''.join(filter(str.isdigit, str(image_id)))

# df_modified의 IMAGE_ID에서 숫자만 추출
df_modified['image_number'] = df_modified['IMAGE_ID'].apply(extract_number)

# ida_search의 Image ID를 문자열로 변환
ida_search['image_number'] = ida_search['Image ID'].astype(str)

# df_modified에는 있지만 ida_search에는 없는 이미지 번호 찾기
unmatched_images = set(df_modified['image_number']) - set(ida_search['image_number'])

# 결과를 정렬하고 콤마로 구분된 문자열로 변환
unmatched_list = sorted(list(unmatched_images))
unmatched_string = ', '.join(unmatched_list)

# 결과 출력
print("\n매칭되지 않은 이미지 ID 개수:", len(unmatched_images))
print("\n매칭되지 않은 이미지 ID 목록:")
print(unmatched_string)

# 결과를 텍스트 파일로 저장
output_path = '/Users/test_terminal/Desktop/adni_cnn/behavioral/2_output_unmatched_images.txt'
with open(output_path, 'w') as f:
    f.write(unmatched_string)

print(f"\n결과가 다음 파일에 저장되었습니다: {output_path}")

# 추가적인 분석 정보 출력
print(f"\n전체 통계:")
print(f"1_1_input_df_modified의 총 이미지 수: {len(df_modified)}")
print(f"1_2_input_ida_search의 총 이미지 수: {len(ida_search)}")
print(f"매칭되지 않은 이미지 비율: {(len(unmatched_images) / len(df_modified)) * 100:.2f}%")