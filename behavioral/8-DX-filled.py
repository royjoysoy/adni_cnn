"""
Summary:
- Input: 7-add-DX.csv (컴마로 구분된 CSV 파일)
- 주요 기능:
  1. 원본 데이터의 모든 컬럼 유지
  2. Subject와 Acq Date가 같은 그룹 내에서 DX 값으로 DX_fill 컬럼 채우기
  3. DX가 없는 그룹의 통계 출력
- Output: 8-DX_filled.csv
"""

import pandas as pd

def fill_dx_values(file_path):
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    
    # DX_fill 컬럼 초기화
    df['DX_fill'] = ''
    
    # 각 그룹별로 DX값 채우기
    empty_groups = []
    filled_groups = []
    
    # Subject와 Acq Date로 그룹화
    for (subject, date), group in df.groupby(['Subject', 'Acq Date']):
        # 그룹 내 DX값이 하나라도 있는지 확인 (공백과 NaN 처리)
        dx_values = group['DX'].replace('', pd.NA).dropna().unique()
        if len(dx_values) > 0:
            # DX값이 있는 경우, 해당 값으로 채우기
            df.loc[group.index, 'DX_fill'] = dx_values[0]
            filled_groups.append((subject, date))
        else:
            # DX값이 없는 경우
            empty_groups.append((subject, date))
    
    # 결과를 새로운 CSV 파일로 저장
    df.to_csv('8-DX_filled.csv', index=False)
    
    # 통계 출력
    total_images = len(df)
    total_subjects = df['Subject'].nunique()
    total_groups = len(filled_groups) + len(empty_groups)
    
    print(f"\n=== 통계 정보 ===")
    print(f"전체 이미지 수: {total_images}")
    print(f"전체 Subject 수: {total_subjects}")
    print(f"유니크한 그룹 수 (Subject+날짜 조합): {total_groups}")
    print(f"DX값이 없는 그룹 수: {len(empty_groups)}")
    print(f"DX값이 있는 그룹 수: {len(filled_groups)}")
    
    return df

# 스크립트 실행
if __name__ == "__main__":
    input_file = '7-add-DX.csv'
    result_df = fill_dx_values(input_file)