"""
Summary:
- Input: 7-add-DX.csv
- 주요 기능:
 1. 원본 데이터의 모든 컬럼 유지하면서 DX_fill 컬럼 추가
    - Output: 8-1-DX_filled1.csv
 2. DX_fill이 빈 row 제거한 버전 생성 
    - Output: 8-2-DX_filled2.csv
 3. 상세 통계 정보 출력
    - 기본 통계 (이미지 수, Subject 수 등)
    - 성별 통계
    - 진단 관련 통계
    - 방문 횟수 통계
    - 이미지 처리 방식 통계
"""

import pandas as pd
import numpy as np

def analyze_dx_changes(df):
   # Subject별 DX 변화 분석
   dx_changes = {}
   for subject in df['Subject'].unique():
       subject_dx = df[df['Subject'] == subject]['DX'].dropna().unique()
       dx_changes[subject] = len(subject_dx) > 1
   
   changed = sum(dx_changes.values())
   return changed, len(dx_changes) - changed

def count_visit_stats(df):
   visit_counts = df.groupby('Subject')['Acq Date'].nunique()
   return visit_counts.describe()

def count_processing_variations(df):
   processing_stats = df.groupby(['Subject', 'Acq Date']).agg({
       'Image Data ID': 'nunique'
   }).reset_index()
   return processing_stats['Image Data ID'].describe()

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
       dx_values = group['DX'].replace('', pd.NA).dropna().unique()
       if len(dx_values) > 0:
           df.loc[group.index, 'DX_fill'] = dx_values[0]
           filled_groups.append((subject, date))
       else:
           empty_groups.append((subject, date))
   
   # 원본 데이터 저장
   df.to_csv('8-1-DX_filled1.csv', index=False)
   
   # DX_fill이 빈 행 제거한 버전 저장
   df_filtered = df[df['DX_fill'] != '']
   df_filtered.to_csv('8-2-DX_filled2.csv', index=False)
   
   # 기본 통계
   total_images = len(df)
   total_subjects = df['Subject'].nunique()
   total_groups = len(filled_groups) + len(empty_groups)
   
   # 성별 통계
   gender_stats = df.groupby('Subject')['Sex'].first().value_counts()

   # DX label이 있는/없는 이미지 수 계산
   labeled_images = len(df[df['DX_fill'] != ''])
   unlabeled_images = len(df[df['DX_fill'] == ''])
   
   # 첫 진단 통계 (각 Subject의 첫 방문 시 Group)
   first_diagnoses = df.sort_values('Acq Date').groupby('Subject')['Group'].first()
   diagnosis_stats = first_diagnoses.value_counts()
   
   # 진단 변화 통계
   changed_dx, unchanged_dx = analyze_dx_changes(df)
   
   # 방문 횟수 통계
   visit_stats = count_visit_stats(df)
   
   # 이미지 처리 방식 통계
   processing_stats = count_processing_variations(df)
   
   # 통계 출력
   print("\n=== 기본 통계 정보 ===")
   print(f"전체 이미지 수: {total_images}")
   print(f"전체 Subject 수: {total_subjects}")
   print(f"유니크한 그룹 (Subject 중 Acq Date이 같은) 수 : {total_groups}")
   print(f"DX label이 있는 그룹 수: {len(filled_groups)}")
   print(f"DX label이 없는 그룹 수: {len(empty_groups)}")
   print(f"DX label이 있는 이미지 수: {labeled_images}")
   print(f"DX label이 없는 이미지 수: {unlabeled_images}")
   
   print("\n=== 성별 통계 ===")
   print(f"여성 수: {gender_stats.get('F', 0)}")
   print(f"남성 수: {gender_stats.get('M', 0)}")
   
   print("\n=== 진단 통계 ===")
   print("첫 진단 분포:")
   for dx, count in diagnosis_stats.items():
       print(f"- {dx}: {count}명")
   print(f"진단명이 바뀐 Subject 수: {changed_dx}")
   print(f"진단명이 바뀌지 않은 Subject 수: {unchanged_dx}")
   
   print("\n=== 방문 통계 ===")
   print("Subject당 방문 횟수:")
   print(visit_stats)
   
   print("\n=== 이미지 처리 방식 통계 ===")
   print("방문당 이미지 처리 방식 수:")
   print(processing_stats)
   
   return df

# 스크립트 실행
if __name__ == "__main__":
   input_file = '7-add-DX.csv'
   result_df = fill_dx_values(input_file)