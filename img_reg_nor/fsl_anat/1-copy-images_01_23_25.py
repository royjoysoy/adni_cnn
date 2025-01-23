'''
01-23-25 Roy Seo Korea
source folder: /ibic/scratch/royseo_workingdir/raw_w_acq_date 
source: fsl_anat_subj_list_2409_raw.log; 
          -  복사해올 파일 리스트를 가지고 있음 
          -  (예: 002_S_0295_2006-04-18_S13408_I45107.nii)

Desitnation directory: /ibic/scratch/royseo_workingdir/fsl_anat/raw
'''

import os
import shutil

# 경로 설정
source_dir = "/ibic/scratch/royseo_workingdir/raw_w_acq_date"
destination_dir = "/ibic/scratch/royseo_workingdir/fsl_anat/raw"
file_list_path = "fsl_anat_subj_list_2409_raw.log"

# 복사할 파일 목록 읽기
with open(file_list_path, 'r') as f:
    file_list = [line.strip() for line in f if line.strip()]  # 빈 줄 제거

# 출력 디렉터리 확인 및 생성
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# 파일 복사
copied_files = 0
missing_files = []

for file_name in file_list:
    source_path = os.path.join(source_dir, file_name)
    destination_path = os.path.join(destination_dir, file_name)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        copied_files += 1
    else:
        missing_files.append(file_name)

# 결과 출력
print(f"총 {copied_files}개의 파일을 복사했습니다.")
if missing_files:
    print(f"다음 {len(missing_files)}개의 파일을 찾을 수 없었습니다:")
    for missing_file in missing_files:
        print(missing_file)
