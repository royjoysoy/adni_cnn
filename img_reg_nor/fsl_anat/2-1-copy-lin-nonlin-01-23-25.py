# 01-23-2025 Korea Roy Seo
'''
1. 소스 디렉토리: /ibic/scratch/royseo_workingdir/fsl_anat/raw
   * 이 디렉토리 안에는 약 2409개의 '.anat' 폴더들이 있습니다
   * 예시 폴더명: 941_S_1363_2007-03-12_S28008_I63896.anat
2. 대상 디렉토리:
   * Linear 파일용: /ibic/scratch/royseo_workingdir/fsl_anat/processed/linear
   * Nonlinear 파일용: /ibic/scratch/royseo_workingdir/fsl_anat/processed/nonlinear
3. 복사할 파일:
   * T1_to_MNI_lin.nii.gz → linear 폴더로
   * T1_to_MNInonlin.nii.gz → nonlinear 폴더로
4. 파일 이름 변경 규칙:
   * 원본 폴더명에서 '.anat'을 제거하고 밑줄()을 추가한 후 원본 파일명을 붙입니다
   * 예시:
      * 941_S_1363_2007-03-12_S28008_I63896_T1_to_MNI_lin.nii.gz
      * 941_S_1363_2007-03-12_S28008_I63896_T1_to_MNI_nonlin.nii.gz
스크립트는 다음을 포함해야 합니다:
* 존재하지 않는 대상 디렉토리 자동 생성
* 파일 복사 및 이름 변경 진행상황 표시
* 에러 처리 (파일이나 디렉토리가 없는 경우 등)
'''

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def setup_directories(linear_dir, nonlinear_dir):
    """Create target directories if they don't exist."""
    for dir_path in [linear_dir, nonlinear_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def process_files(source_dir, linear_dir, nonlinear_dir):
    """Process .anat directories and copy files with new names."""
    # Get list of .anat directories
    anat_dirs = [d for d in os.listdir(source_dir) if d.endswith('.anat')]
    
    if not anat_dirs:
        raise FileNotFoundError(f"No .anat directories found in {source_dir}")
    
    # Process each directory
    for anat_dir in tqdm(anat_dirs, desc="Processing directories"):
        base_name = anat_dir.replace('.anat', '')
        source_path = Path(source_dir) / anat_dir
        
        # Define file mappings (source file -> (target dir, new name))
        file_mappings = {
            'T1_to_MNI_lin.nii.gz': (
                linear_dir,
                f"{base_name}_T1_to_MNI_lin.nii.gz"
            ),
            'T1_to_MNI_nonlin.nii.gz': (
                nonlinear_dir,
                f"{base_name}_T1_to_MNI_nonlin.nii.gz"
            )
        }
        
        # Process each file
        for source_file, (target_dir, new_name) in file_mappings.items():
            source_file_path = source_path / source_file
            target_file_path = Path(target_dir) / new_name
            
            try:
                if source_file_path.exists():
                    shutil.copy2(source_file_path, target_file_path)
                else:
                    print(f"Warning: {source_file} not found in {anat_dir}")
            except Exception as e:
                print(f"Error processing {source_file} from {anat_dir}: {str(e)}")

def main():
    # Define directories
    source_dir = "/ibic/scratch/royseo_workingdir/fsl_anat/raw"
    linear_dir = "/ibic/scratch/royseo_workingdir/fsl_anat/processed/linear"
    nonlinear_dir = "/ibic/scratch/royseo_workingdir/fsl_anat/processed/nonlinear"
    
    try:
        setup_directories(linear_dir, nonlinear_dir)
        process_files(source_dir, linear_dir, nonlinear_dir)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()