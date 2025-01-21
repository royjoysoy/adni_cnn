# 1-10-25 Roy Seo Korea
# before transfering the nonlinear normalization한 ${subject_base}_brain_mni152_1mm.nii.gz 파일들 to Google Cloud 
# ${subject_base}_brain_mni152_1mm.nii.gz파일들 폴더에 모으기
# flirt 한뒤에 fnirt처리 했지만, fnirt 즉 warped 파일부터 옮겨서 이 스크립트가 6-2이다 (6-1을 fnirt)
# 주의해야 할 것 ###으로 두군데 표시
#!/bin/bash
# 1-20-25: 18001-28001
# Directory setup

BASE_DIR="/ibic/scratch/royseo_workingdir"
LINEAR_DIR="${BASE_DIR}/3_linear_18001-28001"

# Create linear directory if it doesn't exist
mkdir -p "${LINEAR_DIR}"

# Seventh process: 1-10 (1st-6th processes are below)
echo "Starting copy process for 18001-28001..."
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_18001-28001wo18"

while IFS= read -r subject; do
    subject_base=$(echo "${subject}" | sed 's/\.nii$//')
    src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_brain_mni152_1mm.nii.gz"
    dst_file="${LINEAR_DIR}/${subject_base}_brain_mni152_1mm.nii.gz"
    
    if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
        #echo "Copying ${subject_base}_brain_mni152_1mm.nii.gz"
        cp "${src_file}" "${dst_file}"
    fi
done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_18001-28001.log"

echo "Copy process for 18001-28001 complete"




# # Directory setup
# BASE_DIR="/ibic/scratch/royseo_workingdir"
# LINEAR_DIR="${BASE_DIR}/1_linear_1-6000"

# # Create linear directory if it doesn't exist
# mkdir -p "${LINEAR_DIR}"

# # First process: 1-10
# echo "Starting copy process for 1-10..."
# INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_1-10"

# while IFS= read -r subject; do
#     subject_base=$(echo "${subject}" | sed 's/\.nii$//')
#     src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_brain_mni152_1mm.nii.gz"
#     dst_file="${LINEAR_DIR}/${subject_base}_brain_mni152_1mm.nii.gz"
    
#     if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
#         #echo "Copying ${subject_base}_brain_mni152_1mm.nii.gz"
#         cp "${src_file}" "${dst_file}"
#     fi
# done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_1-10.log"

# echo "Copy process for 1-10 complete"

# # Second process: 11-110
# echo "Starting copy process for 11-110..."
# INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_11-110"

# while IFS= read -r subject; do
#     subject_base=$(echo "${subject}" | sed 's/\.nii$//')
#     src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_brain_mni152_1mm.nii.gz"
#     dst_file="${LINEAR_DIR}/${subject_base}_brain_mni152_1mm.nii.gz"
    
#     if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
#         #echo "Copying ${subject_base}_brain_mni152_1mm.nii.gz"
#         cp "${src_file}" "${dst_file}"
#     fi
# ### 이밑에 좀 다름!! 11-110번째 Subjects만 가지고 있는 log 파일 다시 만들기 귀찮아서 head 사용
# done < <(head -n 100 "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_11-4010.log") 

# echo "All copy processes 11-110 complete!"

# # third process: 111-4110
# echo "Starting copy process for 111-4110..."
# ### 이 폴더 이름만 pattern 다름 유의할 것 normalized2mni152_1mm_111-4110아님 152 없음
# INPUT_BASE="${BASE_DIR}/normalized2mni_1mm_111-4110"

# while IFS= read -r subject; do
#     subject_base=$(echo "${subject}" | sed 's/\.nii$//')
#     src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_brain_mni152_1mm.nii.gz"
#     dst_file="${LINEAR_DIR}/${subject_base}_brain_mni152_1mm.nii.gz"
    
#     if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
#         #echo "Copying ${subject_base}_brain_mni152_1mm.nii.gz"
#         cp "${src_file}" "${dst_file}"
#     fi
# done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_111-4110.log"

# echo "All copy processes 111-4110 complete!"

# # fourth process: 4111-6000
# echo "Starting copy process for 4111-6000..."
# INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_4111-6000wo11"

# while IFS= read -r subject; do
#     subject_base=$(echo "${subject}" | sed 's/\.nii$//')
#     src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_brain_mni152_1mm.nii.gz"
#     dst_file="${LINEAR_DIR}/${subject_base}_brain_mni152_1mm.nii.gz"
    
#     if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
#         #echo "Copying ${subject_base}_brain_mni152_1mm.nii.gz"
#         cp "${src_file}" "${dst_file}"
#     fi
# done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_4111-6000.log"

# echo "All copy processes 4111-6000 complete!"


# # Directory setup 2
# BASE_DIR="/ibic/scratch/royseo_workingdir"
# LINEAR_DIR2="${BASE_DIR}/2_linear_6001-18000"

# # Create linear directory if it doesn't exist
# mkdir -p "${LINEAR_DIR2}"

# # fifth process: 6001-12000
# echo "Starting copy process for 6001-12000..."
# INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_6001-12000wo12"

# while IFS= read -r subject; do
#     subject_base=$(echo "${subject}" | sed 's/\.nii$//')
#     src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_brain_mni152_1mm.nii.gz"
#     dst_file="${LINEAR_DIR2}/${subject_base}_brain_mni152_1mm.nii.gz"
    
#     if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
#         #echo "Copying ${subject_base}_brain_mni152_1mm.nii.gz"
#         cp "${src_file}" "${dst_file}"
#     fi
# done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_6001-12000.log"

# echo "All copy processes 6001-12000 complete!"

# # sixth process: 12001-18000
# echo "Starting copy process for 12001-18000..."
# INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_12001-18000wo1"

# while IFS= read -r subject; do
#     subject_base=$(echo "${subject}" | sed 's/\.nii$//')
#     src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_brain_mni152_1mm.nii.gz"
#     dst_file="${LINEAR_DIR2}/${subject_base}_brain_mni152_1mm.nii.gz"
    
#     if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
#         #echo "Copying ${subject_base}_brain_mni152_1mm.nii.gz"
#         cp "${src_file}" "${dst_file}"
#     fi
# done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_12001-18000.log"

# echo "All copy processes 12001-18000 complete!"









