# 1-10-25 Roy Seo Korea
# before transfering the nonlinear normalization한 ${subject_base}_warped_brain.nii.gz 파일들 to Google Cloud 
# ${subject_base}_warped_brain.nii.gz 파일들 폴더에 모으기
#!/bin/bash
# 1-20-25: 18001-28001

# Directory setup
BASE_DIR="/ibic/scratch/royseo_workingdir"
RSYNC_DIR="${BASE_DIR}/3_nonlinear_18001-28001"

# Create rsync directory if it doesn't exist
mkdir -p "${RSYNC_DIR}"

# Third process: 6001-12000 (First and second ones are below)
echo "Starting copy process for 18001-28001..."
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_18001-28001wo18"

while IFS= read -r subject; do
    subject_base=$(echo "${subject}" | sed 's/\.nii$//')
    src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_warped_brain.nii.gz"
    dst_file="${RSYNC_DIR}/${subject_base}_warped_brain.nii.gz"
    
    if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
        echo "Copying ${subject_base}_warped_brain.nii.gz"
        cp "${src_file}" "${dst_file}"
    fi
done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_18001-28001.log"

echo "Copy process for 18001-28001 complete"

# # Directory setup
# BASE_DIR="/ibic/scratch/royseo_workingdir"
# RSYNC_DIR="${BASE_DIR}/2_nonlinear_6001-18000"

# # Create rsync directory if it doesn't exist
# mkdir -p "${RSYNC_DIR}"

# # First process: 6001-12000
# echo "Starting copy process for 6001-12000..."
# INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_6001-12000wo12"

# while IFS= read -r subject; do
#     subject_base=$(echo "${subject}" | sed 's/\.nii$//')
#     src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_warped_brain.nii.gz"
#     dst_file="${RSYNC_DIR}/${subject_base}_warped_brain.nii.gz"
    
#     if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
#         echo "Copying ${subject_base}_warped_brain.nii.gz"
#         cp "${src_file}" "${dst_file}"
#     fi
# done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_6001-12000.log"

# echo "Copy process for 6001-12000 complete"

# # Second process: 12001-18000
# echo "Starting copy process for 12001-18000..."
# INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_12001-18000wo1"

# while IFS= read -r subject; do
#     subject_base=$(echo "${subject}" | sed 's/\.nii$//')
#     src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_warped_brain.nii.gz"
#     dst_file="${RSYNC_DIR}/${subject_base}_warped_brain.nii.gz"
    
#     if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
#         echo "Copying ${subject_base}_warped_brain.nii.gz"
#         cp "${src_file}" "${dst_file}"
#     fi
# done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_12001-18000.log"

# echo "All copy processes complete!"







#!/bin/bash
# # Directory setup
# BASE_DIR="/ibic/scratch/royseo_workingdir"
# INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_11-6000"
# RSYNC_DIR="${BASE_DIR}/1_nonlinear_1-6000"

# # Create rsync directory if it doesn't exist
# mkdir -p "${RSYNC_DIR}"

# # Process first 3000 subjects from the subject list
# while IFS= read -r subject; do
#     # Get base name without extension
#     subject_base=$(echo "${subject}" | sed 's/\.nii$//')
    
#     # Source and destination paths
#     src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_warped_brain.nii.gz"
#     dst_file="${RSYNC_DIR}/${subject_base}_warped_brain.nii.gz"
    
#     # Copy only if source exists and destination doesn't
#     if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
#         echo "Copying ${subject_base}_warped_brain.nii.gz"
#         cp "${src_file}" "${dst_file}"
#     fi
    
# #done < <(head -n 2990 "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_11-6000_fnirt.log")
# # the rest of them
# done < <(tail -n +2991 "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_11-6000_fnirt.log") 

# echo "Copy process complete"