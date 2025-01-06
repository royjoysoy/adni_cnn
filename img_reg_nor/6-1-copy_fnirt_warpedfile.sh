#!/bin/bash

# Directory setup
BASE_DIR="/ibic/scratch/royseo_workingdir"
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_11-6000"
RSYNC_DIR="${BASE_DIR}/rsync1"

# Create rsync directory if it doesn't exist
mkdir -p "${RSYNC_DIR}"

# Process first 3000 subjects from the subject list
while IFS= read -r subject; do
    # Get base name without extension
    subject_base=$(echo "${subject}" | sed 's/\.nii$//')
    
    # Source and destination paths
    src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_warped_brain.nii.gz"
    dst_file="${RSYNC_DIR}/${subject_base}_warped_brain.nii.gz"
    
    # Copy only if source exists and destination doesn't
    if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
        echo "Copying ${subject_base}_warped_brain.nii.gz"
        cp "${src_file}" "${dst_file}"
    fi
    
done < <(head -n 2990 "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_11-6000_fnirt.log")

echo "Copy process complete"