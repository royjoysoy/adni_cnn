#!/bin/bash
# 01-05-2025 Check FNIRT processing progress by looking for warped_brain.nii.gz files

# Check for correct number of arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <subject_list_file> <base_directory>"
    echo "Example: $0 subj_list_ADNI1234_28001_1-10.log /ibic/scratch/royseo_workingdir/normalized2mni152_1mm_1-10"
    exit 1
fi

SUBJECT_LIST=$1
BASE_DIR=$2
total=0
complete=0

# Read through the subject list
while IFS= read -r subject; do
    total=$((total + 1))
    
    # Remove .nii extension if present
    subject_base=$(echo "${subject}" | sed 's/\.nii$//')
    
    # Check if warped brain file exists
    if [ -f "${BASE_DIR}/${subject_base}/mni152_1mm/${subject_base}_warped_brain.nii.gz" ]; then
        complete=$((complete + 1))
    else
        echo "${subject_base} not complete"
    fi
done < "$SUBJECT_LIST"

# Calculate completion percentage
percentage=$(echo "scale=2; $complete * 100 / $total" | bc)

echo "----------------------------------------"
echo "Summary:"
echo "Total subjects: $total"
echo "Complete: $complete"
echo "Incomplete: $((total - complete))"
echo "Completion rate: ${percentage}%"
echo "----------------------------------------"
echo "Check complete!"