# Directory setup
BASE_DIR="/ibic/scratch/royseo_workingdir"
LINEAR_DIR="${BASE_DIR}/test_111-4110"

# Create linear directory if it doesn't exist
mkdir -p "${LINEAR_DIR}"

# third process: 111-4110
echo "Starting copy process for 111-4110..."
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_111-4110"

while IFS= read -r subject; do
    subject_base=$(echo "${subject}" | sed 's/\.nii$//')
    src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_brain_mni152_1mm.nii.gz"
    dst_file="${LINEAR_DIR}/${subject_base}_brain_mni152_1mm.nii.gz"
    
    if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
        echo "Copying ${subject_base}_brain_mni152_1mm.nii.gz"
        cp "${src_file}" "${dst_file}"
    fi
done < "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_111-4110.log"

echo "All copy processes 111-4110 complete!"