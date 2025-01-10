# Directory setup
BASE_DIR="/ibic/scratch/royseo_workingdir"
LINEAR_DIR="${BASE_DIR}/test_11-110"

# Create linear directory if it doesn't exist
mkdir -p "${LINEAR_DIR}"

# Second process: 11-110
echo "Starting copy process for 11-110..."
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_11-110"

while IFS= read -r subject; do
    subject_base=$(echo "${subject}" | sed 's/\.nii$//')
    src_file="${INPUT_BASE}/${subject_base}/mni152_1mm/${subject_base}_brain_mni152_1mm.nii.gz"
    dst_file="${LINEAR_DIR}/${subject_base}_brain_mni152_1mm.nii.gz"
    
    if [ -f "${src_file}" ] && [ ! -f "${dst_file}" ]; then
        echo "Copying ${subject_base}_brain_mni152_1mm.nii.gz"
        cp "${src_file}" "${dst_file}"
    fi
### 이밑에 좀 다름!! 11-110번째 Subjects만 가지고 있는 log 파일 다시 만들기 귀찮아서 head 사용
done < <(head -n 100 "/ibic/scratch/royseo_workingdir/scripts/subj_list_ADNI1234_28001_11-4010.log") 

echo "All copy processes 11-110 complete!"
