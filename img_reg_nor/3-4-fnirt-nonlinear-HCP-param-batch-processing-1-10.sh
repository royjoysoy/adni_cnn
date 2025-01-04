#!/bin/bash
# 01-02-2025 Roy Seo, Korea
# This script performs nonlinear registration using FNIRT,following the HCP pipeline parameters. 
# It processes multiple subjects in parallel using SGE's qsub system.
# This script is based on 3-4-fnirt-nonlinear-HCP-param.sh but modified for qsub batch processing to handle multiple subjects
# 01-03-2025 : Modified to save outputs in the same directory as input files (/mni152_1mm)
# 01-04-2025 : the first 10 subjects

#$ -S /bin/bash
#$ -N fnirt_job
#$ -V
#$ -t 1-10
#$ -cwd
#$ -o fnirt_stdout/$JOB_NAME.$TASK_ID.stdout
#$ -e fnirt_stderr/$JOB_NAME.$TASK_ID.stderr


# Setup FSL
export FSLDIR=/usr/local/fsl/6.0
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# Directory setup
BASE_DIR="/ibic/scratch/royseo_workingdir"
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_1-10"
SCRIPT_DIR="${BASE_DIR}/scripts"

# Get subject identifier from the list file
subject=$(sed -n -e "${SGE_TASK_ID}p" "${SCRIPT_DIR}/subj_list_ADNI1234_28001_1-10.log")

# Create symbolic links with both task ID and subject ID
ln -sf fnirt_stdout/$JOB_NAME.$TASK_ID.stdout fnirt_stdout/$JOB_NAME.task${TASK_ID}_${subject}.stdout
ln -sf fnirt_stderr/$JOB_NAME.$TASK_ID.stderr fnirt_stderr/$JOB_NAME.task${TASK_ID}_${subject}.stderr

# Define the input/output directory structure
subject_dir="${INPUT_BASE}/${subject}/mni152_1mm"
input_file="${subject_dir}/${subject}_brain_mni152_1mm.nii.gz"

# Use FSL's standard T1_2_MNI152_2mm config
fnirtconfig="${FSLDIR}/etc/flirtsch/T1_2_MNI152_2mm.cnf" 

# Echo debug information
echo "Processing subject: ${subject}"
echo "Input file: ${input_file}"
echo "Output directory: ${subject_dir}"

# Check if input file exists
if [ ! -f "${input_file}" ]; then
    echo "ERROR: Input file does not exist: ${input_file}"
    exit 1
fi

# Run FNIRT with HCP parameters
fnirt --in=${input_file} \
      --ref=$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz \
      --refmask=$FSLDIR/data/standard/MNI152_T1_1mm_brain_mask.nii.gz \
      --config=${fnirtconfig} \
      --iout=${subject_dir}/${subject}_warped_brain.nii.gz \
      --cout=${subject_dir}/${subject}_warp_coef.nii.gz \
      --fout=${subject_dir}/${subject}_warp_field.nii.gz \
      --jout=${subject_dir}/${subject}_jacobian.nii.gz \
      --verbose

# Check if FNIRT completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: FNIRT failed for subject: ${subject}"
    exit 1
fi

# Generate inverse warp
invwarp --ref=${input_file} \
        --warp=${subject_dir}/${subject}_warp_coef.nii.gz \
        --out=${subject_dir}/${subject}_inv_warp.nii.gz

# Check if invwarp completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Inverse warp generation failed for subject: ${subject}"
    exit 1
fi

echo "FNIRT registration complete for subject: ${subject}"



