#!/bin/bash
# 01-02-2025 Roy Seo, Korea
# This script performs nonlinear registration using FNIRT,following the HCP pipeline parameters. 
# It processes multiple subjects in parallel using SGE's qsub system.
# This script is based on 3-4-fnirt-nonlinear-HCP-param.sh but modified for qsub batch processing to handle multiple subjects
# 01-03-2025 : Modified to save outputs in the same directory as input files (/mni152_1mm)
# 01-04-2025 : the subjects 11-6000
# 01-06-2025 : the subjects 6001-12000 
# 01-07-2025 : the subjects 12001-18000
# 01-19-2025 : the subjects 18001-28001
# 01-20-2025 : the subjects 18001-28002 : 중간에 qsub 을 또 눌러버려서, 다시 돌리는 김에 누락되었던 이미지 "I1882576"도 함께 돌림. 
# 01-21-2025 : qsub 돌리다가 멈춰서 다시 돌림

# Usage: example
# bash ./4-1-fnirt-nonlinear-HCP-param-batch-processing-submit-fnirt-jobs-18001-28002_1_21_2025.sh

#$ -S /bin/bash
#$ -N fnirt_job
#$ -V
#$ -t 1-10002
#$ -cwd
#$ -o fnirt_stdout_18001-28002/$JOB_NAME.$TASK_ID.stdout
#$ -e fnirt_stderr_18001-28002/$JOB_NAME.$TASK_ID.stderr


# Setup FSL
export FSLDIR=/usr/local/fsl/6.0
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# Directory setup
BASE_DIR="/ibic/scratch/royseo_workingdir"
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_18001-28002"
SCRIPT_DIR="${BASE_DIR}/scripts"

# Debug line for subject list file
echo "Reading subject from: ${SCRIPT_DIR}/subj_list_ADNI1234_28002_18001-28002_1_20_2025.log"

# Get subject identifier from the list file
subject=$(sed -n -e "${SGE_TASK_ID}p" "${SCRIPT_DIR}/subj_list_ADNI1234_28002_18001-28002_1_20_2025.log")

# Create symbolic links with both task ID and subject ID
ln -sf $JOB_NAME.$TASK_ID.stdout fnirt_stdout_18001-28002/$JOB_NAME.task${TASK_ID}_${subject}.stdout
ln -sf $JOB_NAME.$TASK_ID.stderr fnirt_stderr_18001-28002/$JOB_NAME.task${TASK_ID}_${subject}.stderr

# Define the input/output directory structure
subject_base=$(echo "${subject}" | sed 's/\.nii$//')
# Clean up any potential double slashes and ensure proper path construction
subject_dir=$(echo "${INPUT_BASE}/${subject_base}/mni152_1mm" | sed 's#//#/#g')
input_file=$(echo "${subject_dir}/${subject_base}_brain_mni152_1mm.nii.gz" | sed 's#//#/#g')

# Add debug output
echo "subject: ${subject}"
echo "subject_base: ${subject_base}"
echo "subject_dir: ${subject_dir}"
echo "input_file: ${input_file}"

# Add directory existence check
if [ ! -d "${subject_dir}" ]; then
    echo "ERROR: Subject directory does not exist: ${subject_dir}"
    exit 1
fi

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
      --iout=${subject_dir}/${subject_base}_warped_brain.nii.gz \
      --cout=${subject_dir}/${subject_base}_warp_coef.nii.gz \
      --fout=${subject_dir}/${subject_base}_warp_field.nii.gz \
      --jout=${subject_dir}/${subject_base}_jacobian.nii.gz \
      --verbose

# Check if FNIRT completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: FNIRT failed for subject: ${subject_base}"
    exit 1
fi

# Generate inverse warp
invwarp --ref=${input_file} \
        --warp=${subject_dir}/${subject_base}_warp_coef.nii.gz \
        --out=${subject_dir}/${subject_base}_inv_warp.nii.gz

# Check if invwarp completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Inverse warp generation failed for subject: ${subject_base}"
    exit 1
fi

echo "FNIRT registration complete for subject: ${subject_base}"