#!/bin/bash
# 01-02-2025 Roy Seo, Korea
# This script performs nonlinear registration using FNIRT,following the HCP pipeline parameters. 
# It processes multiple subjects in parallel using SGE's qsub system.
# This script is based on 3-4-fnirt-nonlinear-HCP-param.sh but modified for qsub batch processing to handle multiple subjects

#$ -S /bin/bash
#$ -N fnirt_job
#$ -V
#$ -t 1-6000 
#$ -cwd
#$ -o fnirt_stdout
#$ -e fnirt_stderr

# Setup FSL
export FSLDIR=/usr/local/fsl/6.0
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# Directory setup
BASE_DIR="/ibic/scratch/royseo_workingdir"
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm"
OUTPUT_BASE="${BASE_DIR}/fnirt_output"
SCRIPT_DIR="${BASE_DIR}/scripts"

# Get subject identifier from the list file
subject=$(sed -n -e "${SGE_TASK_ID}p" "${SCRIPT_DIR}/subj_list_ADNI1234_complete.log")

# Define the full input path structure
input_file="${INPUT_BASE}/${subject}/mni152_1mm/${subject}_brain_mni152_1mm.nii.gz"

# Create output directory matching input structure
subject_output_dir="${OUTPUT_BASE}/${subject}"
mkdir -p ${subject_output_dir}

# Use FSL's standard T1_2_MNI152_2mm config (T1_2_MNI152_2mm.cnf contains the parameters that control the multi-resolution progression (4mm->2mm->1mm)
fnirtconfig="${FSLDIR}/etc/flirtsch/T1_2_MNI152_2mm.cnf" 

# Echo debug information
echo "Processing subject: ${subject}"
echo "Input file: ${input_file}"
echo "Output directory: ${subject_output_dir}"

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
      --iout=${subject_output_dir}/${subject}_warped_brain.nii.gz \
      --cout=${subject_output_dir}/${subject}_warp_coef.nii.gz \
      --fout=${subject_output_dir}/${subject}_warp_field.nii.gz \
      --jout=${subject_output_dir}/${subject}_jacobian.nii.gz \
      --verbose

# Check if FNIRT completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: FNIRT failed for subject: ${subject}"
    exit 1
fi

# Generate inverse warp
invwarp --ref=${input_file} \
        --warp=${subject_output_dir}/${subject}_warp_coef.nii.gz \
        --out=${subject_output_dir}/${subject}_inv_warp.nii.gz

# Check if invwarp completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Inverse warp generation failed for subject: ${subject}"
    exit 1
fi

echo "FNIRT registration complete for subject: ${subject}"



