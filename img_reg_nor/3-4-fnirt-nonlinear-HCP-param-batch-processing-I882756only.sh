#!/bin/bash
# 1-20-2025 Roy Seo Korea 
# 3-4-fnirt-nonlinear-HCP-param-batch-processing-18001-28001.sh 를 Modified for single image (I882756) processing
# I882756만 안돌아 간것을 발견 (raw_w_acq_date에 카피가 되지 않아있었다,)
# This script performs nonlinear registration using FNIRT, following the HCP pipeline parameters
# Modified for processing a single image without Grid Engine (즉, qsub없이 돌리려고)

# Setup FSL
export FSLDIR=/usr/local/fsl/6.0
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# Directory setup
BASE_DIR="/ibic/scratch/royseo_workingdir"
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_I882756only"

# Get input file name (should be provided as argument)
if [ $# -ne 1 ]; then
    echo "Usage: $0 <subject_name>"
    echo "Example: $0 941_S_6052_2017-07-20_S585807_I882756"
    exit 1
fi

subject_base="$1"

# Define the input/output directory structure
subject_dir="${INPUT_BASE}/${subject_base}/mni152_1mm"
input_file="${subject_dir}/${subject_base}_brain_mni152_1mm.nii.gz"

# Add debug output
echo "Processing subject: ${subject_base}"
echo "Subject directory: ${subject_dir}"
echo "Input file: ${input_file}"

# Check if input directory exists
if [ ! -d "${subject_dir}" ]; then
    echo "ERROR: Subject directory does not exist: ${subject_dir}"
    exit 1
fi

# Use FSL's standard T1_2_MNI152_2mm config
fnirtconfig="${FSLDIR}/etc/flirtsch/T1_2_MNI152_2mm.cnf"

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