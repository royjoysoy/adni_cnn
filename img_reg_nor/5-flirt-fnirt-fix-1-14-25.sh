#!/bin/bash
# 1/14/2025 Roy Seo
# Purpose: Combined FLIRT-FNIRT registration script with single interpolation
# Previous workflow issue:
# - Used FLIRT output image as input for FNIRT
# - This caused normalization to the same template twice   
# - Results in cumulative interpolation errors
# Solution:
# - Generate transformation matrix from FLIRT
# - Use transformation matrix as input for FNIRT
# - Avoids double interpolation and runs one reslcing    
# Reference: Email conversation from 1/13/2025 seoroy15@gmail.com            

# Set up FSL environment
export FSLDIR=/usr/local/fsl/6.0
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# Input parameters
input_brain="${norm_dir}/temp_brain_reoriented.nii.gz"
reference="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
ref_mask="$FSLDIR/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
output_base="${subject_dir}/${subject_base}"

# Step 1: Generate FLIRT transformation matrix without applying transformation
echo "Generating FLIRT transformation matrix..."
flirt -in ${input_brain} \
      -ref ${reference} \
      -omat ${output_base}_flirt.mat \
      -dof 12 \
      -searchrx -90 90 \
      -searchry -90 90 \
      -searchrz -90 90 \
      -cost normcorr \
      -nosearch # Prevents output image generation

# Step 2: Run FNIRT using FLIRT matrix as initialization
echo "Running FNIRT with FLIRT initialization..."
fnirt --in=${input_brain} \
      --ref=${reference} \
      --refmask=${ref_mask} \
      --aff=${output_base}_flirt.mat \
      --config=${fnirtconfig} \
      --iout=${output_base}_warped_brain.nii.gz \
      --cout=${output_base}_warp_coef.nii.gz \
      --fout=${output_base}_warp_field.nii.gz \
      --jout=${output_base}_jacobian.nii.gz \
      --intout=${output_base}_intensity_mapping.nii.gz \
      --verbose

# Generate inverse warp (if needed)
echo "Generating inverse warp field..."
invwarp --ref=${input_brain} \
        --warp=${output_base}_warp_coef.nii.gz \
        --out=${output_base}_inv_warp.nii.gz

# Cleanup
rm -f ${output_base}_flirt.mat

# Validate outputs
if [ -f "${output_base}_warped_brain.nii.gz" ]; then
    echo "Registration completed successfully"
    echo "Output files:"
    echo "  - Warped brain: ${output_base}_warped_brain.nii.gz"
    echo "  - Warp coefficients: ${output_base}_warp_coef.nii.gz"
    echo "  - Warp field: ${output_base}_warp_field.nii.gz"
    echo "  - Jacobian: ${output_base}_jacobian.nii.gz"
    echo "  - Inverse warp: ${output_base}_inv_warp.nii.gz"
else
    echo "Error: Registration failed"
    exit 1
fi