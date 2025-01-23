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

#$ -S /bin/bash
#$ -N flirt_fnirt
#$ -V
#$ -t 1-4
#$ -cwd
#$ -o ./flirt_fnirt_logs/
#$ -e ./flirt_fnirt_logs/

# Set up FSL environment
export FSLDIR=/usr/local/fsl/6.0
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}
fnirtconfig="${FSLDIR}/etc/flirtsch/T1_2_MNI152_2mm.cnf"

# Check if subject list exists
if [ ! -f "subj_list_flirt_fnirt_1_22_2025.log" ]; then
    echo "Error: subj_list_flirt_fnirt_1_22_2025.log not found!"
    exit 1
fi

# Get the current subject from the array job ID and clean it
subject_line=$(sed -n "${SGE_TASK_ID}p" subj_list_flirt_fnirt_1_22_2025.log | tr -d '[:space:]')
echo "Original subject line: ${subject_line}"

# 확장자와 공백 제거
subject_id=$(echo "${subject_line}" | sed 's/\.nii$//' | tr -d '[:space:]')
echo "Cleaned subject ID: ${subject_id}"

# Set directories
norm_dir="/ibic/scratch/royseo_workingdir/testing_flirt_fnirt/${subject_id}/mni152_1mm"
echo "Directory path: ${norm_dir}"

# Check if input brain exists
input_brain="${norm_dir}/temp_brain_reoriented.nii.gz"
echo "Looking for input file at: ${input_brain}"

# Input parameters
reference="$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz"
ref_mask="$FSLDIR/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
output_base="${norm_dir}/${subject_id}"

    # Step 1: Run FLIRT and generate transformation matrix
    echo "Running FLIRT and generating transformation matrix..."
    flirt -in ${input_brain} \
          -ref ${reference} \
          -out ${output_base}_flirt_output.nii.gz \
          -omat ${output_base}_flirt.mat \
          -dof 12 \
          -searchrx -30 30 \
          -searchry -30 30 \
          -searchrz -30 30 \
          -cost normcorr

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

    # Generate inverse warp
    echo "Generating inverse warp field..."
    invwarp --ref=${input_brain} \
            --warp=${output_base}_warp_coef.nii.gz \
            --out=${output_base}_inv_warp.nii.gz

    # Validate outputs
    if [ -f "${output_base}_warped_brain.nii.gz" ]; then
        echo "Registration completed successfully"
        echo "Output files:"
        echo "  - FLIRT output: ${output_base}_flirt_output.nii.gz"
        echo "  - FLIRT matrix: ${output_base}_flirt.mat"
        echo "  - Warped brain: ${output_base}_warped_brain.nii.gz"
        echo "  - Warp coefficients: ${output_base}_warp_coef.nii.gz"
        echo "  - Warp field: ${output_base}_warp_field.nii.gz"
        echo "  - Jacobian: ${output_base}_jacobian.nii.gz"
        echo "  - Inverse warp: ${output_base}_inv_warp.nii.gz"
    else
        echo "Error: Registration failed"
        exit 1
    fi