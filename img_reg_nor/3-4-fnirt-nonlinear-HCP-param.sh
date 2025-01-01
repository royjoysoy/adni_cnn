# !/bin/bash

# 01-02-2025 Roy Seo
# This script applies FNIRT after FLIRT normalization 
# It takes the FLIRT-aligned output and performs nonlinear registration to MNI152 1mm space
# The parameters are extracted from HCP, https://github.com/Washington-University/HCPpipelines/blob/master/PreFreeSurfer/PreFreeSurferPipeline.sh

# Uses FSL's standard T1_2_MNI152_2mm.cnf which provides multi-resolution approach:
# 4mm -> 2mm -> 1mm progressive refinement

# Setup FSL
export FSLDIR=/usr/local/fsl/6.0
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}

# Directory and file setup
input_dir="../fnirt_HCP_param_test"
input_file="${input_dir}/033_S_1098_2014-12-16_S36617_I771127_brain_mni152_1mm.nii.gz"
output_dir="${input_dir}/fnirt_output"
mkdir -p ${output_dir}

# Use FSL's standard T1_2_MNI152_2mm config
fnirtconfig="${FSLDIR}/etc/flirtsch/T1_2_MNI152_2mm.cnf"

# Run FNIRT with HCP parameters
fnirt --in=${input_file} \
      --ref=$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz \
      --refmask=$FSLDIR/data/standard/MNI152_T1_1mm_brain_mask.nii.gz \
      --config=${fnirtconfig} \
      --iout=${output_dir}/warped_brain.nii.gz \
      --cout=${output_dir}/warp_coef.nii.gz \
      --fout=${output_dir}/warp_field.nii.gz \
      --jout=${output_dir}/jacobian.nii.gz \
      --verbose      

# Generate inverse warp
invwarp --ref=${input_file} \
        --warp=${output_dir}/warp_coef.nii.gz \
        --out=${output_dir}/inv_warp.nii.gz

echo "FNIRT registration complete. Outputs in: ${output_dir}"