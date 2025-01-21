# 12-7-2024 Roy Seo
# 3-1 run_fs_modified_plus_Normazlied2mni152_plus_Convert2nii.sh의 2 output files
# normalize가 되긴 되었는데 brain이 조금 위로 올라와있고 크기가 안맞아서 FSL로 다시 시도
# non-linear (fnirt)은 너무 시간이 오래걸려서 flirt만 사용하여 시도
# 1-20-2025: 누락된 I882756만 돌리기 위해 이 3-3-run_fs_modified_plus_normalized2mni152_1mm_fsl_flirt_linear_transform_I882756only.sh를 만들었음
# 1-20-2025: I882756only폴더를 만들고  "/ibic/scratch/royseo_workingdir/I882756only" 
#            돌릴 파일을 (941_S_6052_2017-07-20_S585807_I882756.nii) raw_w_acq_date 에서 카피해와서 II882756only 안에 넣음
# Usage: bash + script 이름 + 돌릴 파일 이름
# bash 3-3-run_fs_modified_plus_normalized2mni152_1mm_fsl_flirt_linear_transform_I882756only.sh 941_S_6052_2017-07-20_S585807_I882756.nii


#!/bin/bash
# $1 will be the input file name (full name including .nii)

# Set up FreeSurfer environment
export FREESURFER_HOME=/usr/local/freesurfer/stable7
source /usr/local/freesurfer/stable7/SetUpFreeSurfer.sh
export RECON_ALL=/usr/local/freesurfer/stable7/bin/recon-all

# Set up FSL environment
export FSLDIR=/usr/local/fsl/6.0
source ${FSLDIR}/etc/fslconf/fsl.sh
export PATH=${FSLDIR}/bin:${PATH}


# Set input and output directories
raw_dir="/ibic/scratch/royseo_workingdir/I882756only" 
export SUBJECTS_DIR="//ibic/scratch/royseo_workingdir/normalized2mni152_1mm_I882756only"

# Get the filename and parse components
file_name=$(basename "$1") # nii 포함된 파일이름만 추출, 경로는 없애고
# Extract components using _ as delimiter
site_id=$(echo $file_name | cut -d'_' -f1)
session_num=$(echo $file_name | cut -d'_' -f3)
scan_date=$(echo $file_name | cut -d'_' -f4)
subject_num=$(echo $file_name | cut -d'_' -f5)
image_id=$(echo $file_name | cut -d'_' -f6 | cut -d'.' -f1) 

# Create a subject name using image_id as the primary identifier
# 전체 파일명을 그대로 사용하되, .nii 확장자만 제거
subj_name=$(echo $file_name | cut -d'.' -f1)

# Full path to input file
t1_file="${raw_dir}/${file_name}"

# Stage 1: Initial Setup and Input Validation
echo "=== Stage 1: Initial Setup and Input Validation ==="
echo "Processing file: ${file_name}"
echo "Site ID: ${site_id}"
echo "Session Number: ${session_num}"
echo "Scan Date: ${scan_date}"
echo "Subject Number: ${subject_num}"
echo "Image ID: ${image_id}"
echo "FreeSurfer Subject Name: ${subj_name}"
echo "Input file path: ${t1_file}"
echo "Output directory: ${SUBJECTS_DIR}"

# Check if input file exists
if [ ! -f "${t1_file}" ]; then
    echo "Error: Input file ${t1_file} not found!"
    exit 1
fi

# Check if output directory exists, if not create it
if [ ! -d "${SUBJECTS_DIR}" ]; then
    echo "Creating output directory: ${SUBJECTS_DIR}"
    mkdir -p "${SUBJECTS_DIR}"
fi

#Stage 2: FreeSurfer Preprocessing
echo "=== Stage 2: FreeSurfer Preprocessing ==="
echo "Running FreeSurfer recon-all..."
recon-all -s $subj_name -i ${t1_file} -autorecon1 -gcareg -canorm -careg -rmneck -skull-lta -calabel -normalization2

# Stage 3: MNI152 Normalization Setup
echo "=== Stage 3: MNI152 Normalization Setup ==="
echo "Initial preprocessing complete. Starting MNI152 normalization..."

# Define the path to the brain.mgz file that was created
brain_mgz="${SUBJECTS_DIR}/${subj_name}/mri/brain.mgz"

# Check if brain.mgz exists
if [ ! -f "${brain_mgz}" ]; then
   echo "Error: brain.mgz not found at ${brain_mgz}"
   exit 1
fi

# Create a directory for the normalized outputs
norm_dir="${SUBJECTS_DIR}/${subj_name}/mni152_1mm"
mkdir -p ${norm_dir}

# Create new filenames using original input filename without extension
base_filename=${file_name%.*}  # Removes .nii extension
output_mgz="${base_filename}_brain_mni152_1mm.mgz"
output_nii="${base_filename}_brain_mni152_1mm.nii.gz"

# Create log files
vol2vol_log="${norm_dir}/mri_vol2vol.log"
convert_log="${norm_dir}/mri_convert.log"

# Stage 4: MNI152 Normalization Process
echo "=== Stage 4: MNI152 Normalization Process ==="
echo "Performing spatial normalization to MNI152 space..."
echo "Log will be saved to: ${vol2vol_log}"
(
  echo "Stage 4: Command started at: $(date)"

#Convert MGZ to NIFTI for FLIRT
mri_convert ${brain_mgz} ${norm_dir}/temp_brain.nii.gz

# Check the orientatino of the input image
mri_info ${norm_dir}/temp_brain.nii.gz 

# First robustly find the center of the image
robustfov -i ${norm_dir}/temp_brain.nii.gz -r ${norm_dir}/temp_brain_robust.nii.gz

# Then reorient to standard space
fslreorient2std ${norm_dir}/temp_brain_robust.nii.gz ${norm_dir}/temp_brain_reoriented.nii.gz

# Finally run FLIRT
flirt -in ${norm_dir}/temp_brain_reoriented.nii.gz \
      -ref $FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz \
      -out ${norm_dir}/${output_nii} \
      -dof 12 \
      -searchrx -90 90 \
      -searchry -90 90 \
      -searchrz -90 90 \
      -cost normcorr \
      -interp spline

# Clean up temporary file
rm ${norm_dir}/temp_brain.nii.gz

echo "Stage 4: Command completed at: $(date)"
) | tee "${vol2vol_log}"

# # Stage 5: Convert normalized MGZ to NIFTI
# echo "=== Stage 5: NIFTI Conversion ==="
# echo "Converting normalized MGZ to NIFTI format..."
# echo "Log will be saved to: ${convert_log}"
# (
#   echo "Stage 5: Command started at: $(date)"
#   echo "Running command: mri_convert ${norm_dir}/${output_mgz} ${norm_dir}/${output_nii} --out_type nii"
#   mri_convert ${norm_dir}/${output_mgz} \
#              ${norm_dir}/${output_nii} \
#              --out_type nii \
#              --out_orientation RAS \
#              2>&1
#   echo "Stage 5: Command completed at: $(date)"
# ) | tee "${convert_log}"

# Stage 6: Final Validation
echo "=== Stage 6: Final Validation ==="
if [ -f "${norm_dir}/${output_nii}" ]; then
   echo "Stage 6: Successfully created normalized NIFTI file at: ${norm_dir}/${output_nii}"
   echo "Log files can be found at:"
   echo "  - mri_vol2vol log: ${vol2vol_log}"
   echo "  - mri_convert log: ${convert_log}"
else
   echo "Stage 6: Error: NIFTI conversion failed"
   echo "Please check the log files:"
   echo "  - mri_vol2vol log: ${vol2vol_log}"
   echo "  - mri_convert log: ${convert_log}"
   exit 1
fi

echo "=== All Stages Complete ==="
echo "Normalized MGZ file: ${norm_dir}/${output_mgz}"
echo "Normalized NIFTI file: ${norm_dir}/${output_nii}"
