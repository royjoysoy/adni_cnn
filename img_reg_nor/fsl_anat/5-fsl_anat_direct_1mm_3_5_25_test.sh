#!/bin/bash

#$ -S /bin/bash
#$ -N fsl_anat_1mm
#$ -V
#$ -t 1-3
#$ -cwd
#$ -o ./fsl_anat_logs_1mm_test/
#$ -e ./fsl_anat_logs_1mm_test/

# Create log directory if it doesn't exist
mkdir -p ./fsl_anat_logs_1mm_test/

# Load FSL environment
. $FSLDIR/etc/fslconf/fsl.sh
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH=$FSLDIR/bin:$PATH

# Print FSL information for debugging
echo "FSLDIR is set to: $FSLDIR"
echo "PATH is: $PATH"
which flirt

# Define directories
INPUT_DIR="/ibic/scratch/royseo_workingdir/fsl_anat/raw_test"
OUTPUT_DIR="/ibic/scratch/royseo_workingdir/fsl_anat/raw_test"
SUBJECT_LIST="/ibic/scratch/royseo_workingdir/fsl_anat/scripts/fsl_anat_subj_list_3_raw_test.log"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Get the current subject from the list
SUBJECT_WITH_NII=$(sed -n "${SGE_TASK_ID}p" ${SUBJECT_LIST})

# Remove .nii extension if present to get the .anat directory name
SUBJECT=${SUBJECT_WITH_NII%.nii}

echo "Processing subject: ${SUBJECT}"

# Check if input file exists
if [ ! -f "${INPUT_DIR}/${SUBJECT_WITH_NII}" ]; then
    echo "Error: Input file ${INPUT_DIR}/${SUBJECT_WITH_NII} does not exist."
    exit 1
fi

# Check if subject directory exists
if [ -d "${INPUT_DIR}/${SUBJECT}.anat" ]; then
    echo "Warning: Subject directory ${INPUT_DIR}/${SUBJECT}.anat already exists. Proceeding with caution."
fi

# Additional debugging information
echo "Input file: ${INPUT_DIR}/${SUBJECT_WITH_NII}"
echo "Output directory: ${OUTPUT_DIR}/${SUBJECT}"

# Run fsl_anat with error handling
if ! fsl_anat -i "${INPUT_DIR}/${SUBJECT_WITH_NII}" -o "${OUTPUT_DIR}/${SUBJECT}" -t MNI152_T1_1mm --nobias --highres; then
    echo "Error: fsl_anat failed for ${SUBJECT}"
    exit 1
fi

# Verify the output was created successfully
if [ -f "${OUTPUT_DIR}/${SUBJECT}.anat/T1_to_MNI_nonlin_1mm.nii.gz" ]; then
    echo "Successfully created 1mm MNI space image for ${SUBJECT}"
    
    # Additional verification of output image
    echo "Output image details:"
    fslinfo "${OUTPUT_DIR}/${SUBJECT}.anat/T1_to_MNI_nonlin_1mm.nii.gz"
else
    echo "Error: Failed to create 1mm MNI space image for ${SUBJECT}"
    exit 1
fi

echo "Completed processing ${SUBJECT}"