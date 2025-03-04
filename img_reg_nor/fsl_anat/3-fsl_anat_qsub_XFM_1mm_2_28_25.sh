#!/bin/bash
#This script performs batch processing of T1-weighted MRI images 
#to convert them from 2mm MNI space to 1mm MNI space using FSL's FLIRT tool. 
#It is designed to run on a grid computing environment using the Sun Grid Engine (SGE) job scheduler.

#$ -S /bin/bash
#$ -N fsl_anat_1mm
#$ -V
#$ -t 1-2409
#$ -cwd
#$ -o ./fsl_anat_logs_1mm/
#$ -e ./fsl_anat_logs_1mm/

# Load FSL environment
. $FSLDIR/etc/fslconf/fsl.sh
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH=$FSLDIR/bin:$PATH

# Print FSL information for debugging
echo "FSLDIR is set to: $FSLDIR"
echo "PATH is: $PATH"
which flirt

# Define directories
INPUT_DIR="/ibic/scratch/royseo_workingdir/fsl_anat/raw"
OUTPUT_DIR="/ibic/scratch/royseo_workingdir/fsl_anat/raw"
SUBJECT_LIST="/ibic/scratch/royseo_workingdir/fsl_anat/scripts/fsl_anat_subj_list_2409_raw.log"

# Get the current subject from the list
SUBJECT_WITH_NII=$(sed -n "${SGE_TASK_ID}p" ${SUBJECT_LIST})

# Remove .nii extension if present to get the .anat directory name
SUBJECT=${SUBJECT_WITH_NII%.nii}

echo "Processing subject: ${SUBJECT}"

# Check if subject directory exists
if [ ! -d "${INPUT_DIR}/${SUBJECT}.anat" ]; then
    echo "Error: Subject directory ${INPUT_DIR}/${SUBJECT}.anat does not exist."
    exit 1
fi

# Run flirt to convert 2mm to 1mm
$FSLDIR/bin/flirt -in ${INPUT_DIR}/${SUBJECT}.anat/T1_to_MNI_nonlin.nii.gz \
      -ref $FSLDIR/data/standard/MNI152_T1_1mm.nii.gz \
      -out ${OUTPUT_DIR}/${SUBJECT}.anat/T1_to_MNI_nonlin_1mm.nii.gz \
      -applyisoxfm 1.0

# Verify the output was created successfully
if [ -f "${OUTPUT_DIR}/${SUBJECT}.anat/T1_to_MNI_nonlin_1mm.nii.gz" ]; then
    echo "Successfully created 1mm MNI space image for ${SUBJECT}"
else
    echo "Error: Failed to create 1mm MNI space image for ${SUBJECT}"
    exit 1
fi

echo "Completed processing ${SUBJECT}"