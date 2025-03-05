#!/bin/bash

#$ -S /bin/bash
#$ -N fsl_anat_1mm
#$ -V
#$ -t 1-2049
#$ -cwd
#$ -o ./fsl_anat_logs_1mm_direct/
#$ -e ./fsl_anat_logs_1mm_direct/

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
OUTPUT_DIR="/ibic/scratch/royseo_workingdir/fsl_anat/processed_direct_1mm"
SUBJECT_LIST="/ibic/scratch/royseo_workingdir/fsl_anat/scripts/fsl_anat_subj_list_2409_raw.log"

# Get the current subject from the list
SUBJECT_WITH_NII=$(sed -n "${SGE_TASK_ID}p" ${SUBJECT_LIST})

# Remove .nii extension if present to get the .anat directory name
SUBJECT=${SUBJECT_WITH_NII%.nii}

echo "Processing subject: ${SUBJECT}"

# Check if the input file exists
if [ ! -f "${INPUT_DIR}/${SUBJECT_WITH_NII}" ]; then
    echo "Error: Input file ${INPUT_DIR}/${SUBJECT_WITH_NII} does not exist."
    exit 1
fi

# Run fsl_anat for the specific file
echo "$(date): Running fsl_anat on ${SUBJECT_WITH_NII}"
fsl_anat -i "${INPUT_DIR}/${SUBJECT_WITH_NII}" -o "${OUTPUT_DIR}/${SUBJECT}" -t MNI152_T1_1mm --nobias

# Verify the output was created successfully
if [ -f "${OUTPUT_DIR}/${SUBJECT}.anat/T1_to_MNI_nonlin_1mm.nii.gz" ]; then
    echo "Successfully created 1mm MNI space image for ${SUBJECT}"
else
    echo "Error: Failed to create 1mm MNI space image for ${SUBJECT}"
    exit 1
fi

echo "Completed processing ${SUBJECT}"