 
# FSL_ANAT Batch Processing Script
# Script for running FSL's anatomical processing pipeline (fsl_anat) on multiple subjects using Sun Grid Engine.
# Monitor: 
        # 1. qstat -u royseo
        # 2. cat fsl_anat_logs/fsl_anat_normalization.o*

# Requirements
        # 1. FSL 6.0 installed at /usr/local/fsl/6.0/
        # 2. Sun Grid Engine (SGE) environment
        # 3. Input files in NIFTI format (.nii)
        # 4. Subject list file with one filename per line : fsl_anat_subj_list_2409_raw.log
# Setup
        # 1. Create directory structure:
        #       2. scripts/fsl_anat_logs/
        #       3. raw/ (input NIFTI files)
        #       4. processed/ (output .anat folders) 
        #           - 01-23-2025: 'processed' folder was empty after the job is completed
        #           - 01-23-2025: had to use /fsl_anat/scripts/2-1-copy-lin-nonlin-01-23-2025.py


#$ -S /bin/bash
#$ -N fsl_anat_normalization
#$ -V
#$ -t 1-2409
#$ -cwd
#$ -o ./fsl_anat_logs/
#$ -e ./fsl_anat_logs/

source /usr/local/fsl/6.0/etc/fslconf/fsl.sh
export FSLDIR=/usr/local/fsl/6.0
export PATH=${FSLDIR}/bin:${PATH}

# Set file paths
FILE_LIST="/ibic/scratch/royseo_workingdir/fsl_anat/scripts/fsl_anat_subj_list_2409_raw.log"
INPUT_DIR="/ibic/scratch/royseo_workingdir/fsl_anat/raw"
OUTPUT_DIR="/ibic/scratch/royseo_workingdir/fsl_anat/processed"

# Check if file list exists
if [ ! -f "$FILE_LIST" ]; then
    echo "$(date): Error - File list $FILE_LIST not found" >&2
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Extract the filename for the current task ID
FILE_NAME=$(sed -n "${SGE_TASK_ID}p" $FILE_LIST)

# Exit if filename is empty
if [ -z "$FILE_NAME" ]; then
    echo "$(date): Error - No filename found for task ID ${SGE_TASK_ID}" >&2
    exit 1
fi

# Run fsl_anat for the specific file
if [ -f "${INPUT_DIR}/${FILE_NAME}" ] && [ -r "${INPUT_DIR}/${FILE_NAME}" ]; then
    echo "$(date): Processing ${FILE_NAME}"
    fsl_anat -i "${INPUT_DIR}/${FILE_NAME}" --nobias
    mv "${FILE_NAME%.*}.anat" "${OUTPUT_DIR}/"
else
    echo "$(date): Error - File ${INPUT_DIR}/${FILE_NAME} not found or not readable" >&2
    exit 1
fi