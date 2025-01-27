 
# 01-24-25 Roy Seo Korea 1-28002모두 돌리기 위해 (just_in_case 필요할까봐)

# FSL_ANAT Batch Processing Script
# Script for running FSL's anatomical processing pipeline (fsl_anat) on multiple subjects using Sun Grid Engine.
# Monitor: 
        # 1. qstat -u royseo
        # 2. cat fsl_anat_logs/fsl_anat_normalization.o*

# Requirements
        # 1. FSL 6.0 installed at /usr/local/fsl/6.0/
        # 2. Sun Grid Engine (SGE) environment
        # 3. Input files in NIFTI format (.nii)
        # 4. Subject list file with one filename per line : subj_list_28002_raw.log
# Setup
        # 1. Create directory structure:
        #       2. scripts/fsl_anat_logs/
        #       3. /raw_w_acq_date (input NIFTI files)
        #       4. processed/ (output .anat folders) 
        #           - 01-23-2025: 'processed' folder was empty after the job is completed
        #           - 01-23-2025: had to use /fsl_anat/scripts/2-1-copy-lin-nonlin-01-23-2025.py
        # 5. qsub돌리기 전에 로그 디렉토리 생성: mkdir -p ./fsl_anat_logs_1-28002 

        # 돌리는 방법: 다른 파일을 돌릴 필요는 없고 이렇게 터미널에만 치면 끝 "qsub 2-fsl_anat_qsub_1-28002-01_24_25.sh"


#$ -S /bin/bash
#$ -N fsl_anat_normalization
#$ -V
#$ -t 1-28002
#$ -cwd
#$ -o ./fsl_anat_logs_1-28002/
#$ -e ./fsl_anat_logs_1-28002/

source /usr/local/fsl/6.0/etc/fslconf/fsl.sh
export FSLDIR=/usr/local/fsl/6.0
export PATH=${FSLDIR}/bin:${PATH}

# Set file paths
FILE_LIST="/ibic/scratch/royseo_workingdir/fsl_anat/scripts/subj_list_28002_raw.log"
INPUT_DIR="/ibic/scratch/royseo_workingdir/raw_w_acq_date"
OUTPUT_DIR="/ibic/scratch/royseo_workingdir/fsl_anat/processed_28002"

# Check if file list exists
if [ ! -f "$FILE_LIST" ]; then
    echo "$(date): Error - File list $FILE_LIST not found" >&2
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
# Check if output directory is writable
if [ ! -w "$OUTPUT_DIR" ]; then
    echo "$(date): Error - Output directory $OUTPUT_DIR is not writable" >&2
    exit 1
fi

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
    
    # Change to output directory before running fsl_anat
    cd "${OUTPUT_DIR}"
    
    # Run fsl_anat
    fsl_anat -i "${INPUT_DIR}/${FILE_NAME}" --nobias -o "${OUTPUT_DIR}/${FILE_NAME%.*}"
    
    # Check if processing was successful
    if [ $? -ne 0 ]; then
        echo "$(date): Error - fsl_anat processing failed for ${FILE_NAME}" >&2
        exit 1
    fi
else
    echo "$(date): Error - File ${INPUT_DIR}/${FILE_NAME} not found or not readable" >&2
    exit 1
fi