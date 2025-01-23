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