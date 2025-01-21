#!/bin/bash
# check-fnirt-status.sh
# Script to check the status of FNIRT processing jobs with clear distinction between
# incomplete processes and missing files

##!/bin/bash
# check-fnirt-status.sh
# Script to check the status of FNIRT processing jobs with clear distinction between
# incomplete processes and missing files

# Setup directories
BASE_DIR="/ibic/scratch/royseo_workingdir"
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_18001-28001"
SCRIPT_DIR="${BASE_DIR}/scripts"
LOG_DIR="fnirt_stdout_18001-28001"
ERROR_DIR="fnirt_stderr_18001-28001"

# Output files
INCOMPLETE_PROCESS_LOG="incomplete_process.txt"  # Jobs that started but didn't finish
MISSING_FILES_LOG="missing_files.txt"           # Expected files don't exist
ERROR_LOG="error_subjects.txt"                  # Jobs with explicit errors
NEVER_STARTED_LOG="never_started.txt"           # Jobs that never started
COMPLETE_LOG="complete_subjects.txt"            # Successfully completed jobs

# Clear previous logs
> "${INCOMPLETE_PROCESS_LOG}"
> "${MISSING_FILES_LOG}"
> "${ERROR_LOG}"
> "${NEVER_STARTED_LOG}"
> "${COMPLETE_LOG}"

# Read subject list
SUBJECT_LIST="${SCRIPT_DIR}/subj_list_ADNI1234_28001_18001-28001.log"

if [ ! -f "${SUBJECT_LIST}" ]; then
    echo "ERROR: Subject list file not found: ${SUBJECT_LIST}"
    exit 1
fi

echo "Checking FNIRT processing status..."

check_process_status() {
    local subject=$1
    local subject_base=$2
    local subject_dir=$3
    
    # Expected output files
    local warped_brain="${subject_dir}/${subject_base}_warped_brain.nii.gz"
    local warp_coef="${subject_dir}/${subject_base}_warp_coef.nii.gz"
    local warp_field="${subject_dir}/${subject_base}_warp_field.nii.gz"
    local jacobian="${subject_dir}/${subject_base}_jacobian.nii.gz"
    local inv_warp="${subject_dir}/${subject_base}_inv_warp.nii.gz"
    
    # Check for stdout/stderr logs
    local stdout_log="${LOG_DIR}/fnirt_job.task*_${subject_base}.stdout"
    local error_log="${ERROR_DIR}/fnirt_job.task*_${subject_base}.stderr"
    
    # First, check if the process ever started (look for log files)
    if ! ls ${stdout_log} 2>/dev/null | grep -q .; then
        echo "${subject}: Never started" >> "${NEVER_STARTED_LOG}"
        return
    fi
    
    # Check for explicit errors in error log
    if ls ${error_log} 2>/dev/null | grep -q .; then
        if grep -q "ERROR\|Failed\|Exception" ${error_log}; then
            echo "${subject}: Error found in logs" >> "${ERROR_LOG}"
            return
        fi
    fi
    
    # Check if process started but didn't complete
    if ls ${stdout_log} 2>/dev/null | grep -q .; then
        if ! grep -q "FNIRT registration complete" ${stdout_log}; then
            # Process started but didn't reach completion message
            echo "${subject}: Process interrupted" >> "${INCOMPLETE_PROCESS_LOG}"
            return
        fi
    fi
    
    # Check for existence and size of all expected output files
    local missing_files=()
    local has_partial_files=false
    
    for file in "${warped_brain}" "${warp_coef}" "${warp_field}" "${jacobian}" "${inv_warp}"; do
        if [ ! -f "${file}" ]; then
            missing_files+=("$(basename "${file}")")
        elif [ ! -s "${file}" ]; then
            # File exists but is empty
            has_partial_files=true
            missing_files+=("$(basename "${file}") (empty)")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        if [ "${has_partial_files}" = true ]; then
            echo "${subject}: Partial files present (${missing_files[*]})" >> "${INCOMPLETE_PROCESS_LOG}"
        else
            echo "${subject}: Missing files (${missing_files[*]})" >> "${MISSING_FILES_LOG}"
        fi
        return
    fi
    
    # If we get here, everything is complete
    echo "${subject}: Complete" >> "${COMPLETE_LOG}"
}

while read subject; do
    subject_base=$(echo "${subject}" | sed 's/\.nii$//')
    subject_dir="${INPUT_BASE}/${subject_base}/mni152_1mm"
    
    # Check if directory exists
    if [ ! -d "${subject_dir}" ]; then
        echo "${subject}: Directory missing" >> "${NEVER_STARTED_LOG}"
        continue
    fi
    
    check_process_status "${subject}" "${subject_base}" "${subject_dir}"
    
done < "${SUBJECT_LIST}"

# Summary
echo "Processing Status Summary:"
echo "========================="
echo "Complete: $(wc -l < ${COMPLETE_LOG})"
echo "Never Started: $(wc -l < ${NEVER_STARTED_LOG})"
echo "Incomplete (interrupted): $(wc -l < ${INCOMPLETE_PROCESS_LOG})"
echo "Missing Files: $(wc -l < ${MISSING_FILES_LOG})"
echo "Errors: $(wc -l < ${ERROR_LOG})"
echo

echo "Detailed logs have been written to:"
echo "- ${COMPLETE_LOG} (successfully completed jobs)"
echo "- ${NEVER_STARTED_LOG} (jobs that never started)"
echo "- ${INCOMPLETE_PROCESS_LOG} (jobs that started but didn't finish)"
echo "- ${MISSING_FILES_LOG} (jobs missing expected output files)"
echo "- ${ERROR_LOG} (jobs with explicit errors)"

# Create resubmission scripts for different categories
echo "#!/bin/bash" > resubmit_never_started.sh
echo "#!/bin/bash" > resubmit_incomplete.sh
echo "#!/bin/bash" > resubmit_errors.sh

# Add header comments
for script in resubmit_never_started.sh resubmit_incomplete.sh resubmit_errors.sh; do
    echo "# Auto-generated script to resubmit FNIRT jobs" >> "${script}"
    echo "# Generated on: $(date)" >> "${script}"
done

# Populate resubmission scripts
while read line; do
    subject_id=$(echo ${line} | cut -d: -f1)
    echo "qsub -N fnirt_retry_${subject_id} 3-4-fnirt-nonlinear-HCP-param-batch-processing-18001-28001.sh" >> resubmit_never_started.sh
done < "${NEVER_STARTED_LOG}"

while read line; do
    subject_id=$(echo ${line} | cut -d: -f1)
    echo "qsub -N fnirt_retry_${subject_id} 3-4-fnirt-nonlinear-HCP-param-batch-processing-18001-28001.sh" >> resubmit_incomplete.sh
done < "${INCOMPLETE_PROCESS_LOG}"

while read line; do
    subject_id=$(echo ${line} | cut -d: -f1)
    echo "qsub -N fnirt_retry_${subject_id} 3-4-fnirt-nonlinear-HCP-param-batch-processing-18001-28001.sh" >> resubmit_errors.sh
done < "${ERROR_LOG}"

chmod +x resubmit_never_started.sh resubmit_incomplete.sh resubmit_errors.sh

echo
echo "Resubmission scripts have been created:"
echo "1. resubmit_never_started.sh (for jobs that never started)"
echo "2. resubmit_incomplete.sh (for interrupted jobs)"
echo "3. resubmit_errors.sh (for jobs with errors)"
echo
echo "Review these scripts before running them."