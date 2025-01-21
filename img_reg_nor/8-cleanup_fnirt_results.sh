#!/bin/bash
# fnirt process stopped in the middle of qsub this script performs rm the fnirt results

#!/bin/bash

# Basic directory setup
BASE_DIR="/ibic/scratch/royseo_workingdir"
INPUT_BASE="${BASE_DIR}/normalized2mni152_1mm_18001-28002"
SCRIPT_DIR="${BASE_DIR}/scripts"
LOG_FILE="cleanup_$(date +%Y%m%d_%H%M%S).log"
FAILED_SUBJECTS_FILE="failed_subjects_$(date +%Y%m%d_%H%M%S).txt"
PARTIAL_SUBJECTS_FILE="partial_subjects_$(date +%Y%m%d_%H%M%S).txt"

# Log function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Result file patterns
declare -a patterns=(
    "_warped_brain.nii.gz"
    "_warp_coef.nii.gz"
    "_warp_field.nii.gz"
    "_jacobian.nii.gz"
    "_inv_warp.nii.gz"
)
TOTAL_EXPECTED_FILES=${#patterns[@]}

# Read subject list
SUBJECTS_FILE="${SCRIPT_DIR}/subj_list_ADNI1234_28002_18001-28002_1_20_2025.log"
if [ ! -f "$SUBJECTS_FILE" ]; then
    log_message "ERROR: Subjects list file not found: $SUBJECTS_FILE"
    exit 1
fi

# Initialize counters
total_processed=0
total_failed=0
total_partial=0

log_message "Starting cleanup process"
log_message "Total subjects to process: $(wc -l < "$SUBJECTS_FILE")"

# Process each subject
while IFS= read -r subject; do
    subject_base=$(echo "${subject}" | sed 's/\.nii$//')
    subject_dir="${INPUT_BASE}/${subject_base}/mni152_1mm"
    files_found=0
    files_deleted=0

    # Check if directory exists
    if [ ! -d "$subject_dir" ]; then
        log_message "ERROR: Directory not found: $subject_dir"
        echo "$subject_base" >> "$FAILED_SUBJECTS_FILE"
        ((total_failed++))
        continue
    fi  # Changed from } to fi

    # Check and delete each pattern
    missing_files=""
    for pattern in "${patterns[@]}"; do
        file="${subject_dir}/${subject_base}${pattern}"
        if [ -f "$file" ]; then
            ((files_found++))
            if rm -f "$file"; then
                ((files_deleted++))
            fi
        else
            missing_files="${missing_files}${pattern}, "
        fi
    done

    # Handle different cases
    if [ $files_found -eq 0 ]; then
        log_message "No FNIRT results found: $subject_base"
        echo "$subject_base" >> "$FAILED_SUBJECTS_FILE"
        ((total_failed++))
    elif [ $files_found -lt $TOTAL_EXPECTED_FILES ]; then
        # Remove trailing comma and space from missing_files
        missing_files=${missing_files%, }
        log_message "Partial FNIRT results ($files_found/$TOTAL_EXPECTED_FILES) for: $subject_base"
        log_message "  Missing files: $missing_files"
        echo "$subject_base (Missing: $missing_files)" >> "$PARTIAL_SUBJECTS_FILE"
        ((total_partial++))
    else
        log_message "Complete cleanup ($files_deleted/$TOTAL_EXPECTED_FILES): $subject_base"
    fi

    ((total_processed++))
    
    # Show progress every 100 subjects
    if [ $((total_processed % 100)) -eq 0 ]; then
        log_message "Progress: $total_processed subjects processed"
    fi
done < "$SUBJECTS_FILE"

# Print final summary
log_message "Cleanup process completed"
log_message "Total subjects processed: $total_processed"
log_message "Total subjects with no results: $total_failed"
log_message "Total subjects with partial results: $total_partial"
log_message "Total subjects with complete results: $((total_processed - total_failed - total_partial))"

# Print list of failed subjects
if [ $total_failed -gt 0 ]; then
    log_message "List of subjects with no FNIRT results:"
    cat "$FAILED_SUBJECTS_FILE" | while read -r failed_subject; do
        log_message "  - $failed_subject"
    done
fi

# Print list of partial subjects
if [ $total_partial -gt 0 ]; then
    log_message "List of subjects with partial FNIRT results:"
    cat "$PARTIAL_SUBJECTS_FILE" | while read -r partial_subject; do
        log_message "  - $partial_subject"
    done
fi