#!/bin/bash

# DICOM to NIfTI Conversion Script
# This script converts DICOM files to NIfTI format using dcm2niix
# 
# Description:
# This script navigates through a nested directory structure containing DICOM files
# from the ADNI dataset and converts them to NIfTI format using dcm2niix.
# It preserves the original directory structure in the output folder and maintains
# subject/sequence information in the filenames. The script includes comprehensive
# error handling and detailed logging of the conversion process.
#
# Directory structure:
# /ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/ADNI1234_1/ADNI/[subject_id]/[sequence]/[timestamp]/[series]/
# Example: /ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/ADNI1234_1/ADNI/941_S_7106/Sagittal_3D_FLAIR/2022-09-09_09_55_29.0/I1619410/
#
# Output:
# Converted NIfTI files are saved to /ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_adni1234_5-10-2025-nifti
# with appropriate subject directories and meaningful filenames
# input과 output directories만 바꿔서 쓰면 됨



# Define directories
INPUT_DIR="/ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/ADNI1234_2/ADNI"
OUTPUT_DIR="/ibic/scratch/royseo_workingdir/t2_adni1234_5-10-2025/t2_adni1234_5-10-2025-nifti_zip2"
LOG_FILE="${OUTPUT_DIR}/conversion_log_$(date +%Y%m%d_%H%M%S).log"

# Function to log messages
log_message() {
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[${timestamp}] $1" | tee -a "$LOG_FILE"
}

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    log_message "Created output directory: $OUTPUT_DIR"
fi

# Check if dcm2niix is installed
if ! command -v dcm2niix &> /dev/null; then
    log_message "ERROR: dcm2niix is not installed or not in PATH"
    exit 1
fi

# Initialize counters
total_subjects=0
total_series=0
successful_conversions=0
failed_conversions=0

log_message "Starting DICOM to NIfTI conversion process"
log_message "Input directory: $INPUT_DIR"
log_message "Output directory: $OUTPUT_DIR"

# Loop through all subject IDs in ADNI folder
for subject_dir in "$INPUT_DIR"/*; do
    if [ -d "$subject_dir" ]; then
        subject_id=$(basename "$subject_dir")
        log_message "Processing subject: $subject_id"
        
        # Create subject directory in output
        subject_output_dir="${OUTPUT_DIR}/${subject_id}"
        mkdir -p "$subject_output_dir"
        
        ((total_subjects++))
        
        # Loop through all sequence types
        for sequence_dir in "$subject_dir"/*/; do
            if [ -d "$sequence_dir" ]; then
                sequence_name=$(basename "$sequence_dir")
                log_message "  Processing sequence: $sequence_name"
                
                # Create sequence directory in output
                sequence_output_dir="${subject_output_dir}/${sequence_name}"
                mkdir -p "$sequence_output_dir"
                
                # Loop through all timestamp directories (visit dates)
                for timestamp_dir in "$sequence_dir"/*; do
                    if [ -d "$timestamp_dir" ]; then
                        timestamp=$(basename "$timestamp_dir")
                        log_message "    Processing timestamp: $timestamp"
                        
                        # Loop through all I-series directories
                        for series_dir in "$timestamp_dir"/I*; do
                            if [ -d "$series_dir" ]; then
                                series_name=$(basename "$series_dir")
                                log_message "      Processing series: $series_name"
                                
                                ((total_series++))
                                
                                # Create output filename with meaningful information
                                output_prefix="${subject_id}_${sequence_name}_${timestamp}_${series_name}"
                                
                                # Convert DICOM to NIfTI
                                log_message "      Running dcm2niix on: $series_dir"
                                conversion_output=$(dcm2niix -z y -f "$output_prefix" -o "$sequence_output_dir" "$series_dir" 2>&1)
                                
                                # Check conversion result
                                if [ $? -eq 0 ]; then
                                    log_message "      Conversion successful"
                                    log_message "      $conversion_output"
                                    ((successful_conversions++))
                                else
                                    log_message "      ERROR: Conversion failed"
                                    log_message "      $conversion_output"
                                    ((failed_conversions++))
                                fi
                            fi
                        done
                    fi
                done
            fi
        done
    fi
done

# Print summary
log_message "====== Conversion Summary ======"
log_message "Total subjects processed: $total_subjects"
log_message "Total series processed: $total_series"
log_message "Successful conversions: $successful_conversions"
log_message "Failed conversions: $failed_conversions"
log_message "=============================="

if [ $failed_conversions -gt 0 ]; then
    log_message "WARNING: Some conversions failed. Check the log for details."
    exit 1
else
    log_message "All conversions completed successfully."
    exit 0
fi