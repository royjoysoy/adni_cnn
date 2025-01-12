#1-12-2025 Roy Seo

#!/bin/bash

# Summary:
# GPU가 있는 컴퓨터로 파일을 옮기려고 다음과 같이 4개의 폴더를 만들없다
#     1_linear_1-6000: 파일이 5987개 
#     2_linear_6001-18000: 파일이 11985개
#     1_nonlinear_1-6000: 파일이 5989개
#     2_nonlinear_6001-18000: 파일이 11985개
# 그래서 어떤 파일이 missing되었는지 찾고자 함

#############################################################################
# Script 1: Check for missing *_brain_mni152_1mm.nii.gz files
#############################################################################
# Summary:
# This script compares subject IDs in log files against actual files in specified directories
# to find missing files. 
#
# Input format:
# - Log files contain subject IDs in format: 002_S_0295_2006-04-18_S13407_I45112.nii
# - Actual files have format: 002_S_0295_2006-04-18_S13407_I45112_brain_mni152_1mm.nii.gz
#
# Process:
# 1. Reads subject IDs from each log file
# 2. Converts subject IDs to expected file names (adds "_brain_mni152_1mm.nii.gz")
# 3. Checks if these files exist in the corresponding directory
# 4. Reports any missing files

# Function to find missing files for a given log file and directory
check_missing_files_mni152() {
    local log_file=$1
    local target_dir=$2
    
    echo -e "\nChecking ${log_file} against directory ${target_dir}"
    echo "------------------------------------------------"
    
    # Create temporary files to store processed names
    tmp_subjects=$(mktemp)
    
    # Read log file and convert subject names to expected file names
    while IFS= read -r line; do
        if [ ! -z "$line" ]; then
            echo "${line//.nii/_brain_mni152_1mm.nii.gz}" >> "$tmp_subjects"
        fi
    done < "$log_file"
    
    # Check each expected file
    while IFS= read -r subject; do
        if [ ! -f "${target_dir}/${subject}" ]; then
            echo "Missing: ${subject}"
        fi
    done < "$tmp_subjects"
    
    # Clean up temporary file
    rm "$tmp_subjects"
}

echo "Checking for missing *_brain_mni152_1mm.nii.gz files..."
echo "======================================================"

# Process each log file and directory pair for first check
check_missing_files_mni152 "subj_list_ADNI1234_28001_1-10.log" "../1_linear_1-6000"
check_missing_files_mni152 "subj_list_ADNI1234_28001_11-6000_fnirt.log" "../1_linear_1-6000"
check_missing_files_mni152 "subj_list_ADNI1234_28001_6001-12000.log" "../2_linear_6001-18000"
check_missing_files_mni152 "subj_list_ADNI1234_28001_12001-18000.log" "../2_linear_6001-18000"

#############################################################################
# Script 2: Check for missing *_warped_brain.nii.gz files
#############################################################################
# Summary:
# This script compares subject IDs in log files against actual files in specified directories
# to find missing warped brain files. 
#
# Input format:
# - Log files contain subject IDs in format: 002_S_0295_2006-04-18_S13407_I45112.nii
# - Actual files have format: 002_S_0295_2006-04-18_S13407_I45112_warped_brain.nii.gz
#
# Process:
# 1. Reads subject IDs from each log file
# 2. Converts subject IDs to expected file names (adds "_warped_brain.nii.gz")
# 3. Checks if these files exist in the corresponding directory
# 4. Reports any missing files

# Function to find missing warped brain files
check_missing_files_warped() {
    local log_file=$1
    local target_dir=$2
    
    echo -e "\nChecking ${log_file} against directory ${target_dir}"
    echo "------------------------------------------------"
    
    # Create temporary files to store processed names
    tmp_subjects=$(mktemp)
    
    # Read log file and convert subject names to expected file names
    while IFS= read -r line; do
        if [ ! -z "$line" ]; then
            echo "${line//.nii/_warped_brain.nii.gz}" >> "$tmp_subjects"
        fi
    done < "$log_file"
    
    # Check each expected file
    while IFS= read -r subject; do
        if [ ! -f "${target_dir}/${subject}" ]; then
            echo "Missing: ${subject}"
        fi
    done < "$tmp_subjects"
    
    # Clean up temporary file
    rm "$tmp_subjects"
}

echo -e "\n\nChecking for missing *_warped_brain.nii.gz files..."
echo "================================================="

# Process each log file and directory pair for second check
check_missing_files_warped "subj_list_ADNI1234_28001_1-10.log" "../1_nonlinear_1-6000"
check_missing_files_warped "subj_list_ADNI1234_28001_11-6000_fnirt.log" "../1_nonlinear_1-6000" 
check_missing_files_warped "subj_list_ADNI1234_28001_6001-12000.log" "../2_nonlinear_6001-18000"
check_missing_files_warped "subj_list_ADNI1234_28001_12001-18000.log" "../2_nonlinear_6001-18000"

[이전 스크립트 내용은 동일하게 유지...]

#############################################################################
# Script 3: Print Summary of Missing Files
#############################################################################
# Summary:
# This script counts and prints the total number of missing files for each log file
# and directory combination from both previous checks.

echo -e "\n\nSummary of Missing Files"
echo "========================"

# Function to count missing files
count_missing_files() {
    local log_file=$1
    local target_dir=$2
    local suffix=$3
    local count=0
    
    # Create temporary files
    tmp_subjects=$(mktemp)
    
    # Process log file
    while IFS= read -r line; do
        if [ ! -z "$line" ]; then
            echo "${line//.nii/${suffix}}" >> "$tmp_subjects"
        fi
    done < "$log_file"
    
    # Count missing files
    while IFS= read -r subject; do
        if [ ! -f "${target_dir}/${subject}" ]; then
            count=$((count + 1))
        fi
    done < "$tmp_subjects"
    
    # Clean up
    rm "$tmp_subjects"
    
    # Return count
    echo $count
}

echo "Missing *_brain_mni152_1mm.nii.gz files:"
echo "---------------------------------------"
for pair in \
    "subj_list_ADNI1234_28001_1-10.log;../1_linear_1-6000" \
    "subj_list_ADNI1234_28001_11-6000_fnirt.log;../1_linear_1-6000" \
    "subj_list_ADNI1234_28001_6001-12000.log;../2_linear_6001-18000" \
    "subj_list_ADNI1234_28001_12001-18000.log;../2_linear_6001-18000"
do
    IFS=";" read -r log_file target_dir <<< "$pair"
    count=$(count_missing_files "$log_file" "$target_dir" "_brain_mni152_1mm.nii.gz")
    echo "${log_file}; ${target_dir}: ${count} files missing"
done

echo -e "\nMissing *_warped_brain.nii.gz files:"
echo "------------------------------------"
for pair in \
    "subj_list_ADNI1234_28001_1-10.log;../1_nonlinear_1-6000" \
    "subj_list_ADNI1234_28001_11-6000_fnirt.log;../1_nonlinear_1-6000" \
    "subj_list_ADNI1234_28001_6001-12000.log;../2_nonlinear_6001-18000" \
    "subj_list_ADNI1234_28001_12001-18000.log;../2_nonlinear_6001-18000"
do
    IFS=";" read -r log_file target_dir <<< "$pair"
    count=$(count_missing_files "$log_file" "$target_dir" "_warped_brain.nii.gz")
    echo "${log_file}; ${target_dir}: ${count} files missing"
done