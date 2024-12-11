#!/bin/bash

# Usage function to display help
usage() {
    echo "Usage: $0 <subject_list_file> <output_directory>"
    echo "Example: $0 scripts/subj_list_ADNI1234_28001_1-10.log normalized2mni152_1mm_1-10"
    echo
    echo "Parameters:"
    echo "  subject_list_file  - File containing list of subjects (one per line)"
    echo "  output_directory   - Base directory where normalized data is stored"
    exit 1
}

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    usage
fi

subject_list="$1"
output_dir="$2"

# Check if files/directories exist
if [ ! -f "$subject_list" ]; then
    echo "Error: Subject list file '$subject_list' not found"
    exit 1
fi

if [ ! -d "$output_dir" ]; then
    echo "Error: Output directory '$output_dir' not found"
    exit 1
fi

# Initialize counters
total_count=0
incomplete_count=0
complete_count=0

# Process each subject
echo "Checking for incomplete subjects..."
while read -r subject; do
    # Increment total count
    ((total_count++))
    
    # Remove .nii extension if present
    subject_base=${subject%%.nii*}
    
    # Check if the expected file exists
    if [ ! -f "$output_dir/$subject_base/mri/brain.mgz" ]; then
        echo "$subject_base not complete"
        ((incomplete_count++))
    else
        ((complete_count++))
    fi
done < "$subject_list"

echo "----------------------------------------"
echo "Summary:"
echo "Total subjects: $total_count"
echo "Complete: $complete_count"
echo "Incomplete: $incomplete_count"
echo "Completion rate: $(( (complete_count * 100) / total_count ))%"
echo "----------------------------------------"
echo "Check complete!"