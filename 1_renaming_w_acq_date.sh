#!/bin/bash
# Roy Seo
# 12/02/2024 in Korea Time
# to rename (to do 'cp') the ADNI image files with data acq date from /projects4/royseo_workingdir/raw to /projects4/royseo_workingdir/raw_w_acq_date
# input file1: /projects4/royseo_workingdir/scripts/1_inputfile1_11-14-24-ADNI_1_2_3_4.csv
# input file2: /projects4/royseo_workingdir/raw
# output file: /projects4/royseo_workingdir/raw_w_acq_date 
# 1/20/2025: 이 스크립트가 첫번째 (input file1에 헤더 바로 밑에 있는 'I882756'는 처리(renaming하고 raw_w_acq_date에 저장)를 해주지 않았다.
# 1/20/2025:  결국 이 'I882756' 이미지는 내가 직접 이름 rename하고 raw_w_acq_date에 저장

:'
1. On the 10th column (column "J" if you open this input file 1 with Excel) 
from "input file1" image acquisition dates (header: "Acq Date") are found.
2. using "I" number in the each image file name in input file 2, 
this script looks up the acquistion date in the 10th column (header: "Acq Date")
by matching the 1st column in input file 1 (called "Image Data ID" ) 
using the acqusition date found in the Acq Date column it made a name like
"094_S_1330_2009-03-09-S22439_I436815.nii"

notes: inputfile1 includes information about 28002 and it was donwlonaded in 11/14/24 from ADNI image colloection called ADNI_1_2_3_4
it does NOT include the behavioral datasets 
'
# Directory paths
source_dir="/projects4/royseo_workingdir/raw"
target_dir="/projects4/royseo_workingdir/raw_w_acq_date"
csv_file="/projects4/royseo_workingdir/scripts/1_inputfile1_11-14-24-ADNI_1_2_3_4.csv"

# Add debug flag
debug=1  # Set to 1 to enable debugging output

# Debug function - define this BEFORE any debug_print calls
debug_print() {
    if [ $debug -eq 1 ]; then
        echo "DEBUG: $1"
    fi
}

# Now we can use debug_print
debug_print "=== Directory Contents Check ==="
debug_print "Checking source directory: $source_dir"
debug_print "Files found:"
ls -l "$source_dir"/ADNI_*.nii 2>/dev/null | head -n 3
debug_print "==========================="

# Check if source directory exists
if [ ! -d "$source_dir" ]; then
    echo "Error: Source directory $source_dir does not exist."
    exit 1
fi

# Check if directory exists and is not empty
if [ -d "$target_dir" ] && [ "$(ls -A $target_dir)" ]; then
    echo "Warning: Directory $target_dir already exists and contains files."
    read -p "Do you want to continue? Files may be overwritten [y/N]: " response
    if [[ ! $response =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 1
    fi
fi

# Create directory if it doesn't exist
mkdir -p "$target_dir"
chmod 777 "$target_dir"

# Function to convert date from M/D/YY to YYYY-MM-DD
convert_date() {
    local input_date=$1
    echo "$input_date" | awk -F'/' '
    {
        month = $1
        day = $2
        year = $3
        if (length(month) == 1) month = "0" month
        if (length(day) == 1) day = "0" day
        printf("%s-%s-%s", year, month, day)
    }'
}

# Create an associative array to store ID-date mappings
declare -A dates

# Debug: Print CSV file content
debug_print "Reading CSV file..."
head -n 5 "$csv_file"

# Read and parse CSV file
while IFS=, read -r id rest; do
    # Remove quotes if present
    id=${id//\"/}
    # Get the date field
    date=$(echo "$rest" | awk -F, '{print $9}' | tr -d '"')
    
    debug_print "Processing ID: $id, Date: $date"
    
    if [ ! -z "$id" ] && [ ! -z "$date" ]; then
        # Convert date format
        converted_date=$(convert_date "$date")
        debug_print "Converted date for $id: $converted_date"
        dates[$id]=$converted_date
    fi
done < "$csv_file"
# File processing section
debug_print "Starting file processing..."
debug_print "Looking for files in: $source_dir/ADNI_*.nii"

for file in "$source_dir"/ADNI_*.nii; do
    [ -e "$file" ] || { debug_print "No files found matching pattern"; continue; }
    
    filename=$(basename "$file")
    debug_print "Original filename: $filename"
    
    # Extract just the I number from the ID (I#####)
    id=$(echo "$filename" | grep -o 'I[0-9]*' | tail -n 1)
    debug_print "Extracted ID: $id"
    
    # Extract subject ID (###_S_####)
    subject_id=$(echo "$filename" | grep -o '[0-9]\{3\}_S_[0-9]\{4\}')
    debug_print "Extracted subject ID: $subject_id"
    
    # Extract S number (S#####)
    s_number=$(echo "$filename" | grep -o 'S[0-9]\{5\}')
    debug_print "Extracted S number: $s_number"
    
    # Get corresponding date
    date="${dates[$id]}"
    debug_print "Found date for ID $id: $date"
    
    if [ ! -z "$date" ]; then
        new_name="${subject_id}_${date}_${s_number}_${id}.nii"
        debug_print "New name will be: $new_name"
        
        debug_print "Attempting to copy: $file to $target_dir/$new_name"
        cp "$file" "$target_dir/$new_name"
        if [ $? -eq 0 ]; then
            echo "Successfully copied $filename to $target_dir/$new_name"
        else
            echo "Error copying $filename"
        fi
    else
        echo "Warning: No acquisition date found for ID: $id in file: $filename"
    fi
done

# Debug: Print final count
debug_print "Number of files in target directory: $(ls "$target_dir" | wc -l)"

echo "Processing complete!"