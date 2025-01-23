#!/bin/bash
# 1/14/2025 Roy Seo
# Purpose: Combined FLIRT-FNIRT registration script with single interpolation
# Previous workflow issue:
# - Used FLIRT output image as input for FNIRT
# - This caused normalization to the same template twice   
# - Results in cumulative interpolation errors
# Solution:
# - Generate transformation matrix from FLIRT
# - Use transformation matrix as input for FNIRT
# - Avoids double interpolation and runs one reslcing    
# Reference: Email conversation from 1/13/2025 seoroy15@gmail.com  
# Usage: qsub ./3-5-flirt-fnirt-fix-1-14-25.sh  

# Create logs directory if it doesn't exist
mkdir -p flirt_fnirt_logs

# Count number of subjects in the list
num_subjects=$(wc -l < subj_list_flirt_fnirt_1_22_2025.log)

# Submit job with correct number of tasks
qsub -t 1-${num_subjects} ./3-5-flirt-fnirt-fix-1-22-25.sh







