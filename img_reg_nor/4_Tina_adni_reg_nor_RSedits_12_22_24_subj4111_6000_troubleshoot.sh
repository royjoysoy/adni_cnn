#!/bin/bash
#Usage: ADNI-Registration/Normalization

#$ -S /bin/bash
#$ -N adnijob
#$ -V
#$ -t 1-11          # 11 subjects 가 돌 수 있도록 바꿈 
#$ -cwd
#$ -o sgestdout       # directory where standard output are saved
#$ -e sgestderr       # directory where standard errors are saved
 
# variables
# myprog="/projects4/royseo_workingdir/scripts/3-3-run_fs_modified_plus_noramalized2mni152_1mm_fsl_flirt_linear_transform.sh"
# DIR="/projects4/royseo_workingdir/scripts" # location of the list of name of subject's image files
myprog="/ibic/scratch/royseo_workingdir/scripts/3-3-run_fs_modified_plus_normalized2mni152_1mm_fsl_flirt_linear_transform_subj4111_6000_troubleshoot.sh"
DIR="/ibic/scratch/royseo_workingdir/scripts" # Tina asked me to change these from /projects4/ to /ibic/scratch/ on 12-9-2024 "

### comment this part when you recurit SGE ###
# Set SGE_TASK_ID to 1 if it's not set (for testing outside of SGE)
: "${SGE_TASK_ID:=1}"
##############################################

subject=$(sed -n -e "${SGE_TASK_ID}p" "$DIR/subj_list_ADNI1234_28001_4111-6000_troubleshoot.log")

#PROJ_DIR="/projects4/royseo_workingdir/raw_w_acq_date" #location of image files
PROJ_DIR="/ibic/scratch/royseo_workingdir/raw_w_acq_date" #location of image files

# Debug output
echo "Processing subject: ${subject}"
echo "Full path will be: ${PROJ_DIR}/${subject}"

# cd to the location of the script
cd "/ibic/scratch/royseo_workingdir/scripts" || exit 1

# Make sure the script is executable
chmod +x "$myprog"

# Execute the script with parameters
"$myprog" "${PROJ_DIR}/${subject}"

# Commented out alternative approach:
# for subj in $(cat /projects4/royseo_workingdir/scripts/subj_list_ADNI1234_28001.log); do 
#     echo "run qsub"
#     qsub -q all.q -V -N "fs_${subj}" -o fs.log -e fs_error.log "3-3-run_fs_modified_plus_noramalized2mni152_1mm_fsl_flirt_linear_transform.sh" "${PROJ_DIR}" "${subj}"
# done