#!/bin/bash
#Usage: ADNI-Registeration/Normalization

PROJ_DIR=/project_space/ADNI_ROY/test #location of subject directories

#cd to the location of the script
cd /project_space/ADNI_ROY/scripts

for subj in `cat /project_space/ADNI_ROY/scripts/list_subj.log`; do 
	echo run qsub
	qsub -q global.q -V -N fs_${subj} -o fs.log -e fs_error.log run_fs_modified ${PROJ_DIR} ${subj}
done