#!/bin/bash
# 01-02-2025 Roy Seo, Korea
# This is a submission script for FNIRT batch processing. (3-4-fnirt-nonlinear-HCP-param-batch-processing.sh)
# It creates necessary output directories and submits the job array using qsub.
# output directories: fnirt_stdout (standard output), fnirt_stderr (error logs)
# 01-04-2025 subject 1-10


# Create necessary directories
mkdir -p fnirt_stdout
mkdir -p fnirt_stderr

# Submit the job array
#qsub 3-4-fnirt-nonlinear-HCP-param-batch-processing.sh
qsub 3-4-fnirt-nonlinear-HCP-param-batch-processing-1-10.sh
#qsub 3-4-fnirt-nonlinear-HCP-param-batch-processing-11-6000.sh

echo "Submitted FNIRT processing jobs for all subjects"