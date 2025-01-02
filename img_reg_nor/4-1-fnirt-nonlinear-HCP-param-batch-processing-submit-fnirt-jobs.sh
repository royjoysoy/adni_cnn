#!/bin/bash
# 01-02-2025 Roy Seo, Korea
# This is a submission script for FNIRT batch processing. (3-4-fnirt-nonlinear-HCP-param-batch-processing.sh)
# It creates necessary output directories and submits the job array using qsub.
# output directories: fnirt_output (results), fnirt_stdout (standard output), fnirt_stderr (error logs)


# Create necessary directories
mkdir -p /ibic/scratch/royseo_workingdir/fnirt_output
mkdir -p fnirt_stdout
mkdir -p fnirt_stderr

# Submit the job array
qsub run_fnirt_batch.sh

echo "Submitted FNIRT processing jobs for all subjects"