#03-05-2025 South Korea Roy Seo
#!/usr/bin/env python3

"""
FSL 1mm MNI Normalization File Copy and Rename Script
(raw to 1mm MNI normalization files)

This script copies T1_to_MNI_nonlin_1mm.nii.gz files from .anat directories 
in the source directory to a target directory, renaming them according to 
the following pattern:
- Remove '.anat' from the original folder name
- Add the original filename with an underscore
- Example: 941_S_1363_2007-03-12_S28008_I63896.anat/T1_to_MNI_nonlin_1mm.nii.gz
  becomes 941_S_1363_2007-03-12_S28008_I63896_T1_to_MNI_nonlin_1mm.nii.gz

Features:
- Automatically creates target directory if it doesn't exist
- Displays progress during processing
- Handles errors for missing files or directories
- Creates a log file of processed files and any errors
"""

import os
import shutil
import glob
import sys
from datetime import datetime

def main():
    # Define source and target directories
    source_dir = "/ibic/scratch/royseo_workingdir/fsl_anat/processed_direct_1mm"
    target_dir = "/ibic/scratch/royseo_workingdir/fsl_anat/processed/direct_1mm"
    
    # Setup logging
    log_file = os.path.join(target_dir, f"copy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
            print(f"Created target directory: {target_dir}")
        except Exception as e:
            print(f"Error creating target directory: {e}")
            sys.exit(1)
    
    # Find all .anat directories
    anat_dirs = glob.glob(os.path.join(source_dir, "*.anat"))
    total_dirs = len(anat_dirs)
    
    print(f"Found {total_dirs} .anat directories to process")
    
    # Initialize counters
    processed = 0
    errors = 0
    
    # Open log file
    with open(log_file, 'w') as log:
        log.write(f"Copy and rename log - Started at {datetime.now()}\n")
        log.write(f"Source directory: {source_dir}\n")
        log.write(f"Target directory: {target_dir}\n")
        log.write("-" * 80 + "\n\n")
        
        # Process each .anat directory
        for i, anat_dir in enumerate(anat_dirs, 1):
            base_name = os.path.basename(anat_dir)
            source_file = os.path.join(anat_dir, "T1_to_MNI_nonlin.nii.gz")
            
            # Create new filename by removing .anat and adding underscore
            new_name = base_name.replace(".anat", "") + "_T1_to_MNI_nonlin.nii.gz"
            target_file = os.path.join(target_dir, new_name)
            
            # Show progress
            progress = (i / total_dirs) * 100
            print(f"Processing [{i}/{total_dirs}] ({progress:.1f}%): {base_name}", end="\r")
            
            # Check if source file exists
            if not os.path.exists(source_file):
                error_msg = f"ERROR: Source file not found: {source_file}"
                print(f"\n{error_msg}")
                log.write(f"{error_msg}\n")
                errors += 1
                continue
            
            # Copy and rename file
            try:
                shutil.copy2(source_file, target_file)
                log.write(f"SUCCESS: Copied {base_name}/T1_to_MNI_nonlin.nii.gz to {new_name}\n")
                processed += 1
            except Exception as e:
                error_msg = f"ERROR: Failed to copy {base_name}: {str(e)}"
                print(f"\n{error_msg}")
                log.write(f"{error_msg}\n")
                errors += 1
        
        # Write summary to log
        summary = f"\nSummary: Processed {processed} files with {errors} errors"
        log.write("-" * 80 + "\n")
        log.write(summary)
        log.write(f"\nCompleted at {datetime.now()}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print(summary)
    print(f"Log file created at: {log_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()