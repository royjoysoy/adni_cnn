# python 4/15/2025 Seattle by Roy Seo
# to transfer ADNI 1,2,3,4 cleaned T1-weighted (n = 4508) to "s3://adni.nrdg.uw.edu" bucket 
# ref: email from Ariel Rokem "Vessles paper and more" 11:31AM 4/14/2025
# ref: email from junehyun63@gmail.com "Clean data에 관한 질문/subj ID와 Image ID 요청"


import os
import csv
import glob
import boto3
from botocore.exceptions import ClientError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_csv_and_extract_ids(csv_file_path):
    """Read the CSV file and extract all 'Image Data ID' values."""
    image_data_ids = []
    
    try:
        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # Extract the ID from the column "Image Data ID"
                image_id = row["Image Data ID"]
                if image_id:  # Only add non-empty IDs
                    image_data_ids.append(image_id)
        
        logger.info(f"Successfully extracted {len(image_data_ids)} Image Data IDs from CSV")
        return image_data_ids
    
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return []

def find_matching_files(image_data_ids, search_directory):
    """Find files in the directory that match the pattern I{Image_Data_ID}.nii"""
    matching_files = {}
    missing_ids = []
    
    for image_id in image_data_ids:
        # Pattern to search for: files ending with I{image_id}.nii
        pattern = os.path.join(search_directory, f"*I{image_id}.nii")
        matching_file_paths = glob.glob(pattern)
        
        if matching_file_paths:
            # Use the first match if multiple are found
            matching_files[image_id] = matching_file_paths[0]
            logger.info(f"Found matching file for ID {image_id}: {matching_file_paths[0]}")
        else:
            missing_ids.append(image_id)
            logger.warning(f"No matching file found for ID {image_id}")
    
    return matching_files, missing_ids

def upload_to_s3(file_paths, bucket_name):
    """Upload the files to the specified S3 bucket."""
    s3_client = boto3.client('s3')
    successful_uploads = []
    failed_uploads = []
    
    for image_id, file_path in file_paths.items():
        try:
            # Extract just the filename for the S3 object key
            file_name = os.path.basename(file_path)
            
            # Upload the file
            logger.info(f"Uploading {file_path} to S3 bucket {bucket_name}")
            s3_client.upload_file(file_path, bucket_name, file_name)
            
            successful_uploads.append((image_id, file_path))
            logger.info(f"Successfully uploaded {file_path} to S3")
            
        except ClientError as e:
            logger.error(f"Error uploading {file_path} to S3: {e}")
            failed_uploads.append((image_id, file_path, str(e)))
    
    return successful_uploads, failed_uploads

def log_results(missing_ids, successful_uploads, failed_uploads, log_file_path):
    """Log missing files, successful uploads, and failed uploads to a text file."""
    with open(log_file_path, 'w') as log_file:
        # Log summary statistics
        log_file.write("=== SUMMARY ===\n")
        log_file.write(f"Total files attempted: {len(successful_uploads) + len(missing_ids) + len(failed_uploads)}\n")
        log_file.write(f"Successfully transferred: {len(successful_uploads)}\n")
        log_file.write(f"Missing files: {len(missing_ids)}\n")
        log_file.write(f"Failed uploads: {len(failed_uploads)}\n\n")
        
        # Log successful uploads
        log_file.write("=== SUCCESSFULLY COPIED FILES ===\n")
        if successful_uploads:
            for image_id, file_path in successful_uploads:
                file_name = os.path.basename(file_path)
                log_file.write(f"ID: {image_id}, File: {file_name}\n")
        else:
            log_file.write("No files were successfully copied\n")
        
        # Log missing files
        log_file.write("\n=== MISSING FILES ===\n")
        if missing_ids:
            for image_id in missing_ids:
                log_file.write(f"No file found for Image Data ID: {image_id}\n")
        else:
            log_file.write("No missing files\n")
        
        # Log failed uploads
        log_file.write("\n=== FAILED UPLOADS ===\n")
        if failed_uploads:
            for image_id, file_path, error in failed_uploads:
                file_name = os.path.basename(file_path)
                log_file.write(f"Failed to upload file for ID {image_id} ({file_name}): {error}\n")
        else:
            log_file.write("No failed uploads\n")
    
    logger.info(f"Results logged to {log_file_path}")

def main():
    # Configuration
    csv_file_path = "/ibic/scratch/royseo_workingdir/behavioral/total_matched_adni_instances.csv"
    search_directory = "/ibic/scratch/royseo_workingdir/raw"
    s3_bucket_name = "adni.nrdg.uw.edu"
    log_file_path = "transfer2AWS_Ariels_041525_log.txt"
    
    # Step 1: Extract IDs from CSV
    image_data_ids = read_csv_and_extract_ids(csv_file_path)
    if not image_data_ids:
        logger.error("No Image Data IDs found in CSV. Exiting.")
        return
    
    # Step 2: Find matching files
    matching_files, missing_ids = find_matching_files(image_data_ids, search_directory)
    logger.info(f"Found {len(matching_files)} matching files. {len(missing_ids)} files are missing.")
    
    # Step 3: Upload files to S3
    if matching_files:
        successful_uploads, failed_uploads = upload_to_s3(matching_files, s3_bucket_name)
        logger.info(f"Successfully uploaded {len(successful_uploads)} files. {len(failed_uploads)} uploads failed.")
    else:
        logger.warning("No matching files found to upload.")
        successful_uploads = []
        failed_uploads = []
    
    # Step 4: Log results with enhanced information
    log_results(missing_ids, successful_uploads, failed_uploads, log_file_path)
    
    # Print summary to console
    print("\n=== TRANSFER SUMMARY ===")
    print(f"Total files attempted: {len(image_data_ids)}")
    print(f"Successfully transferred: {len(successful_uploads)}")
    print(f"Missing files: {len(missing_ids)}")
    print(f"Failed uploads: {len(failed_uploads)}")
    print(f"See {log_file_path} for complete details")
    
    logger.info("Script execution completed")

if __name__ == "__main__":
    main()