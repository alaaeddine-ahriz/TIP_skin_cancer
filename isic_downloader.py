import os
import subprocess
import pandas as pd

# Define parameters
output_dir = "images"
benign_dir = os.path.join(output_dir, "benign")
malignant_dir = os.path.join(output_dir, "malignant")
combined_metadata_file = os.path.join(output_dir, "combined_metadata.csv")

# Ensure output directories exist
os.makedirs(benign_dir, exist_ok=True)
os.makedirs(malignant_dir, exist_ok=True)

# Define commands
benign_command = f'isic image download --search benign_malignant:benign --limit 100 {benign_dir}'
malignant_command = f'isic image download --search benign_malignant:malignant --limit 100 {malignant_dir}'

# Helper function to run commands
def run_command(command, description):
    print(description)
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr}")

# Run commands
run_command(benign_command, "Downloading benign images...")
run_command(malignant_command, "Downloading malignant images...")

# Combine metadata if files exist
benign_metadata_file = os.path.join(benign_dir, "metadata.csv")
malignant_metadata_file = os.path.join(malignant_dir, "metadata.csv")

if os.path.exists(benign_metadata_file) and os.path.exists(malignant_metadata_file):
    print("Combining metadata files...")
    try:
        # Read metadata files, ensuring proper handling of any malformed rows
        benign_metadata = pd.read_csv(benign_metadata_file, engine='python', on_bad_lines='skip')
        malignant_metadata = pd.read_csv(malignant_metadata_file, engine='python', on_bad_lines='skip')
        
        # Combine metadata
        combined_metadata = pd.concat([benign_metadata, malignant_metadata], ignore_index=True)
        
        # Save combined metadata without introducing blank lines
        combined_metadata.to_csv(combined_metadata_file, index=False, line_terminator='\n')
        print(f"Combined metadata saved to {combined_metadata_file}.")
    except Exception as e:
        print(f"Error combining metadata files: {e}")
else:
    print("Metadata files are missing. Check the download commands and queries.")

# Move all images to a single directory
print("Organizing images into a single directory...")
for subdir in [benign_dir, malignant_dir]:
    for root, _, files in os.walk(subdir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):  # Adjust extensions as needed
                src = os.path.join(root, file)
                dst = os.path.join(output_dir, file)
                if not os.path.exists(dst):  # Avoid overwriting files
                    os.rename(src, dst)
                else:
                    print(f"File {file} already exists. Skipping.")

# Cleanup empty directories
for subdir in [benign_dir, malignant_dir]:
    try:
        os.rmdir(subdir)
    except OSError:
        print(f"Directory {subdir} is not empty or could not be removed.")

print("All images and metadata have been organized successfully.")
