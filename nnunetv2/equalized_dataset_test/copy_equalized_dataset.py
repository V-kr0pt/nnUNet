import os
import pandas as pd
import numpy as np
import subprocess
import SimpleITK as sitk
import pydicom
import logging

class CopyEqualizedDataset:
    def __init__(self):
        csv_path = os.path.join('nnunetv2','equalized_dataset_test','random_selected_MLO.csv')
        self.df = pd.read_csv(csv_path)

        self.destination_path = os.path.join('..','media','input_equalized_dataset_test')
        os.makedirs(self.destination_path, exist_ok=True)

        # logger configuration
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_filename = os.path.join(current_dir, 'copy_errors.log')
        logging.basicConfig(
            filename=log_filename,
            level=logging.ERROR,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        
    def run_copy(self):
        # Iterate through each row in the DataFrame
        self.copy_files()
        
        # Convert DICOM to NIfTI
        self.stack_dicom_series()

    
    def copy_files(self):
        nb_rows = self.df.shape[0]
        for index, row in self.df.iterrows():
            dummy_id = str(row['Dummy_ID']).zfill(8)  # Ensure Dummy_ID is zero-padded to 8 digits
            subfolder_L = row['Subfolder_L']
            subfolder_R = row['Subfolder_R']
            birads_density = row['birads_density']

            origin_path_left = f'/mnt/rede/{dummy_id}/PROC_Tomo_RC/{subfolder_L}'
            origin_path_right= f'/mnt/rede/{dummy_id}/PROC_Tomo_RC/{subfolder_R}'
            print(f"{index}/{nb_rows}", flush=True)
            # Command to copy the folder
            try:
                command = f"cp -r {origin_path_left} {self.destination_path}"
                # Execute the command
                subprocess.run(command, shell=True, check=True)    
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to copy {subfolder_L}", flush=True)
                self.logger.error(f"Failed to copy {subfolder_L} from {origin_path_left} - BIRADS: {birads_density}")
                continue
            try:
                command = f"cp -r {origin_path_right} {self.destination_path}"
                # Execute the command
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to copy {subfolder_R}", flush=True)
                self.logger.error(f"Failed to copy {subfolder_R} from {origin_path_right} - BIRADS: {birads_density}")
                continue
            

    def stack_dicom_series(self):
        dicom_folders = os.listdir(self.destination_path)
        for folder in dicom_folders:
            input_folder = os.path.join(self.destination_path, folder)
            if not os.path.isdir(input_folder):
                continue
            
            output_file = os.path.join(self.destination_path, f"{folder}.nii.gz")
            if os.path.exists(output_file):
                print(f"[SKIP] NiFTI file already exists: {output_file}")
                continue

            print(f"Processing folder: {input_folder}")

            if not os.path.exists(input_folder):
                print(f"[ERROR] Folder does not exist: {input_folder}")
                continue

            if not os.listdir(input_folder):
                print(f"[ERROR] Folder is empty: {input_folder}")
                continue

            # Read DICOM files and stack them
            print(f"Reading DICOM files from {input_folder}")
            slices = []
            sorted_files = sorted(os.listdir(input_folder))
            for fname in sorted_files:
                path = os.path.join(input_folder, fname)
                try:
                    ds = pydicom.dcmread(path)
                    slices.append(ds.pixel_array)
                except pydicom.errors.InvalidDicomError as e:
                    print(f"[ERROR] Invalid DICOM file: {path}")
                    self.logger.error(f"Invalid DICOM file: {path} - {e}")
                    break
            
            if len(slices) == 0:
                print("No valid DICOM slices found in the folder.")
                return

            volume = np.stack(slices, axis=0)  # [Z, Y, X]
            image = sitk.GetImageFromArray(volume)
            sitk.WriteImage(image, output_file)
            print(f"[OK] Saved NiFTI file: {output_file}")


if __name__ == "__main__":
    copier = CopyEqualizedDataset()
    copier.run_copy()
    print("All files copied and converted to NIfTI format successfully.")