import pandas as pd
import os


df = pd.read_csv('nnunetv2/equalized_dataset_test/random_selected_MLO.csv')
folders = df['Subfolder_L'].tolist() + df['Subfolder_R'].tolist()


#rede_folders = os.listdir('/mnt/rede/')
#
#print(f"Total folders in rede: {len(rede_folders)}")

input_folders = [f for f in os.listdir('../media/input_equalized_dataset_test/') if os.path.isdir(os.path.join('../media/input_equalized_dataset_test/', f))]
print(f"Total folders in input_folders: {len(input_folders)}")

# I want to see which folders are in the input_folders but not in the folders list
delete_folders = [f for f in input_folders if f not in folders]
print(f"Total folders to delete: {len(delete_folders)}")
for folder in delete_folders:
    folder_path = os.path.join('../media/input_equalized_dataset_test/', folder)
    print(f"Deleting folder: {folder_path}")
    os.system(f'rm -rf {folder_path}')  # Use os.system to remove the folder
    os.system(f'rm -rf {folder_path}.nii.gz')  # Remove the corresponding NIfTI file if it exists