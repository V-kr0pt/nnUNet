import os
import shutil
import json
from nnunetv2.paths import nnUNet_raw

# Function to rename and move files
def organize_files(src_input_dir, src_output_dir, dest_images_dir, dest_labels_dir, case_identifier):
    input_file = os.path.join(src_input_dir, f'{case_identifier}.nii.gz')
    output_file = os.path.join(src_output_dir, f'{case_identifier}.nii.gz')
    
    # Rename and move input file
    new_input_file = os.path.join(dest_images_dir, f'{case_identifier}_0000.nii.gz')
    shutil.copy(input_file, new_input_file)
    
    # Rename and move output file
    new_output_file = os.path.join(dest_labels_dir, f'{case_identifier}.nii.gz')
    shutil.copy(output_file, new_output_file)


if __name__ == '__main__':
    # Define paths
    base_dir = '../media/source/sam_segmentation'  # Change this to your dataset path
    dataset_id = '995'  # Change this to your desired dataset ID
    dataset_name = 'BreastPectoralSegmentation'  # Change this to your desired dataset name

    # Create nnU-Net dataset directory
    dataset_dir = os.path.join(nnUNet_raw, f'Dataset{dataset_id}_{dataset_name}')
    os.makedirs(dataset_dir, exist_ok=True)

    # Create subdirectories
    images_tr_dir = os.path.join(dataset_dir, 'imagesTr')
    images_ts_dir = os.path.join(dataset_dir, 'imagesTs')
    labels_tr_dir = os.path.join(dataset_dir, 'labelsTr')
    labels_ts_dir = os.path.join(dataset_dir, 'labelsTs')
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(images_ts_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)
    os.makedirs(labels_ts_dir, exist_ok=True)

    # Process training data
    train_input_dir = os.path.join(base_dir, 'train', 'input')
    train_output_dir = os.path.join(base_dir, 'train', 'output')
    for file_name in os.listdir(train_input_dir):
        case_identifier = file_name.split('.')[0]
        organize_files(train_input_dir, train_output_dir, images_tr_dir, labels_tr_dir, case_identifier)

    # Process test data
    test_input_dir = os.path.join(base_dir, 'test', 'input')
    test_output_dir = os.path.join(base_dir, 'test', 'output')
    for file_name in os.listdir(test_input_dir):
        case_identifier = file_name.split('.')[0]
        organize_files(test_input_dir, test_output_dir, images_ts_dir, labels_ts_dir, case_identifier)

    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "BreastImage"
        },
        "labels": {
            "background": 0,
            "pectoral_muscle": 1
        },
        "numTraining": len(os.listdir(images_tr_dir)),
        "file_ending": ".nii.gz"
    }

    with open(os.path.join(dataset_dir, 'dataset.json'), 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Dataset has been successfully transformed and saved to {dataset_dir}")