import argparse
import subprocess
import os

def run_preprocessing(input_folder):
    ... # Placeholder for preprocessing function


def run_inference(input_folder, output_folder):
    print(f"Running inference...\nInput: {input_folder}\nOutput: {output_folder}")
    command = [
        'nnUNetv2_predict',
        '-d', 'Dataset995_BreastPectoralSegmentation',
        '-i', input_folder,
        '-o', output_folder,
        '-f', '0',
        '-tr', 'nnUNetTrainer',
        '-c', '3d_fullres',
        '-p', 'nnUNetPlans'
    ]
    subprocess.run(command, check=True)

def run_postprocessing(output_folder):
    output_folder_pp = output_folder + '_PP'
    print(f"Running postprocessing...\nInput: {output_folder}\nOutput: {output_folder_pp}")
    command = [
        'nnUNetv2_apply_postprocessing',
        '-i', output_folder,
        '-o', output_folder_pp,
        '-pp_pkl_file', '/mnt/d/Users/UFPB/vitor/nn_unet/media/nnUNet_results/Dataset995_BreastPectoralSegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl',
        '-np', '8',
        '-plans_json', '/mnt/d/Users/UFPB/vitor/nn_unet/media/nnUNet_results/Dataset995_BreastPectoralSegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json'
    ]
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run nnUNet inference and optional postprocessing.")
    parser.add_argument('-i', '--input_folder', required=True, help='Path to the input folder')
    parser.add_argument('-o', '--output_folder', required=True, help='Path to the output folder')
    parser.add_argument('--skip_pre', action='store_true', help='Skip preprocessing before inference')
    parser.add_argument('--skip_post', action='store_true', help='Skip postprocessing after inference')    
    args = parser.parse_args()

    # Validate input and output folders
    if not os.path.exists(args.input_folder):
        raise FileNotFoundError(f"Input folder {args.input_folder} does not exist.")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Output folder {args.output_folder} created.")
    input_folder = args.input_folder
    
    # Preprocess input if not skipped
    if not args.skip_pre:
        if not os.path.exists(args.input_folder + '_PP'):
            os.makedirs(args.input_folder + '_PP')
            input_folder = args.input_folder + '_PP' # update input folder to preprocessed one
            print(f"Preprocessed input folder {args.input_folder + '_PP'} created.")
        run_preprocessing(args.input_folder)
        
    
    # Run inference
    run_inference(input_folder, args.output_folder)
    
    if not args.skip_post:
        if not os.path.exists(args.output_folder + '_PP'):
            os.makedirs(args.output_folder + '_PP')
            print(f"Postprocessed output folder {args.output_folder + '_PP'} created.")

        run_postprocessing(args.output_folder)

if __name__ == "__main__":
    main()
