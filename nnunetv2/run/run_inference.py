import subprocess
import os
from run_utils import downsample_nii_file, flip_nii_file
from post_processing import open_run_and_save_nifti_postprocess

class Inference:
    def __init__(self, input_folder, output_folder, skip_pre=False, skip_post=False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.skip_pre = skip_pre
        self.skip_post = skip_post

        # Validate input and output folders
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder {input_folder} does not exist.")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Output folder {output_folder} created.")

    
    def run_preprocessing(self):
        input_folder = os.path.basename(self.input_folder) if os.path.basename(self.input_folder) else 'input'
        input_folder += '_PP'
        input_folder = os.path.join(self.input_folder, input_folder)
        os.makedirs(input_folder, exist_ok=True)
        print(f"Running preprocessing...\nInput: {self.input_folder}\nOutput: {input_folder}")
        
        # Downsample and flip
        for file in os.listdir(self.input_folder):
            if file.endswith('.nii.gz'):
                input_file = os.path.join(self.input_folder, file)
                downsampled_file = f"downsampled_{file}"
                
                downsample_nii_file(nii_file = input_file,
                                    downsample_factor = 10,
                                    save_dir = input_folder,
                                    save_name = downsampled_file)
                
                if 'R_MLO' in file:
                    flip_nii_file(os.path.join(input_folder, downsampled_file)) # flip the input image


        self.input_folder = input_folder # Update input_folder to the new preprocessed folder
        print(f"Preprocessing completed. Preprocessed files saved to {input_folder}.")

    def run_inference(self):
        print(f"Running inference...\nInput: {self.input_folder}\nOutput: {self.output_folder}")
        command = [
            'nnUNetv2_predict',
            '-i', self.input_folder,
            '-o', self.output_folder,
            '-d', '995',
            '-c', '3d_fullres',
            '-f', '0',
            '-step_size', '1',
            '--disable_tta',
            '-npp', '1'
        ]
        subprocess.run(command, check=True)


    def run_postprocessing(self):
        output_folder = os.path.basename(self.output_folder) if os.path.basename(self.output_folder) else 'output'
        output_folder += '_PP'
        output_folder = os.path.join(self.output_folder, output_folder)
        os.makedirs(output_folder, exist_ok=True)
        print(f"Running postprocessing...\nInput: {self.output_folder}\nOutput: {output_folder}")
        command = [
            'nnUNetv2_apply_postprocessing',
            '-i', self.output_folder,
            '-o', self.output_folder,
            '-pp_pkl_file', '/mnt/d/Users/UFPB/vitor/nn_unet/media/nnUNet_results/Dataset995_BreastPectoralSegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl',
            '-np', '8',
            '-plans_json', '/mnt/d/Users/UFPB/vitor/nn_unet/media/nnUNet_results/Dataset995_BreastPectoralSegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json'
        ]
        subprocess.run(command, check=True)

        # Upsample and flip
        for file in os.listdir(self.output_folder):
            if file.endswith('.nii.gz'):
                output_file = os.path.join(self.output_folder, file)
                upsampled_file = file.replace('downsampled_', '')

                # postprocess the mask
                open_run_and_save_nifti_postprocess(nii_img_name = file,
                                                    nii_img_path = self.output_folder,
                                                    save_dir = output_folder,
                                                    save_name = upsampled_file)
                
                # flip the output image
                if 'R_MLO' in file:
                    flip_nii_file(os.path.join(output_folder, upsampled_file)) 
                
                # upsample here
                downsample_nii_file(nii_file = output_file,
                                    downsample_factor = 0.1,
                                    save_dir = output_folder,
                                    save_name = upsampled_file)
                
        print(f"Postprocessing completed. Postprocessed files saved to {output_folder}.")
    
    def run(self):
        # Preprocess input if not skipped
        if not self.skip_pre:
            self.run_preprocessing()
            
        # Run inference
        self.run_inference()
        
        if not self.skip_post:
            self.run_postprocessing()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run nnUNet inference and optional postprocessing.")
    parser.add_argument('-i', '--input_folder', required=True, help='Path to the input folder')
    parser.add_argument('-o', '--output_folder', required=True, help='Path to the output folder')
    parser.add_argument('--skip_pre', action='store_true', help='Skip preprocessing before inference')
    parser.add_argument('--skip_post', action='store_true', help='Skip postprocessing after inference')    
    args = parser.parse_args()
    
    #input_folder = '../media/input'
    #output_folder = '../media/output'
    #inference = Inference(input_folder, output_folder, False, True)

    inference = Inference(args.input_folder, args.output_folder, args.skip_pre, args.skip_post)
    inference.run()
