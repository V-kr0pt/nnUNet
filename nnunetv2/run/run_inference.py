import subprocess
import os

class Inferece:
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
        os.makedirs(self.input_folder + '_PP', exist_ok=True)
        print(f"Running preprocessing...\nInput: {self.input_folder}\nOutput: {self.input_folder + '_PP'}")
        self.input_folder = self.input_folder + '_PP'
        ... # I'll put a preprocessing here (I should resize the images)


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
        output_folder_pp = self.output_folder + '_PP'
        print(f"Running postprocessing...\nInput: {self.output_folder}\nOutput: {output_folder_pp}")
        command = [
            'nnUNetv2_apply_postprocessing',
            '-i', self.output_folder,
            '-o', output_folder_pp,
            '-pp_pkl_file', '/mnt/d/Users/UFPB/vitor/nn_unet/media/nnUNet_results/Dataset995_BreastPectoralSegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl',
            '-np', '8',
            '-plans_json', '/mnt/d/Users/UFPB/vitor/nn_unet/media/nnUNet_results/Dataset995_BreastPectoralSegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json'
        ]
        subprocess.run(command, check=True)


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
    
    inferece = Inferece(args.input_folder, args.output_folder, args.skip_pre, args.skip_post)
    inferece.run()
