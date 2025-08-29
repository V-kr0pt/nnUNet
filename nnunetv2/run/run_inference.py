import subprocess
import os
from nnunetv2.paths import nnUNet_results
from run_utils import downsample_nii_file, upsample_nii_file, flip_nii_file
from post_processing import open_run_and_save_nifti_postprocess

class Inference:
    def __init__(self, input_folder, output_folder, dataset_id, skip_pre=False, skip_post=False, restart_preprocess=False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.dataset_ID = dataset_id
        self.skip_pre = skip_pre
        self.skip_post = skip_post
        self.restart_preprocess = restart_preprocess
        self.downsample_factor = 10
        self.files_shape_factor = {}

        # Validate input and output folders
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder {input_folder} does not exist.")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Output folder {output_folder} created.")


        # Check paths to upload plans
        # the trainner is the same for both
        if int(self.dataset_ID) == 995:
            plan_name = 'nnUNetPlans'
            dataset_name = 'Dataset995_BreastPectoralSegmentation'
        elif int(self.dataset_ID) == 996:
            plan_name = 'PlansFrom995'
            dataset_name = 'Dataset996_BreastPectoralSegmentation_finetunning'
        else:
            raise ValueError('dataset_ID (-did or dataset_id) should be 995 or 996!')
        trainer_path = 'nnUNetTrainer__'+plan_name+'__3d_fullres'
        
        self.plan_name = plan_name
        self.plans_path = os.path.join(nnUNet_results, dataset_name, trainer_path, 'plans.json')
        assert os.path.exists(self.plans_path), f'plans path not found: {self.plans_path}'

    def preprocess_file(self, file, input_folder_pp):
        if file.endswith('.nii.gz'):
            input_file = os.path.join(self.input_folder, file)
            downsampled_file = file.replace('.nii.gz', '')
            downsampled_file = f"downsampled_{downsampled_file}_0000.nii.gz"
            
            # Downsample e salva o fator real
            real_factor = downsample_nii_file(
                nii_file=input_file,
                downsample_factor=self.downsample_factor,
                save_dir=input_folder_pp,
                save_name=downsampled_file
            )
            self.files_shape_factor[file] = (1 / real_factor[0], 1 / real_factor[1], real_factor[2])
            
            # Flip se for uma imagem R_MLO
            #if 'R_MLO' in file:
            #    flip_nii_file(os.path.join(input_folder_pp, downsampled_file))

    def postprocess_file(self, file, output_folder_pp):
        if file.endswith('.nii.gz'):
            output_file = os.path.join(self.output_folder, file)
            upsampled_file = file.replace('downsampled_', '')
            upsampled_file = upsampled_file.replace('_0000', '')

            # postprocess the mask
            open_run_and_save_nifti_postprocess(nii_img_name = file,
                                                nii_img_path = self.output_folder,
                                                save_dir = output_folder_pp,
                                                save_name = upsampled_file)
            
            output_file = os.path.join(output_folder_pp, upsampled_file)
            # flip the output image
            #if 'R_MLO' in file:
            #    flip_nii_file(output_file) 
            
            # upsample here
            upsample_factor = self.files_shape_factor[upsampled_file]
            upsample_nii_file(nii_file = output_file,
                                upsample_factor = upsample_factor,
                                save_dir = output_folder_pp,
                                save_name = upsampled_file)

    
    def run_preprocessing(self):
        input_folder_pp = os.path.basename(self.input_folder) if os.path.basename(self.input_folder) else 'input'
        input_folder_pp += '_PP'
        input_folder_pp = os.path.join(self.input_folder, input_folder_pp)

        # if the preprocessed folder already exists and restart_preprocess was set, remove it
        if os.path.exists(input_folder_pp) and self.restart_preprocess:
            subprocess.run(['rm', '-rf', input_folder_pp])
            os.makedirs(input_folder_pp)
        elif not os.path.exists(input_folder_pp):
            print(f"Creating preprocessed folder: {input_folder_pp}")
            os.makedirs(input_folder_pp)
        else:
            print(f"Preprocessed folder already exists: {input_folder_pp}. Use --restart_preprocess if you want to overwrite it.")
        
        print(f"Running preprocessing...\nInput: {self.input_folder}\nOutput: {input_folder_pp}")
        
        # TODO Parallelize file processing
        all_files_input = os.listdir(self.input_folder)
        all_files_input = [f for f in all_files_input if f.endswith('.nii.gz')]

        already_done_files = [file for file in all_files_input if\
                               os.path.exists(os.path.join(input_folder_pp,
                                                           'downsampled_'+file.strip('.nii.gz')+'_0000.nii.gz')) ]

        all_files_len = len(all_files_input) - len(already_done_files)
        
        if not all_files_len:
            print(f"No files to preprocess in {self.input_folder}.")
            # Open the files_shape_factor file if it exists
            try:
                with open(os.path.join(input_folder_pp, 'files_shape_factor.txt'), 'r') as f:
                    for line in f:
                        key, value = line.strip().split(': ')
                        self.files_shape_factor[key] = eval(value)
                self.input_folder = input_folder_pp  # Update input_folder to the new preprocessed folder
                return
            except FileNotFoundError:
                # If the file does not exist, prompt the user to rerun with --restart_preprocess
                print(f"[ERROR] No files_shape_factor.txt found in {input_folder_pp}.")
                input("Do you want to rerun with --restart_preprocess? Press Enter to continue or Ctrl+C to exit.")
                self.restart_preprocess = True
                self.rerun = True
                self.run_preprocessing()


        for i, file in enumerate(all_files_input):
            print(f"\rPreprocessing file {i+1}/{all_files_len}: {file}", end='', flush=True)
            if file in already_done_files:
                print(f"\rSkipping file {i+1}/{all_files_len}: {file} as it already exists in {input_folder_pp}.", end='', flush=True) 
                continue
            # Preprocess each file
            self.preprocess_file(file, input_folder_pp)
        
        # save the files_shape_factor to a file
        with open(os.path.join(input_folder_pp, 'files_shape_factor.txt'), 'w') as f:
            for key, value in self.files_shape_factor.items():
                f.write(f"{key}: {value}\n")
        
        self.input_folder = input_folder_pp # Update input_folder to the new preprocessed folder
        print(f"\nPreprocessing completed. Preprocessed files saved to {input_folder_pp}.")

    def run_inference(self):
        print(f"Running inference...\nInput: {self.input_folder}\nOutput: {self.output_folder}")
        command = [
            'nnUNetv2_predict',
            '-i', self.input_folder,
            '-o', self.output_folder,
            '-d', str(self.dataset_ID),
            '-c', '3d_fullres',
            '-p', self.plan_name,
            '-tr', 'nnUNetTrainer',
            '-f', '0', '1', '2', '3', '4',
            '-chk', 'checkpoint_best.pth',
            '-npp', '1'
        ]
        subprocess.run(command, check=True)


    def run_postprocessing(self):
        output_folder_pp = os.path.basename(self.output_folder) if os.path.basename(self.output_folder) else 'output'
        output_folder_pp += '_PP'
        output_folder_pp = os.path.join(self.output_folder, output_folder_pp)
        os.makedirs(output_folder_pp, exist_ok=True)
        print(f"Running postprocessing...\nInput: {self.output_folder}\nOutput: {output_folder_pp}")
        command = [
            'nnUNetv2_apply_postprocessing',
            '-i', self.output_folder,
            '-o', self.output_folder,
            '-pp_pkl_file', '/mnt/d/Users/UFPB/vitor/nn_unet/media/nnUNet_results/Dataset995_BreastPectoralSegmentation/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl',
            '-np', '8',
            '-plans_json', self.plans_path
        ]
        subprocess.run(command, check=True)

        # Upsample and flip
        all_files_output = os.listdir(self.output_folder)
        all_files_output = [f for f in all_files_output if f.endswith('.nii.gz')]
        all_files_len = len(all_files_output)
        print('...')
        for i, file in enumerate(all_files_output):
            print(f"\rPostprocessing file {i+1}/{all_files_len}: {file}", end='', flush=True)
            # Postprocess each file
            self.postprocess_file(file, output_folder_pp)    
                
        print(f"Postprocessing completed. Postprocessed files saved to {output_folder_pp}.")
    
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

    DEFAULT_INPUT_FOLDER = os.path.join('..','media','bad_performance_imgs')
    DEFAULT_OUTPUT_FOLDER = os.path.join('..','media','output_995_bad_performance_imgs')
    DEFAULT_ID = 995

    parser = argparse.ArgumentParser(description="Run nnUNet inference and optional postprocessing.")
    parser.add_argument('-i', '--input_folder', default=DEFAULT_INPUT_FOLDER, help='Path to the input folder')
    parser.add_argument('-o', '--output_folder', default=DEFAULT_OUTPUT_FOLDER, help='Path to the output folder')
    parser.add_argument('-did', '--dataset_id', default=DEFAULT_ID, help='Dataset model ID')
    parser.add_argument('--skip_pre', action='store_true', help='Skip preprocessing before inference')
    parser.add_argument('--skip_post', action='store_true', help='Skip postprocessing after inference')
    parser.add_argument('--restart_preprocess', action='store_true', help='Restart preprocessing even if it was done before')    
    args = parser.parse_args()
    

    inference = Inference(args.input_folder, args.output_folder, args.dataset_id,
                           args.skip_pre, args.skip_post, args.restart_preprocess)
    
    confirm_inf_model_str = f'Are sure you want to run inferece for {inference.input_folder} -> {inference.output_folder}\n'
    confirm_inf_model_str += f'using the model with ID {inference.dataset_ID} ? (Y/n): '
    
    if input(confirm_inf_model_str) == 'n':
        print('Change configs and rerun.')
        exit()

    inference.run()
