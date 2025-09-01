import score_masks as sm
import os

if __name__ == '__main__':
    # Setup paths
    id = 996
    main_path = os.path.join('finetunning_results', 'validation_inputs')
    input_path = main_path
    output_path = 'masks_finetune_model' if id == 996 else 'masks_original_model'
    input_path = os.path.join('..', 'media', input_path)
    output_path = os.path.join('..', 'media', main_path, output_path)
    
    sm.main(input_path, output_path)