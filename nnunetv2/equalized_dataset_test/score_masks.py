import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import nibabel as nib
import numpy as np
import os
import csv


def list_all_nii_files_in_folder(path:str):
    nii_files = [f.split('.')[0] for f in os.listdir(path) if f.endswith('.nii.gz')]
    return nii_files


def decide_slices_to_show(mask, num_slices=5):
    # Get the shape of the mask
    z_dim = mask.shape[2]
    
    # Decide which slices to show
    if z_dim <= num_slices:
        return list(range(z_dim))  # Show all slices if there are fewer than num_slices
    else:
        step = z_dim // num_slices
        return [i * step for i in range(num_slices)]


def plot_image_and_mask(image, mask):
    # open in a plot 5 image's slices in a column
    # and the 5 mask's slices in the second column
    # ask the user to score all from 0 to 10

    slices_to_show = decide_slices_to_show(mask, num_slices=15)

    fig, ax = plt.subplots(3, 5, figsize=(12, 8))
    ax = ax.flatten()  # make it easier to iterate over the axes

    img_displays = []    
    mask_displays = []
    for i, slice_number in enumerate(slices_to_show):
        slice_image = image[:, :, slice_number] 
        slice_mask = mask[:, :, slice_number]
        
        img_display = ax[i].imshow(slice_image, cmap='gray')
        mask_display = ax[i].imshow(slice_mask, alpha=0.3, cmap='jet')

        img_displays.append(img_display)
        mask_displays.append(mask_display)
     
        ax[i].set_title(f'Slice {slice_number}')
        ax[i].axis('off')
    
    ax_button = plt.axes([0.4, 0.01, 0.2, 0.05])  # Position of the button
    button = Button(ax_button, 'Toggle Mask')

    def toggle_mask(event):
        for mask_display in mask_displays:
            if mask_display.get_alpha() > 0:
                mask_display.set_alpha(0)  # Hide the mask
            else:
                mask_display.set_alpha(0.3)  # Show the mask
        fig.canvas.draw_idle()

    button.on_clicked(toggle_mask)

    #plt.tight_layout()  # Adjust spacing between subplots
    #manager = plt.get_current_fig_manager()
    #manager.full_screen_toggle()  
    
    plt.show()

    # ask the user to score the segmentation
    while True:
        try:
            user_score = float(input('Please score the segmentation from 0 to 10: '))
            if 0 <= user_score <= 10:
                break
            else:
                print("Score must be between 0 and 10. Please try again.")
        except:
            print("Invalid input. Please enter a number between 0 and 10.")
        
    return user_score  
    

def evaluate_mask(mask, image, image_name, score_path):
    # normalize the image to [0, 1]
    image = image.astype(np.float32)
    image = image / np.max(image)
    # mask as 8-bit int
    mask = mask.astype(np.uint8) 

    # function to show the mask in a plot
    user_score = plot_image_and_mask(image, mask)

    file_exists = os.path.isfile(score_path)

    with open(score_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # create the header
            writer.writerow(['image_name', 'score'])
        # save the user score
        writer.writerow([image_name, user_score])


if __name__ == '__main__':    
    # obtain the input path
    input_path = os.path.join('..', 'media', 'input_equalized_dataset_test')  # Adjust this path as needed
    input_images = list_all_nii_files_in_folder(input_path)
    assert len(input_images) > 0, "No input images found in the specified folder."

    # obtain all the output images (all the segmentations done)
    output_path = os.path.join('..','media','output_equalized_dataset_test','output_equalized_dataset_test_PP')  # Adjust this path as needed
    output_images = list_all_nii_files_in_folder(output_path)
    assert len(output_images) > 0, "No output images found in the specified folder."
    
    user = input('Please enter your name: ').strip().lower()
    if not user:
        print("Username cannot be empty. Exiting.")
        exit(1)
    
    # open the csv file with scores to see all the images already scored
    current_path = os.path.dirname(os.path.abspath(__file__))
    score_path = os.path.join(current_path, f'{user}_scores.csv')
    file_exists = os.path.isfile(score_path)
    if file_exists:
        with open(score_path, mode='r') as file:
            reader = csv.reader(file)
            scored_images = [row[0] for row in reader]
    else:
        scored_images = []

    # remove the images already scored
    images_to_be_scored = [image for image in output_images if image not in scored_images]


    # IT WAS RECOMMED TO REMOVE THE RANDOM SHUFFLE
    # randomly shuffle the images to be scored by user
    # maybe one user will not validate all the images 
    #name_number = np.sum([ord(char) for char in user])
    #np.random.seed(name_number)  # For reproducibility based on user name
    #np.random.shuffle(images_to_be_scored)

    print(f'There are {len(images_to_be_scored)} from a total of {len(output_images)} images to be scored!')

    for i, image_name in enumerate(images_to_be_scored):
        mask_path = os.path.join(output_path, image_name+'.nii.gz')
        print(f'\rloading mask of {image_name}...', end='', flush=True)
        mask = nib.load(mask_path).get_fdata()
        mask = mask.transpose(1, 0, 2)  # Transpose to match the expected orientation
        image_path = os.path.join(input_path, image_name+'.nii.gz')
        image = nib.load(image_path).get_fdata()
        image = image.transpose(1, 0, 2) # Transpose to match the expected orientation
        print(f'\rloading image {image_name}...', end='', flush=True)
        print(f'\rThe image has {image.shape[-1]} slices, please analyze the segmentation, after closing the plot, you will be asked to score the segmentation from 0 to 10.', flush=True)
        evaluate_mask(mask, image, image_name, score_path=score_path)
        if (i+1) % 10 == 0:
            print(f'Congratulations!!\nYou scored {i+1} images! Thank you for your contribution! :)\nKeep going!')

