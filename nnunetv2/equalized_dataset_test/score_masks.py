import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import nibabel as nib
import numpy as np
import os
import csv
from collections import deque
from concurrent.futures import ThreadPoolExecutor

NUMBER_OF_PRELOAD = 3  # Number of images to preload

class ImageLoader:
    def __init__(self, input_path, output_path, max_preload=3):
        self.input_path = input_path
        self.output_path = output_path
        self.max_preload = max_preload
        self.cache = {}
        self.loading_queue = deque()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preload_futures = {}
        
    def _load_image_data(self, image_name):
        """Load both mask and image data"""
        try:
            mask_path = os.path.join(self.output_path, image_name + '.nii.gz')
            image_path = os.path.join(self.input_path, image_name + '.nii.gz')
            
            mask = nib.load(mask_path).get_fdata()
            mask = mask.transpose(1, 0, 2)
            
            image = nib.load(image_path).get_fdata()
            image = image.transpose(1, 0, 2)
            
            # Normalize image
            image = image.astype(np.float32)
            image = image / np.max(image)
            mask = mask.astype(np.uint8)
            
            return image, mask
        except Exception as e:
            print(f"Error loading {image_name}: {e}")
            return None, None
    
    def preload_images(self, image_names, current_index):
        """Preload next few images"""
        # Cancel any ongoing preload tasks that are no longer needed
        for name in list(self.preload_futures.keys()):
            if name not in image_names[current_index:current_index + self.max_preload]:
                future = self.preload_futures.pop(name)
                future.cancel()
        
        # Start preloading next images
        for i in range(current_index, min(current_index + self.max_preload, len(image_names))):
            image_name = image_names[i]
            if image_name not in self.cache and image_name not in self.preload_futures:
                future = self.executor.submit(self._load_image_data, image_name)
                self.preload_futures[image_name] = future
    
    def get_image_data(self, image_name):
        """Get image data, either from cache or by loading"""
        if image_name in self.cache:
            return self.cache[image_name]
        
        if image_name in self.preload_futures:
            future = self.preload_futures.pop(image_name)
            try:
                result = future.result(timeout=30)  # 30 second timeout
                self.cache[image_name] = result
                return result
            except Exception as e:
                print(f"Error getting preloaded data for {image_name}: {e}")
        
        # Fallback to synchronous loading
        print(f"Loading {image_name} synchronously...")
        result = self._load_image_data(image_name)
        self.cache[image_name] = result
        return result
    
    def cleanup_cache(self, keep_images=None):
        """Clean up cache to free memory"""
        if keep_images is None:
            keep_images = []
        
        to_remove = [name for name in self.cache.keys() if name not in keep_images]
        for name in to_remove:
            del self.cache[name]

def list_all_nii_files_in_folder(path: str):
    nii_files = [f.split('.')[0] for f in os.listdir(path) if f.endswith('.nii.gz')]
    return nii_files

def update_images_to_be_scored(output_path, score_path, user, min_scores=3):
    output_images = list_all_nii_files_in_folder(output_path)
    assert len(output_images) > 0, "No output images found in the specified folder"

    image_user_count = {img: set() for img in output_images}

    if os.path.isfile(score_path):
        with open(score_path, mode='r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i == 0: # Skip header row
                    continue
                img = row[0]
                usr = row[2] # user who scored the image
                if img in image_user_count:
                    image_user_count[img].add(usr)

    # Only include images with less than min_scores scores
    # and not scored by the current user
    images_to_be_scored = [
        img for img, users in image_user_count.items() 
        if len(users) < min_scores and (user not in users)
    ]

    return images_to_be_scored

def decide_slices_to_show(mask, num_slices=15):
    z_dim = mask.shape[2]
    
    if z_dim <= num_slices:
        return list(range(z_dim))
    else:
        step = z_dim // num_slices
        return [i * step for i in range(num_slices)]

class ImageViewer:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.img_displays = []
        self.mask_displays = []
        self.current_score = None
        self.score_submitted = False
        
    def setup_score_input(self):
        """Setup score input with keyboard shortcuts"""
        self.score_text = self.fig.text(0.5, 0.02, 'Score: 0-9 keys, X or Space for 10, Enter to confirm', 
                                       ha='center', fontsize=12, weight='bold')
        
        def on_key_press(event):
            if event.key in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                score = int(event.key)
                self.current_score = score
                self.score_text.set_text(f'Score: {score} (Press Enter to confirm, or another key to change)')
                self.fig.canvas.draw_idle()
            elif event.key in ['x', 'X', ' ']:  # 'x', 'X', or spacebar for score 10
                score = 10
                self.current_score = score
                self.score_text.set_text(f'Score: {score} (Press Enter to confirm, or another key to change)')
                self.fig.canvas.draw_idle()
            elif event.key == 'enter' and self.current_score is not None:
                self.score_submitted = True
                plt.close(self.fig)
        
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    def plot_image_and_mask(self, image, mask):
        slices_to_show = decide_slices_to_show(mask, num_slices=15)
        
        self.fig, self.ax = plt.subplots(3, 5, figsize=(15, 10))
        self.ax = self.ax.flatten()
        
        self.img_displays = []
        self.mask_displays = []
        
        for i, slice_number in enumerate(slices_to_show):
            slice_image = image[:, :, slice_number]
            slice_mask = mask[:, :, slice_number]
            
            img_display = self.ax[i].imshow(slice_image, cmap='gray')
            mask_display = self.ax[i].imshow(slice_mask, alpha=0.3, cmap='jet')
            
            self.img_displays.append(img_display)
            self.mask_displays.append(mask_display)
            
            self.ax[i].set_title(f'Slice {slice_number}')
            self.ax[i].axis('off')
        
        # Toggle mask button
        ax_button = plt.axes([0.4, 0.08, 0.2, 0.04])
        button = Button(ax_button, 'Toggle Mask (T)')
        
        def toggle_mask(event):
            for mask_display in self.mask_displays:
                if mask_display.get_alpha() > 0:
                    mask_display.set_alpha(0)
                else:
                    mask_display.set_alpha(0.3)
            self.fig.canvas.draw_idle()
        
        def on_key_press_toggle(event):
            if event.key == 't':
                toggle_mask(event)
        
        button.on_clicked(toggle_mask)
        self.fig.canvas.mpl_connect('key_press_event', on_key_press_toggle)
        
        self.setup_score_input()
        
        #plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Instructions
        self.fig.text(0.5, 0.06, 'Instructions: Keys 0-9 for score, X/Space for 10, Enter to confirm, T to toggle mask', 
                     ha='center', fontsize=10, style='italic')
        
        plt.show()
        
        # Fallback to input if keyboard scoring wasn't used
        if not self.score_submitted:
            while True:
                try:
                    user_score = float(input('Please score the segmentation from 0 to 10 or use CRTL+C/CTRL+D to exit: '))
                    if 0 <= user_score <= 10:
                        self.current_score = user_score
                        break
                    else:
                        print("Score must be between 0 and 10. Please try again.")
                except:
                    print("Invalid input. Please enter a number between 0 and 10.")
        
        return self.current_score

def evaluate_mask(mask, image, image_name, score_path, user='unknown'):
    if user == 'unknown':
        user = input('Please enter your name: ').strip().lower()
        if not user:
            print("Username cannot be empty. Exiting.")
            exit(1)
    
    def clean_text(s):
        return str(s).encode('utf-8', 'ignore').decode('utf-8')
    
    viewer = ImageViewer()
    user_score = viewer.plot_image_and_mask(image, mask)
    
    file_exists = os.path.isfile(score_path)
    
    with open(score_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['image_name', 'score', 'user'])
        writer.writerow([clean_text(image_name), user_score, clean_text(user)])
    
    return user_score

def main():
    # Setup paths
    input_path = os.path.join('..', 'media', 'input_equalized_dataset_test')
    output_path = os.path.join('..', 'media', 'output_equalized_dataset_test', 'output_equalized_dataset_test_PP')
    
    # Verify paths exist
    input_images = list_all_nii_files_in_folder(input_path)
    output_images = list_all_nii_files_in_folder(output_path)
    assert len(input_images) > 0, "No input images found."
    assert len(output_images) > 0, "No output images found."
    
    # Get user info
    user = input('Please enter your name: ').strip().lower()
    if not user:
        print("Username cannot be empty. Exiting.")
        exit(1)
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    score_path = os.path.join(current_path, 'scores', f'all_scores.csv')
    
    # Get images to be scored
    images_to_be_scored = update_images_to_be_scored(output_path, score_path, user)
    print(f'There are {len(images_to_be_scored)} from a total of {len(output_images)} images to be scored!')
    
    if len(images_to_be_scored) == 0:
        print("No images to score!")
        return
    
    # Initialize image loader
    loader = ImageLoader(input_path, output_path, max_preload=NUMBER_OF_PRELOAD)
    
    # Start preloading first few images
    print("Starting to preload images...")
    loader.preload_images(images_to_be_scored, 0)
    
    # Process images
    for i, image_name in enumerate(images_to_be_scored):
        print(f'\n--- Processing image {i+1}/{len(images_to_be_scored)}: {image_name} ---')
        
        # Get current image data
        image, mask = loader.get_image_data(image_name)
        
        if image is None or mask is None:
            print(f"Skipping {image_name} due to loading error")
            continue
        
        # Start preloading next images
        loader.preload_images(images_to_be_scored, i + 1)
        
        print(f'Image has {image.shape[-1]} slices. Analyze the segmentation and score from 0 to 10.')
        
        # Evaluate mask
        score = evaluate_mask(mask, image, image_name, score_path, user)
        print(f'Score recorded: {score}')
        
        # Clean up cache periodically to save memory
        if i % 5 == 0:
            keep_next_few = images_to_be_scored[i:i+NUMBER_OF_PRELOAD]
            loader.cleanup_cache(keep_next_few)
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f'\n Congratulations!! You scored {i+1} images! Thank you for your contribution! ')
            print('Keep going!')
    
    print(f'\n Congratulations!! You scored all {len(images_to_be_scored)} images! You are awesome! ')

if __name__ == '__main__':
    main()