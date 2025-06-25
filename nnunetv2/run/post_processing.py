import os
import cv2
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import skimage.measure as measure
from collections import deque
from run_utils import open_nifti_image


def keep_largest_component(mask):
    '''
    Keep the largest connected component in a binary mask.
    Args:
        mask (np.ndarray): The input binary mask.
    Returns:
        np.ndarray: The binary mask with only the largest connected component retained.
    '''

    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        return mask
    largest = max(props, key=lambda x: x.area)
    return (labels == largest.label).astype(np.uint8)


def morphological_closing(mask, kernel_size=15):
    '''
    Apply morphological closing to a binary mask.
    Args:
        mask (np.ndarray): The input binary mask.
        kernel_size (int): The size of the structuring element for morphological operations.
    Returns:
        np.ndarray: The binary mask after applying morphological closing.
    '''

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed.astype(np.uint8)

def smooth_segmentation_volume(volume):
    '''
    Smooth a 3D binary segmentation volume using Gaussian filtering.
    Args:
        volume (np.ndarray): The input 3D binary segmentation volume.
    Returns:
        np.ndarray: The smoothed binary segmentation volume.'''
    smothed = gaussian_filter(volume.astype(np.float32), sigma=1, radius=[10,10,5]) > 0.5
    return smothed.astype(np.uint8)

def identify_mask_quadrant(mask):
    '''
    Identify the dominant quadrant of a binary mask.
    Args:
        mask (np.ndarray): The input binary mask.
    Returns:
        str: The name of the dominant quadrant ('top_left', 'top_right', 'bottom_left', 'bottom_right')
    '''

    rows, cols = mask.shape
    quadrants = {
        'top_left': mask[:rows//2, :cols//2],
        'top_right': mask[:rows//2, cols//2:],
        'bottom_left': mask[rows//2:, :cols//2],
        'bottom_right': mask[rows//2:, cols//2:]
    }
    count = {name: np.sum(q) for name, q in quadrants.items()}
    dominant_quadrant = max(count, key=count.get)
    
    return dominant_quadrant

def mask_to_3_quadrant(mask, original_quadrant):
    '''Convert a mask to a 3-quadrant representation based on the original quadrant. 
        If the mask is already converted it will return to the original quadrant.
    Args:
        mask (np.ndarray): The input mask to be converted.
        original_quadrant (str): The original quadrant of the mask ('top_left', 'top_right', 'bottom_left', 'bottom_right').
    Returns:
        np.ndarray: The converted mask in the 3-quadrant representation.
    '''
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    if original_quadrant == 'top_left':
        # Flip horizontally
        new_mask = np.flip(mask, axis=1)  
    elif original_quadrant == 'top_right':
        # Flip vertically and horizontally
        new_mask = np.flip(mask, axis=0)
        new_mask = np.flip(new_mask, axis=1) 
    elif original_quadrant == 'bottom_left':
        # Doing nothing
        new_mask = mask.copy()    
    elif original_quadrant == 'bottom_right':
        # Flip vertically
        new_mask = np.flip(mask, axis=0)

    return new_mask
        

def l_rool(slice):
    '''Perform a L-rool operation on the mask.
    Args:
        mask (np.ndarray): The input binary mask.
    Returns:
        np.ndarray: The mask after L-rool operation.
    '''

    rows, cols = slice.shape
    queue = deque()
    
    # Initialize queue with all active pixels
    for i in range(rows):
        for j in range(cols):
            if slice[i, j] == 1:
                queue.append((i, j))
    while queue:
        i, j = queue.popleft()
        
        # Check neighbor bellow (i+1, j)
        if i < rows - 1 and slice[i + 1, j] == 0:
            slice[i+1, j] = 1
            queue.append((i+1, j))
        
        # Check neighbor to the right (i, j+1)
        if j > 0 and slice[i, j - 1] == 0:
            slice[i,j-1] = 1
            queue.append((i, j-1))
    
    return slice.astype(np.uint8)


def contour_smoothing(mask):
    '''
    Smooth the contours of a binary mask using the Douglas-Peucker algorithm.
    Args:
        mask (np.ndarray): The input binary mask.
    Returns:
        np.ndarray: The binary mask with smoothed contours.
    '''
    mask = np.ascontiguousarray(mask.astype(np.uint8))

    # Find all contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return mask  # Return original if no contours found

    # Select only the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Smooth the contour using the Douglas-Peucker algorithm
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)  # Adjust for smoothness
    smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    #Create a new mask with the smoothed contour
    smoothed_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(smoothed_mask, [smoothed_contour], -1, 1, thickness=cv2.FILLED)

    return smoothed_mask

def postprocess_mask(mask):
    '''
    Postprocess a binary mask by applying morphological operations.
    This function processes each slice of the 3D mask independently, keeping the largest connected component,
    applying morphological closing, and performing L-rool operation. It also identifies the mask's quadrant to apply the L-rool operation correctly.
    The mask is expected to be a 3D numpy array where each slice is a 2D binary mask.
    Args:
        mask (np.ndarray): The input binary mask.
    Returns:
        np.ndarray: The postprocessed binary mask.
    '''

    print("Postprocessing mask...")
    total_slices = mask.shape[2]
    # flip columns to match the expected orientation
    #mask = mask.transpose(1, 0, 2)
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for slice_idx in range(total_slices):
        slice_mask = mask[:, :, slice_idx]
        slice_mask = keep_largest_component(slice_mask)
        slice_mask = morphological_closing(slice_mask)

        # Identify the mask's quadrant to use l-rool and after reconvert to the original quadrant
        #original_quadrant = identify_mask_quadrant(slice_mask)
        #slice_mask = mask_to_3_quadrant(slice_mask, original_quadrant)
        #slice_mask = l_rool(slice_mask)
        #slice_mask = mask_to_3_quadrant(slice_mask, original_quadrant)
        
        # save the processed slice back to the copy_mask
        new_mask[:, :, slice_idx] = slice_mask

        print(f"Processing slice {slice_idx + 1}/{total_slices}", end='\r')
    #mask = mask.transpose(0, 1, 2)
    return new_mask

def open_run_and_save_nifti_postprocess(nii_img_name, nii_img_path, save_dir, save_name):
    '''
    Open a NIfTI image, postprocess the mask, and save it to the specified directory.
    Args:
        nii_img_name (str): The name of the NIfTI image file.
        nii_img_path (str): The path to the directory containing the NIfTI image.
        save_dir (str): The directory where the postprocessed mask will be saved.
        save_name (str): The name for the saved postprocessed mask file.
    Returns:
        None
    '''

    mask = open_nifti_image(nii_img_name=nii_img_name, nii_img_path=nii_img_path)
    mask = mask.astype(np.uint8)
    new_mask = postprocess_mask(mask)
    new_mask = new_mask.astype(np.uint8)
    nifti_img = nib.Nifti1Image(new_mask, affine=np.eye(4))
    save_path = os.path.join(save_dir, save_name)
    nib.save(nifti_img, save_path)


if __name__ == '__main__':

    #nii_img_name = 'downsampled_06618305_PROC_R_MLO_20230421152905.nii.gz'
    nii_img_name = 'downsampled_09759995_PROC_L_MLO_20230910220321.nii.gz'
    path = '/home/kr0pt/Documents/tcc_project/codes/nn_unet/media/output_equalized_dataset_test/'
    
    nii_img_name = nii_img_name
    
    print('Image: ', nii_img_name)
    open_run_and_save_nifti_postprocess(nii_img_name=nii_img_name, nii_img_path=path, save_dir=path+'postprocessed', save_name=f'{nii_img_name}')
