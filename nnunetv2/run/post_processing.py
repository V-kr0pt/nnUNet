import os
import cv2
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
import skimage.measure as measure
from collections import deque
from run_utils import open_nifti_image


def keep_largest_component(mask):
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    if not props:
        return mask
    largest = max(props, key=lambda x: x.area)
    return (labels == largest.label).astype(np.uint8)


def morphological_closing(mask, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed.astype(np.uint8)

def smooth_segmentation_volume(volume):
    smothed = gaussian_filter(volume.astype(np.float32), sigma=1, radius=[10,10,5]) > 0.5
    return smothed.astype(np.uint8)

def l_rool(mask):
    rows, cols = mask.shape
    queue = deque()
    
    # Initialize queue with all active pixels
    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == 1:
                queue.append((i, j))
    while queue:
        i, j = queue.popleft()
        
        # Check neighbor above (i-1, j)
        if i > 0 and mask[i-1, j] == 0:
            mask[i-1, j] = 1
            queue.append((i-1, j))
        
        # Check neighbor to the right (i, j+1)
        if j < cols - 1 and mask[i, j+1] == 0:
            mask[i, j+1] = 1
            queue.append((i, j+1))
    
    return mask.astype(np.uint8)


def contour_smoothing(mask):
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
    total_slices = mask.shape[2]
    for slice_idx in range(total_slices):
        mask[:, :, slice_idx] = keep_largest_component(mask[:, :, slice_idx])
        mask[:, :, slice_idx] = morphological_closing(mask[:, :, slice_idx])
        mask[:, :, slice_idx] = l_rool(mask[:, :, slice_idx])
        #mask[:, :, slice_idx] = contour_smoothing(mask[:, :, slice_idx])
        print(f"Processing slice {slice_idx + 1}/{total_slices}", end='\r')
    
    return mask

def open_run_and_save_nifti_postprocess(nii_img_name, nii_img_path, save_dir, save_name):
    mask = open_nifti_image(nii_img_name=nii_img_name, nii_img_path=nii_img_path)
    mask = postprocess_mask(mask)
    mask = mask.astype(np.uint8)
    nifti_img = nib.Nifti1Image(mask, affine=np.eye(4))
    save_path = os.path.join(save_dir, save_name)
    nib.save(nifti_img, save_path)

    

if __name__ == '__main__':
    
    #nii_img_name = '4186065_PROC_R_MLO_20120831162844'
    nii_img_name = '01483046_PROC_R_MLO_20220918204538'
    #nii_img_name = '3971372_PROC_L_MLO_20120726155832'

    mask = open_nifti_image(nii_img_name=nii_img_name, nii_img_path='output')

    print('Image: ', nii_img_name)
    print(f"Mask shape: {mask.shape}")

    mask = postprocess_mask(mask)

    mask = mask.astype(np.uint8)
    nifti_img = nib.Nifti1Image(mask, affine=np.eye(4))

    output_postprocessed_folder = os.path.join('output', 'postprocessed')
    if not os.path.exists(output_postprocessed_folder):
        os.makedirs(output_postprocessed_folder)
    nifti_file_path = os.path.join(output_postprocessed_folder,f'{nii_img_name}_slice_wise.nii.gz')
    nib.save(nifti_img, nifti_file_path)
    print(f"Postprocessed mask saved at {nifti_file_path}")
