import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def open_nifti_image(nii_img_name:str, nii_img_path:str=None):
    """
    Open a nifti image and return the data.
    """
    if nii_img_path is None:
        nii_img_path = os.path.join('input', f'{nii_img_name}')
    else:
        nii_img_path = os.path.join(nii_img_path, f'{nii_img_name}')
    nii_img  = nib.load(nii_img_path)
    nii_data = nii_img.get_fdata()
    return nii_data

def upsample_nii_file(nii_file:str, upsample_factor:tuple, save_dir:str, save_name:str=None):
    nii = nib.load(nii_file)
    nii_data = nii.get_fdata()

    assert len(upsample_factor) == 3, "Factor must be a tuple of 3 values."
    assert upsample_factor[0] > 1 and upsample_factor[1] > 1, "Factors must be greater than 1."

    upsampled_data = zoom(nii_data, upsample_factor, order=1)
    upsampled_data = (upsampled_data > 0.5).astype(np.uint8)  # Being sure the data is binary
    
    upsampled_nii = nib.Nifti1Image(upsampled_data, nii.affine, nii.header)
    
    if save_name:
        output_path = os.path.join(save_dir, save_name)
    else:
        output_path = os.path.join(save_dir, os.path.basename(nii_file))
        
    nib.save(upsampled_nii, output_path)


def downsample_nii_file(nii_file:str, downsample_factor:int, save_dir:str, save_name:str=None):
    nii = nib.load(nii_file)
    nii_data = nii.get_fdata()
    #downsampled_data = nii_data[::downsample_factor, ::downsample_factor, :]
    zoom_factors = [1 / downsample_factor, 1 / downsample_factor, 1]
    downsampled_data = zoom(nii_data, zoom_factors, order=1)
    
    downsampled_nii = nib.Nifti1Image(downsampled_data, nii.affine, nii.header)
    
    if save_name:
        output_path = os.path.join(save_dir, save_name)
    else:
        output_path = os.path.join(save_dir, os.path.basename(nii_file))
        
    nib.save(downsampled_nii, output_path)

    real_factor = (downsampled_data.shape[0] / nii_data.shape[0], 
                   downsampled_data.shape[1] / nii_data.shape[1], 
                   1)
    return real_factor

def flip_nii_file(nii_file:str):
    nii = nib.load(nii_file)
    nii_data = nii.get_fdata()
    flipped_data = np.flip(nii_data, axis=0)  # Flip along the first axis (x-axis)
    
    flipped_nii = nib.Nifti1Image(flipped_data, nii.affine, nii.header)
    
    # overwrite the original file
    output_path = os.path.join(os.path.dirname(nii_file), os.path.basename(nii_file))
    nib.save(flipped_nii, output_path)