import nibabel as nib
from nilearn import plotting
from nilearn import image as nli
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import measure

from nibabel import load, save, Nifti1Image

from scipy.ndimage import binary_dilation, generate_binary_structure, label, morphology,  binary_closing, binary_opening
from skimage.measure import regionprops


def create_bone_mask(ct_img, bone_threshold=320):

    ct_data = ct_img.get_fdata()
    bone_mask = ct_data > bone_threshold
    bone_img = nib.Nifti1Image(bone_mask.astype(np.int16), ct_img.affine)
    return bone_img

def expand_mask(bone_img, ct_img, radius_mm=2.0):
    voxel_spacing = bone_img.header.get_zooms()[:3]
    radius_voxels = [int(np.ceil(radius_mm / vs)) for vs in voxel_spacing]

    x = np.arange(-radius_voxels[0], radius_voxels[0] + 1)
    y = np.arange(-radius_voxels[1], radius_voxels[1] + 1)
    z = np.arange(-radius_voxels[2], radius_voxels[2] + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    distances = np.sqrt((xx * voxel_spacing[0])**2 +
                        (yy * voxel_spacing[1])**2 +
                        (zz * voxel_spacing[2])**2)

    structuring_element = distances <= radius_mm

    data = np.asarray(bone_img.dataobj)
    dilated_data = morphology.binary_dilation(data, structure=structuring_element)

    expanded_img = nib.Nifti1Image(dilated_data.astype(np.uint8), affine=bone_img.affine, header=bone_img.header)
    plotting.plot_roi(expanded_img, bg_img=ct_img, cut_coords=(-105, -9, -688))

    return expanded_img

def generate_randomized_expansion_mask(knee_ct_scan, expansion_mm=2.0, bone_threshold=300, random_fraction=0.4, output_path="/content/knee_bone_randomized_adjusted.nii.gz"):
    # Load the CT image
    ct_img = nib.load(knee_ct_scan)
    ct_data = ct_img.get_fdata()
    voxel_spacing = ct_img.header.get_zooms()[:3]  # (x, y, z) spacing in mm

    # Calculate expansion in voxels
    expansion_voxels = [int(np.ceil(expansion_mm / vs)) for vs in voxel_spacing]

    # Create coordinate grids
    x = np.arange(-expansion_voxels[0], expansion_voxels[0] + 1)
    y = np.arange(-expansion_voxels[1], expansion_voxels[1] + 1)
    z = np.arange(-expansion_voxels[2], expansion_voxels[2] + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Calculate distances from the center in millimeters
    distances = np.sqrt((xx * voxel_spacing[0])**2 +
                        (yy * voxel_spacing[1])**2 +
                        (zz * voxel_spacing[2])**2)

    # Define the structuring element
    structuring_element = distances <= expansion_mm

    # Threshold the CT data to create the bone mask
    bone_mask = ct_data > bone_threshold

    # Apply binary dilation
    dilated_mask = binary_dilation(bone_mask, structure=structuring_element)

    # Get the expansion region (dilated minus original)
    expansion_region = np.logical_and(dilated_mask, np.logical_not(bone_mask))

    # Generate a random mask within the expansion region
    random_mask = np.random.rand(*ct_data.shape) < random_fraction

    # Combine original mask and randomized expansion
    randomized_expansion = np.logical_and(expansion_region, random_mask)
    adjusted_mask = np.logical_or(bone_mask, randomized_expansion)

    # Create a new Nifti image and save it
    adjusted_img = nib.Nifti1Image(adjusted_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(adjusted_img, output_path)

    return adjusted_img


randomized_1 = generate_randomized_expansion_mask(knee_ct_scan, expansion_mm=2.0, bone_threshold=300, random_fraction=0.4,output_path="randomized-1.nii.gz")

