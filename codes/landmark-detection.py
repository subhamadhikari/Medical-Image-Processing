from skimage.measure import regionprops,label
import numpy as np
import nibabel as nib
from nilearn import plotting
from nilearn import image as nli

def separate_femur_tibia(bone_mask, ct_data):
    """Separate femur and tibia based on position and size"""
    # Label connected components
    labeled_mask, num_labels = label(bone_mask)

    print(f"Found {num_labels} bone segments")

    # Get properties of each segment
    regions = regionprops(labeled_mask)

    # Sort by area (largest first)
    regions = sorted(regions, key=lambda x: x.area, reverse=True)

    if len(regions) < 2:
        print("Warning: Less than 2 bone segments found!")
        return None, None

    # Usually femur is larger and higher, tibia is smaller and lower
    femur_candidates = []
    tibia_candidates = []

    for i, region in enumerate(regions[:4]):  # Check top 4 largest
        centroid = region.centroid
        area = region.area

        print(f"Segment {i+1}: Area={area}, Centroid={centroid}")

        # Generally, femur is higher (lower Z index) and tibia is lower (higher Z index)
        if centroid[2] < ct_data.shape[2] * 0.4:  # Upper part
            femur_candidates.append((i, region))
        else:  # Lower part
            tibia_candidates.append((i, region))

    # Select the largest from each group
    femur_region = max(femur_candidates, key=lambda x: x[1].area)[1] if femur_candidates else regions[0]
    tibia_region = max(tibia_candidates, key=lambda x: x[1].area)[1] if tibia_candidates else regions[1]

    femur_mask = labeled_mask == femur_region.label
    tibia_mask = labeled_mask == tibia_region.label

    return femur_mask, tibia_mask

def find_tibial_plateau_landmarks(tibia_mask, ct_img, method='lowest_points'):

  tibia_coords = np.array(np.where(tibia_mask)).T  # [i, j, k] voxel coordinates


  world_coords = nib.affines.apply_affine(ct_img.affine, tibia_coords)

        # Get upper 20% of tibia points
  z_threshold = np.percentile(world_coords[:, 2], 80)
  plateau_points = world_coords[world_coords[:, 2] >= z_threshold]

        # From plateau points, get the lowest ones
  plateau_z_min = np.percentile(plateau_points[:, 2], 10)
  lowest_points = plateau_points[plateau_points[:, 2] <= plateau_z_min]

  print(f"Plateau Z threshold: {z_threshold:.1f}")
  print(f"Found {len(lowest_points)} plateau points")

  x_median = np.median(lowest_points[:, 0])

  medial_points = lowest_points[lowest_points[:, 0] < x_median]
  lateral_points = lowest_points[lowest_points[:, 0] >= x_median]

  print(f"Medial points: {len(medial_points)}, Lateral points: {len(lateral_points)}")

  if len(medial_points) == 0 or len(lateral_points) == 0:
    print("Could not separate into medial/lateral groups!")
    return None, None

    # Find the most posterior point in each group (assuming posterior = lower Y)
  medial_idx = np.argmin(medial_points[:, 1])
  lateral_idx = np.argmin(lateral_points[:, 1])

  medial_landmark = medial_points[medial_idx]
  lateral_landmark = lateral_points[lateral_idx]

  return medial_landmark, lateral_landmark
