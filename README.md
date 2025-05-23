#  3D Knee CT Bone Segmentation and Landmark Detection

This project performs bone segmentation on 3D knee CT image volumes, followed by contour expansion, randomized contour adjustment, and landmark detection on the tibial surface. It was developed as part of an assessment project for medical image processing.

# Tasks Overview
-  Task 1.1 – Bone Segmentation
Segmented femur and tibia from the CT scan using intensity thresholding and morphological operations.

-  Task 1.2 – Contour Expansion
Expanded the segmented mask by a specified distance (2 mm by default) using a spherical structuring element.

- Task 1.3 – Randomized Contour Adjustment
Randomly adjusted the contour between the original and expanded masks while keeping it within a 2 mm expansion limit.

- Task 1.4 – Landmark Detection on Tibia
Detected medial and lateral lowest points on the tibial plateau surface and saved their coordinates for evaluation.

All the codes are in the notebook, you can download and view the results, as results may not be visible when viewed directly in github.