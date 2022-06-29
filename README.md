# nitools
 Basic Neuroimaging functionality for Nifti, Gifti, Cifti
 Data structures. Basic useful functions missing from nibabel and nilearn

Coordinate transforms and nifti (volume) utilities
* affine_transform: Affine Coordinate transform with individual x,y,z coordinates
* affine_transform_mat: Affine coordinate transform with coordinates in matrix format
* coords_to_linvidxs: Safe transform of coordinates to linear voxel indices
* euclidean_dist_sq: Squared Euclidean distance between coordinate pairs
* sample_img: Sample volume at arbitrary locations with nearest-neighbor or trilinear interpolation
* check_voxel_range: Check of voxel coordinates are within an image

Gifti Utilities
* make_func_gifti: Make a new functional giftiImage
* make_label_gifti: Make a new label giftiImage (with label table)
* get_gifti_column_names: Extract column names from gifti
* get_gifti_anatomical_struct: Extract Anatomical_structure_primary
* get_gifti_labels: Get label names and colors

Cifti Utilities
* join_giftis: Joins a left- and right-hemispheric Gifti into a single CIFTI
* volume_from_cifti: Extracts Nifti-volume data from a Cifti file
* surf_from_cifti: Extract the surface-based data from a Cifti file
