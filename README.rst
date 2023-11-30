nitools
=======

Basic Neuroimaging functionality for Nifti, Gifti, Cifti
Data structures. Basic useful functions missing from nibabel and nilearn

Installation
------------
Over pip, you can install the package, using the command:
pip install neuroimagingtools

Coordinate transforms and nifti (volume) utilities
--------------------------------------------------

* affine_transform: Affine Coordinate transform with individual x,y,z coordinates
* affine_transform_mat: Affine coordinate transform with coordinates in matrix format
* coords_to_linvidxs: Safe transform of coordinates to linear voxel indices
* euclidean_dist_sq: Squared Euclidean distance between coordinate pairs
* sample_img: Sample volume at arbitrary locations with nearest-neighbor or trilinear interpolation
* check_voxel_range: Check of voxel coordinates are within an image

Gifti Utilities
---------------

* make_func_gifti: Make a new functional giftiImage
* make_label_gifti: Make a new label giftiImage (with label table)
* get_gifti_column_names: Extract column names from gifti
* get_gifti_anatomical_struct: Extract Anatomical_structure_primary
* get_gifti_labels: Get label names and colors

Cifti Utilities
---------------
* join_giftis_to_cifti: Joins a left- and right-hemispheric Gifti into a single CIFTI
* split_cifti_to_giftis: Splits CIFTI into a left- and right-hemispheric Gifti
* volume_from_cifti: Extracts Nifti-volume data from a Cifti file
* surf_from_cifti: Extract the surface-based data from a Cifti file

Color utilities
---------------

* read_lut: Read a lookup table file
* save_lut: Save a lookup table file
* save_cmap: Save a FSLeyes colormap file

Border utilities
----------------

* Border: Border class
* Border.get_coords: Get coordinates for a border
* read_borders: Read a workbench border file
* save_borders: Save a workbench border file
* project_border: Project coordinates to a surface
* resample_border: Resample a border with regularly spacing


For documentation, see:
https://nitools.readthedocs.io/en/latest/