.. nitools_index

Nitools
=======
Basic Neuroimaging functionality for Nifti, Gifti, Cifti
Data structures. Basic useful functions missing from nibabel and nilearn

Developed and maintained by the Diedrichsenlab.

Installation
------------
For developement, please clone the Github repository:
https://github.com/DiedrichsenLab/nitools

To simply use it, you can install the package with pip:

``pip install neuroimagingtools``


.. module:: nitools

Nifti and Volume functions
--------------------------
.. autofunction:: affine_transform
.. autofunction:: affine_transform_mat
.. autofunction:: coords_to_linvidxs
.. autofunction:: euclidean_dist_sq
.. autofunction:: sample_image
.. autofunction:: check_voxel_range

Gifti and Surface functions
---------------------------
.. autofunction:: make_func_gifti
.. autofunction:: make_label_gifti
.. autofunction:: get_gifti_data_matrix
.. autofunction:: get_gifti_anatomical_struct
.. autofunction:: get_gifti_column_names
.. autofunction:: get_gifti_colortable
.. autofunction:: get_gifti_labels

Cifti and ROI functions
-----------------------
.. autofunction:: join_giftis
.. autofunction:: volume_from_cifti
.. autofunction:: surf_from_cifti





* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
