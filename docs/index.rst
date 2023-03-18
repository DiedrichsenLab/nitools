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


.. module:: volume

Nifti and Volume functions
--------------------------
.. autofunction:: affine_transform
.. autofunction:: affine_transform_mat
.. autofunction:: coords_to_linvidxs
.. autofunction:: euclidean_dist_sq
.. autofunction:: sample_image
.. autofunction:: check_voxel_range

.. module:: gifti

Gifti and Surface functions
---------------------------
.. autofunction:: make_func_gifti
.. autofunction:: make_label_gifti
.. autofunction:: get_gifti_data_matrix
.. autofunction:: get_gifti_anatomical_struct
.. autofunction:: get_gifti_column_names
.. autofunction:: get_gifti_colortable
.. autofunction:: get_gifti_labels

.. module:: cifti

Cifti and ROI functions
-----------------------
.. automodule:: cifti
    :members:


.. automodule:: other
    :members:





* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
