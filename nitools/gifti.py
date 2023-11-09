"""Nitools Gifti functions
"""
import numpy as np
import nibabel as nb
import matplotlib as mpl
import matplotlib.pyplot as plt

def make_func_gifti(
    data,
    anatomical_struct='Cerebellum',
    column_names=None
    ):
    """Generates a function GiftiImage from a numpy array

    Args:
        data (np.array):
             num_vert x num_col data
        anatomical_struct (string):
            Anatomical Structure default= 'Cerebellum' 
        column_names (list):
            List of strings for names for columns

    Returns:
        FuncGifti (GiftiImage): functional Gifti Image
    """
    
    # Get data dimensions
    if data.ndim == 1:
        data = data.reshape(-1,1)
    num_verts, num_cols = data.shape

    num_verts, num_cols = data.shape
    #
    # Make columnNames if empty
    if column_names is None:
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    C = nb.gifti.GiftiMetaData.from_dict({
    'AnatomicalStructurePrimary': anatomical_struct,
    'encoding': 'XML_BASE64_GZIP'})

    E = nb.gifti.gifti.GiftiLabel()
    E.key = 0
    E.label= '???'
    E.red = 1.0
    E.green = 1.0
    E.blue = 1.0
    E.alpha = 0.0

    D = list()
    for i in range(num_cols):
        d = nb.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_NONE',
            datatype='NIFTI_TYPE_FLOAT32',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.append(E)

    return gifti

def make_label_gifti(
                    data,
                    anatomical_struct='Cerebellum',
                    labels=None,
                    label_names=None,
                    column_names=None,
                    label_RGBA=None
                    ):
    """Generates a label GiftiImage from a numpy array

    Args:
        data (np.array):
             num_vert x num_col data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data. default: 'Cerebellum'
        labels (list):
            Numerical values in data indicating the labels. default: np.unique(data)
        label_names (list):
            List of strings for names for labels
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors for labels
    Returns:
        gifti (GiftiImage): Label gifti image

    """
    if labels is not None:
        if label_names is not None:
            assert len(labels) == len(label_names), "labels and label_names must be the same length"
        if column_names is not None:
            assert len(labels) == len(column_names), "labels and column_names must be the same length"
        if label_RGBA is not None:
            assert len(labels) == len(label_RGBA), "labels and label_RGBA must be the same length"

    # Get data dimensions
    if data.ndim == 1:
        data = data.reshape(-1,1)
    num_verts, num_cols = data.shape
    data = data.astype(int)
    
    # If labels not given 
    if labels is None:
        labels = np.unique(data)
    num_labels = len(labels)

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if label_RGBA is None:
        label_RGBA = np.zeros([num_labels,4])
        hsv = plt.cm.get_cmap('hsv',num_labels)
        color = hsv(np.linspace(0,1,num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        for i in range(num_labels):
            label_RGBA[i] = color[i]

    # Create label names from numerical values
    if label_names is None:
        label_names = []
        for i in labels:
            label_names.append("label-{:02d}".format(int(i)))

    # Create label.gii structure
    C = nb.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})

    E_all = []
    for (label, rgba, name) in zip(labels, label_RGBA, label_names):
        E = nb.gifti.gifti.GiftiLabel()
        E.key = label
        E.label= name
        E.red = rgba[0]
        E.green = rgba[1]
        E.blue = rgba[2]
        E.alpha = rgba[3]
        E.rgba = rgba[:]
        E_all.append(E)

    D = list()
    for i in range(num_cols):
        d = nb.gifti.GiftiDataArray(
            data=np.uint8(data[:, i]),
            intent='NIFTI_INTENT_LABEL',
            datatype='NIFTI_TYPE_UINT8',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.extend(E_all)
    return gifti

def make_surf_gifti(vertices, faces, normals, mat,
                    anatomical_struct='CortexLeft'):
    """ Generates a surface GiftiImage from a surface

    Args:
        vertices (np.array): num_vert x 3 array of vertices
        faces (np.array): num_faces x 3 array of faces
        normals (np.array): num_vert x 3 array of normals
        mat (np.array): a 4x4 affine matrix
        anatomical_struct: Anatomical Structure for the Meta-data

    Returns:
        gifti (GiftiImage): Surface gifti image

    Examples:
        1. Load `vertices` and `faces` from a surface file
            surf = nb.load('sub-01.L.pial.32k.surf.gii')
            vertices = surf.agg_data('NIFTI_INTENT_POINTSET')
            faces = surf.agg_data('NIFTI_INTENT_TRIANGLE')

        2.1 Load normals (from surf file directly if available)
            normals = surf.agg_data('NIFTI_INTENT_NORMAL')

        2.2 Load normals (from a separate file)
            norm_file = 'sub-01.L.norm.32k.shape.gii'
            normals = norm_file.agg_data()

        3. Input `mat` is a 4x4 affine matrix, e.g.:
            mat = np.eye(4)
    """
    data_list = [vertices, faces, normals, mat]
    name = ['vertices', 'faces', 'normals', 'mat']

    C = nb.gifti.GiftiMetaData.from_dict({
    'AnatomicalStructurePrimary': anatomical_struct,
    'encoding': 'XML_BASE64_GZIP'})

    V = nb.gifti.GiftiDataArray(
        data=vertices,
        intent='NIFTI_INTENT_POINTSET',
        datatype='NIFTI_TYPE_FLOAT32',
        meta=nb.gifti.GiftiMetaData.from_dict({'Name': 'vertices'})
    )

    F = nb.gifti.GiftiDataArray(
        data=faces,
        intent='NIFTI_INTENT_TRIANGLE',
        datatype='NIFTI_TYPE_FLOAT32',
        meta=nb.gifti.GiftiMetaData.from_dict({'Name': 'faces'})
    )

    N = nb.gifti.GiftiDataArray(
        data=normals,
        intent='NIFTI_INTENT_VECTOR',
        datatype='NIFTI_TYPE_FLOAT32',
        meta=nb.gifti.GiftiMetaData.from_dict({'Name': 'normals'})
    )

    # M = nb.gifti.GiftiDataArray(
    #     data=mat,
    #     intent='NIFTI_INTENT_GENMATRIX',
    #     datatype='NIFTI_TYPE_FLOAT32',
    #     meta=nb.gifti.GiftiMetaData.from_dict({'Name': 'mat'})
    # )

    D = list()
    D.append(V)
    D.append(F)
    D.append(N)
    # D.append(M)

    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    return gifti

def get_gifti_data_matrix(gifti):
    """Returns the data matrix contained in a GiftiImage.

    Args:
        gifti (giftiImage):
            Nibabel Gifti image

    Returns:
        data (ndarray):
            Concatinated data from the Gifti file
    """
    return np.c_[gifti.agg_data()]


def get_gifti_column_names(gifti):
    """Returns the column names from a GiftiImage

    Args:
        gifti (gifti image):
            Nibabel Gifti image
    Returns:
        names (list):
            List of column names from gifti object attribute data arrays
    """
    N = len(gifti.darrays)
    names = []
    for n in range(N):
        for i in range(len(gifti.darrays[n].meta.data)):
            if 'Name' in gifti.darrays[n].meta.data[i].name:
                names.append(gifti.darrays[n].meta.data[i].value)
    return names

def get_gifti_colortable(gifti,ignore_zero=False):
    """Returns the RGBA color table and matplotlib cmap from gifti object

    Args:
        gifti (gifti image):
            Nibabel GiftiImage
        ignore_zero (bool):
            Skip the color corresponding to label 0? Defaults to False
    Returns:
        rgba (np.ndarray):
            N x 4 of RGB values
        cmap (mpl obj):
            matplotlib colormap

    """
    labels = gifti.labeltable.labels

    rgba = np.zeros((len(labels),4))
    for i,label in enumerate(labels):
        rgba[i,] = labels[i].rgba

    if ignore_zero:
        rgba = rgba[1:]
        labels = labels[1:]

    cmap = mpl.colors.LinearSegmentedColormap.from_list('mylist', rgba, N=len(rgba))
    mpl.cm.unregister_cmap("mycolormap")
    mpl.cm.register_cmap("mycolormap", cmap)

    return rgba, cmap

def get_gifti_anatomical_struct(gifti):
    """ Returns the primary anatomical structure for a GiftiImage

    Args:
        gifti (gifti image):
            Nibabel Gifti image

    Returns:
        anatStruct (string):
            AnatomicalStructurePrimary attribute from gifti object
    """
    N = len(gifti._meta.data)
    for i in range(N):
        if 'AnatomicalStructurePrimary' in gifti._meta.data[i].name:
            anatStruct = gifti._meta.data[i].value
    return anatStruct

def get_gifti_labels(gifti):
    """ Returns labels from gifti Image

    Args:
        gifti (gifti image):
            Nibabel Gifti image
    Returns:
        labels (list):
            labels from gifti object
    """
    label_dict = gifti.labeltable.get_labels_as_dict()
    labels = list(label_dict.values())
    return labels
