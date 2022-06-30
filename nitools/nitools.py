"""Nitools
"""
import numpy as np
import nibabel as nb
import matplotlib as mpl
import matplotlib.pyplot as plt

def affine_transform(x1, x2, x3, M):
    """
    Returns affine transform of x

    Args:
        x1 (np-array):
            X-coordinate of original (any size)
        x2 (np-array):
            Y-coordinate of original
        x3 (np-array):
            Z-coordinate of original
        M (2d-array):
            4x4 transformation matrix

    Returns:
        x1 (np-array):
            X-coordinate of transform
        x2 (np-array):
            Y-coordinate of transform
        x3 (np-array):
            Z-coordinate of transform
    """
    y1 = M[0,0]*x1 + M[0,1]*x2 + M[0,2]*x3 + M[0,3]
    y2 = M[1,0]*x1 + M[1,1]*x2 + M[1,2]*x3 + M[1,3]
    y3 = M[2,0]*x1 + M[2,1]*x2 + M[2,2]*x3 + M[2,3]
    return (y1,y2,y3)

def affine_transform_mat(x, M):
    """
    Returns affine transform of x

    Args:
        x (np-array):
            3xN array for original coordinates
        M (2d-array):
            4x4 transformation matrix
    Returns:
        y (np-array):
            3xN array pof X-coordinate of transformed coordinaters
    """
    y = M[0:3,0:3] @ x + M[0:3,3:]
    return (y)

def coords_to_linvidxs(coords,vol_def,mask=False):
    """
    Maps coordinates to linear voxel indices

    Args:
        coords (3xN matrix or Qx3xN array):
            (x,y,z) coordinates
        vol_def (nibabel object):
            Nibabel object with attributes .affine (4x4 voxel to coordinate transformation matrix from the images to be sampled (1-based)) and shape (1x3 volume dimension in voxels)
        mask (bool):
            If true, uses the mask image to restrict voxels (all outside = -1)
    Returns:
        linvidxs (N-array or QxN matrix):
            Linear voxel indices
        good (bool):
            boolean array that tells you whether the index was in the mask
    """
    mat = np.linalg.inv(vol_def.affine)

    # Check that coordinate transformation matrix is 4x4
    if (mat.shape != (4,4)):
        raise(NameError('Error: Matrix should be 4x4'))

    rs = coords.shape
    if (rs[-2] != 3):
        raise(NameError('Coords need to be a (Kx) 3xN matrix'))

    # map to 3xP matrix (P coordinates)
    ijk = mat[0:3,0:3] @ coords + mat[0:3,3:]
    ijk = np.rint(ijk).astype(int)

    if ijk.ndim<=2:
        i = ijk[0]
        j = ijk[1]
        k = ijk[2]
    elif ijk.ndim==3:
        i = ijk[:,0]
        j = ijk[:,1]
        k = ijk[:,2]

    # Now set the indices out of range to
    good = (i>=0) & (i<vol_def.shape[0]) & (j>=0) & (j<vol_def.shape[1]) &  (k[2]>=0) & (k[2]<vol_def.shape[2])

    linindx = np.ravel_multi_index((i,j,k),vol_def.shape,mode='clip')

    if mask:
        M=vol_def.get_fdata().ravel()
        good = good & (M[linindx]>0)

    return linindx, good


def euclidean_dist_sq(coordA,coordB):
    """Computes full matrix of square
    Euclidean  distance between sets of coordinates

    Args:
        coordA (ndarray):
            3xP matrix of coordinates
        coordB (ndarray):
            3xQ matrix of coordinats

    Returns:
        Dist (ndarray):
            PxQ matrix of squared distances
    """
    D = coordA.reshape(3,-1,1)-coordB.reshape(3,1,-1)
    D = np.sum(D**2,axis=0)
    return D

def sample_image(img,xm,ym,zm,interpolation):
    """
    Return values after resample image

    Args:
        img (Nifti image):
            Input Nifti Image
        xm (np-array):
            X-coordinate in world coordinates
        ym (np-array):
            Y-coordinate in world coordinates
        zm (np-array):
            Z-coordinate in world coordinates
        interpolation (int):
            0: Nearest neighbor
            1: Trilinear interpolation
    Returns:
        value (np-array):
            Array contains all values in the image
    """
    im,jm,km = affine_transform(xm,ym,zm,np.linalg.inv(img.affine))

    if interpolation == 1:
        ir = np.floor(im).astype('int')
        jr = np.floor(jm).astype('int')
        kr = np.floor(km).astype('int')

        invalid = np.logical_not((im>=0) & (im<img.shape[0]-1) & (jm>=0) & (jm<img.shape[1]-1) & (km>=0) & (km<img.shape[2]-1))
        ir[invalid] = 0
        jr[invalid] = 0
        kr[invalid] = 0

        id = im - ir
        jd = jm - jr
        kd = km - kr

        D = img.get_fdata()
        if D.ndim == 4:
            ns = id.shape + (1,)
        if D.ndim ==5:
            ns = id.shape + (1,1)
        else:
            ns = id.shape

        id = id.reshape(ns)
        jd = jd.reshape(ns)
        kd = kd.reshape(ns)

        c000 = D[ir, jr, kr]
        c100 = D[ir+1, jr, kr]
        c110 = D[ir+1, jr+1, kr]
        c101 = D[ir+1, jr, kr+1]
        c111 = D[ir+1, jr+1, kr+1]
        c010 = D[ir, jr+1, kr]
        c011 = D[ir, jr+1, kr+1]
        c001 = D[ir, jr, kr+1]

        c00 = c000*(1-id)+c100*id
        c01 = c001*(1-id)+c101*id
        c10 = c010*(1-id)+c110*id
        c11 = c011*(1-id)+c111*id

        c0 = c00*(1-jd)+c10*jd
        c1 = c01*(1-jd)+c11*jd

        value = c0*(1-kd)+c1*kd
    elif interpolation == 0:
        ir = np.rint(im).astype('int')
        jr = np.rint(jm).astype('int')
        kr = np.rint(km).astype('int')

        invalid = check_voxel_range(img, ir, jr, kr)
        ir[invalid] = 0
        jr[invalid] = 0
        kr[invalid] = 0
        value = img.get_fdata()[ir, jr, kr]

    # Kill the invalid elements
    if value.dtype is float:
        value[invalid]=np.nan
    else:
        value[invalid]=0
    return value

def check_voxel_range(img,i,j,k):
    """
    Checks if i,j,k voxels coordinates are within an image

    Args:
        img (niftiImage or ndarray):
            needs shape to determine voxels ranges
        i (np.array):
            all i-coordinates
        j (np.array):
            all j-coordinates
        k (np.array):
            all k-coordinates
    Returns:
        invalid (nd.array):
            boolean array indicating whether i,j,k coordinates are invalid
    """
    invalid = np.logical_not((i>=0) & (i<img.shape[0]) &
                           (j>=0) & (j<img.shape[1]) &
                           (k>=0) & (k<img.shape[2]))
    return invalid

def make_func_gifti(
    data,
    anatomical_struct='Cerebellum',
    column_names=[]
    ):
    """Generates a function GiftiImage from a numpy array

    Args:
        data (np.array):
             num_vert x num_col data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'Cerebellum'
        column_names (list):
            List of strings for names for columns

    Returns:
        FuncGifti (GiftiImage): functional Gifti Image
    """
    num_verts, num_cols = data.shape
    #
    # Make columnNames if empty
    if len(column_names)==0:
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
                    label_names=[],
                    column_names=[],
                    label_RGBA=[]
                    ):
    """Generates a label GiftiImage from a numpy array

    Args:
        data (np.array):
             num_vert x num_col data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'Cerebellum'
        label_names (list):
            List of strings for label names
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors

    Returns:
        gifti (GiftiImage): Label gifti image

    """
    num_verts, num_cols = data.shape
    num_labels = len(np.unique(data))

    # check for 0 labels
    zero_label = 0 in data

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if len(column_names) == 0:
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if len(label_RGBA) == 0:
        hsv = plt.cm.get_cmap('hsv',num_labels)
        color = hsv(np.linspace(0,1,num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        label_RGBA = np.zeros([num_labels,4])
        for i in range(num_labels):
            label_RGBA[i] = color[i]
        if zero_label:
            label_RGBA = np.vstack([[0,0,0,1], label_RGBA[1:,]])

    # Create label names
    if len(label_names) == 0:
        idx = 0
        if not zero_label:
            idx = 1
        for i in range(num_labels):
            label_names.append("label-{:02d}".format(i + idx))

    # Create label.gii structure
    C = nb.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})

    num_labels = np.arange(num_labels)
    E_all = []
    for (label, rgba, name) in zip(num_labels, label_RGBA, label_names):
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
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL',
            datatype='NIFTI_TYPE_UINT8',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.extend(E_all)
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

def join_giftis(giftis,mask=[None,None],seperate_labels=False,join_zero=False):
    """ Combines a left and right hemispheric Gifti file into a single Cifti
    file that contains both hemisphere

    Args:
        giftis (list):
            List of 2 Gifti images or list of 2 file names to be merged:
            gives warning if not left and right hemisphere
        mask (list, optional):
            Mask for each hemisphere (only vertices within mask will be used).
            Defaults to [None,None].
            Can be set to list of giftis, filesnames, or nd-arrays
        seperate_labels (bool, optional):
            False (default): Simple merges the two files
            True: Offsets the label for the right hemisphere, and prepend a L/R to label
        join_zero (bool, optional):
            If set to true, the zero-label will be joined for the two hemispheres, resulting
            label less overall. (default = False)

    Returns:
        cifti_img: Cifti-image
    """
    hem_name_bm = ['cortex_left','cortex_right']
    hem_name_gifti = ['CortexLeft','CortexRight']
    hem = ['L','R']
    GIF = []

    # Load the giftis
    for i,g in enumerate(giftis):
        if type(g) is str:
            GIF.append(nb.load(g))
        elif type(g) is nb.GiftiImage:
            GIF.append(g)
        else:
            raise(NameError('Giftis need to be filenames or Gifti (filenames '))

    bm = []
    data = []
    for h in range(2):
        # Check if intent and names across all columns
        intent = []
        names = []
        for i,d in enumerate(GIF[h].darrays):
            intent.append(d.intent)
            for md in d.meta.data:
                if 'Name' in md.name:
                    names.append(md.value)

        # Check if Giftis are denoting left and right hemisphere
        if get_gifti_anatomical_struct(GIF[h])!=hem_name_gifti[h]:
            raise(NameError('Giftis should have anatomical structure CortexLeft/CortexRight'))
        # Get the data and make the brain model axis
        data.append(np.c_[GIF[h].agg_data()])
        if mask[h] is None:
            bm.append(nb.cifti2.BrainModelAxis.from_mask(np.ones((data[h].shape[0],)),hem_name_bm[h]))
        else:
            raise(NameError('Mask not implemented yet'))

    # Label axis:
    if intent[0]==1002:
        new_labels = {}
        # If not seperate labels, use the labels for the left hemisphere
        if seperate_labels is False:
            labels = GIF[0].labeltable.labels
            for i,l in enumerate(labels):
                new_labels[i]=(l.label,l.rgba)
        # If seperate labels, concat the labels from L and R
        # If keep_zero is True uses the same zero label for the two hems
        else:
            max_label = 0
            for h in range(2):
                if join_zero:
                    data[h]+=max_label
                    data[h][data[h]==max_label]=0
                    for i,l in enumerate(GIF[h].labeltable.labels[h:]):
                        new_labels[i+max_label+h]=(f"{hem[h]}-{l.label}",l.rgba)
                    max_label  = data[h].max()
                else:
                    data[h]+=max_label
                    for i,l in enumerate(GIF[h].labeltable.labels):
                        new_labels[i+max_label]=(f"{hem[h]}-{l.label}",l.rgba)
                    max_label  = data[h].max()+1
        D = np.concatenate(data,axis=0).T
        row_axis=nb.cifti2.LabelAxis(names, new_labels)
    # Scalar Axis:
    else:
        row_axis = nb.cifti2.ScalarAxis(names)
        D = np.concatenate(data,axis=0).T

    # Finalize CIFTI object
    header = nb.Cifti2Header.from_axes((row_axis,bm[0]+bm[1]))
    cifti_img = nb.Cifti2Image(dataobj=D,header=header)
    return cifti_img

def volume_from_cifti(ts_cifti, struct_names=None):
        """ Gets the 4D nifti object containing the time series
        for all subcortical (volume-based) structures

        Args:
            ts_cifti (ciftiImage):
                cifti object of the time series
            struct_names (list or None):
                List of structure names that are included
                defaults to None
        Returns:
            nii_vol(niftiImage):
                nifti object containing the time series
        """
        # get brain axis models
        bmf = ts_cifti.header.get_axis(1)
        # get the data array with all the time points, all the structures
        ts_array = ts_cifti.get_fdata(dtype=np.float32)

        # initialize a matrix representing 4D data (x, y, z, time_point)
        subcorticals_vol = np.zeros(bmf.volume_shape + (ts_array.shape[0],))
        for idx, (nam,slc,bm) in enumerate(bmf.iter_structures()):

            # if (struct_names is None) | (nam in struct_names):

            # get the voxels/vertices corresponding to the current brain model
            ijk = bm.voxel
            bm_data = ts_array[:, slc]
            i  = (ijk[:,0] > -1)

            # fill in data
            subcorticals_vol[ijk[i, 0], ijk[i, 1], ijk[i, 2], :]=bm_data[:,i].T

        # save as nii
        nii_vol_4d = nb.Nifti1Image(subcorticals_vol,bmf.affine)
        return nii_vol_4d

def surf_from_cifti(cifti,
                    struct_names=['CIFTI_STRUCTURE_CORTEX_LEFT',
                                    'CIFTI_STRUCTURE_CORTEX_RIGHT']):
        """ Gets the data for cortical surface vertices (Left and Right)

        Args:
            cifti (cifti2Image):
                Input cifti that contains surface data
            struct_names (list):
                Names of anatomical structures (in cifti format).
                Defaults to left and right hemisphere
        Returns:
            list of ndarrays:
                Data for all surface, with data x numVert for each surface
        """
        # get brain axis models
        bmf = cifti.header.get_axis(1)
        # print(dir(bmf))
        # get the data array with all the time points, all the structures
        ts_array = cifti.get_fdata()
        ts_list = []
        for idx, (nam,slc,bm) in enumerate(bmf.iter_structures()):
            # just get the cortical surfaces
            if nam in struct_names:
                values = np.full((ts_array.shape[0],bmf.nvertices[nam]),np.nan)
                # get the values corresponding to the brain model
                values[:,bm.vertex] = ts_array[:, slc]
                ts_list.append(values)
            else:
                break
        return ts_list
