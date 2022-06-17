"""Nitools
"""
def affine_transform(x1, x2, x3, M):
    """
    Returns affine transform of x

    Args:
        x1 (np-array):
            X-coordinate of original
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

    INPUT:
        coords (3xN matrix or Qx3xN array):
            (x,y,z) coordinates
        vol_def (nibabel object):
            Nibabel object with attributes .affine (4x4 voxel to coordinate transformation matrix from the images to be sampled (1-based)) and shape (1x3 volume dimension in voxels)
        mask (bool):
            If true, uses the mask image to restrict voxels (all outside = -1)
    OUTPUT:
        linvidxs (N-array or QxN matrix):
            Linear voxel indices
        good (bool) boolean array that tells you whether the index was in the mask
    """
    mat = inv(vol_def.affine)

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

def sq_eucl_distances(coordA,coordB): 
    D = coordA.reshape(3,-1,1)-coordB.reshape(3,1,-1)
    D = np.sum(D**2,axis=0)
    return D


def volume_from_cifti(ts_cifti, struct_names=None):
        """
        Gets the 4D nifti object containing the time series
        for all the subcortical structures
        Args:
            ts_cifti (cifti obj ) - cifti object of the time series
        Returns:
            nii_vol(nifti vol object) - nifti object containing the time series of subcorticals
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
        # if save:
        #     ts_nifti = dir+'/sub-100307_ses-01_task-rest_space-subcortex_run-01_bold.nii'
        #     nb.save(nii_vol,ts_nifti)
        return nii_vol_4d

def surf_from_cifti(ts_cifti,
                    struct_names=['CIFTI_STRUCTURE_CORTEX_LEFT',
                                    'CIFTI_STRUCTURE_CORTEX_RIGHT']):
        """
        Gets the time series of cortical surface vertices (Left and Right)
        Args:
            ts_cifti (cifti obj) - cifti object of time series
        Returns:
            cii (cifti object) - contains the time series for the cortex
        """
        # get brain axis models
        bmf = ts_cifti.header.get_axis(1)
        # print(dir(bmf))
        # get the data array with all the time points, all the structures
        ts_array = ts_cifti.get_fdata()
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



def affine_transform(x1,x2,x3,M):
    """
    Returns affine transform of x

    Args:
        x1 (np-array):
            X-coordinate of original
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
    y1 = np.multiply(M[0,0],x1) + np.multiply(M[0,1],x2) + np.multiply(M[0,2],x3) + M[0,3]
    y2 = np.multiply(M[1,0],x1) + np.multiply(M[1,1],x2) + np.multiply(M[1,2],x3) + M[1,3]
    y3 = np.multiply(M[2,0],x1) + np.multiply(M[2,1],x2) + np.multiply(M[2,2],x3) + M[2,3]
    return (y1,y2,y3)

def coords_to_voxelidxs(coords,vol_def):
    """
    Maps coordinates to linear voxel indices

    Args:
        coords (3*N matrix or 3xPxQ array):
            (x,y,z) coordinates
        vol_def (nibabel object):
            Nibabel object with attributes .affine (4x4 voxel to coordinate transformation matrix from the images to be sampled (1-based)) and shape (1x3 volume dimension in voxels)

    Returns:
        linidxsrs (np.ndarray):
            N-array or PxQ matrix of Linear voxel indices
    """
    mat = np.array(vol_def.affine)

    # Check that coordinate transformation matrix is 4x4
    if (mat.shape != (4,4)):
        sys.exit('Error: Matrix should be 4x4')

    rs = coords.shape
    if (rs[0] != 3):
        sys.exit('Error: First dimension of coords should be 3')

    if (np.size(rs) == 2):
        nCoordsPerNode = 1
        nVerts = rs[1]
    elif (np.size(rs) == 3):
        nCoordsPerNode = rs[1]
        nVerts = rs[2]
    else:
        sys.exit('Error: Coordindates have %d dimensions, not supported'.format(np.size(rs)))

    # map to 3xP matrix (P coordinates)
    coords = np.reshape(coords,[3,-1])
    coords = np.vstack([coords,np.ones((1,rs[1]))])

    ijk = np.linalg.solve(mat,coords)
    ijk = np.rint(ijk)[0:3,:]
    # Now set the indices out of range to -1
    for i in range(3):
        ijk[i,ijk[i,:]>=vol_def.shape[i]]=-1
    return ijk

def sample_image(img,xm,ym,zm,interpolation):
    """
    Return values after resample image
    
    Args:
        img (Nifti image)
        xm (np-array)
            X-coordinate in world coordinates 
        ym (np-array)
            Y-coordinate in world coordinates
        zm (np-array)
            Z-coordinate in world coordinates 
        interpolation (int)
            0: Nearest neighbor
            1: Trilinear interpolation 
    Returns:
        value (np-array)
            Array contains all values in the image
    """
    im,jm,km = affine_transform(xm,ym,zm,inv(img.affine))

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

        ir, jr, kr, invalid = check_range(img, ir, jr, kr)
        value = img.get_fdata()[ir, jr, kr]
    
    # Kill the invalid elements
    if value.dtype is float:
        value[invalid]=np.nan
    else: 
        value[invalid]=0
    return value





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

def get_gifti_column_names(gifti):
    """Returns the column names from a gifti file (*.label.gii or *.func.gii)

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

def get_gifti_colortable(gifti,ignore_0=True):
    """Returns the RGBA color table and matplotlib cmap from gifti object (*.label.gii)

    Args:
        gifti (gifti image):
            Nibabel Gifti image

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

    if ignore_0:
        rgba = rgba[1:]
        labels = labels[1:]

    cmap = LinearSegmentedColormap.from_list('mylist', rgba, N=len(rgba))
    mpl.cm.unregister_cmap("mycolormap")
    mpl.cm.register_cmap("mycolormap", cmap)

    return rgba, cmap

def get_gifti_anatomical_struct(gifti):
    """
    Returns the primary anatomical structure for a gifti object (*.label.gii or *.func.gii)

    Args:
        gifti (gifti image):
            Nibabel Gifti image

    Returns:
        anatStruct (string):
            AnatomicalStructurePrimary attribute from gifti object

    """
    N = len(gifti._meta.data)
    anatStruct = []
    for i in range(N):
        if 'AnatomicalStructurePrimary' in gifti._meta.data[i].name:
            anatStruct.append(gifti._meta.data[i].value)
    return anatStruct

def get_gifti_labels(gifti):
    """Returns labels from gifti object (*.label.gii)

    Args:
        gifti (gifti image):
            Nibabel Gifti image

    Returns:
        labels (list):
            labels from gifti object
    """
    # labels = img.labeltable.get_labels_as_dict().values()
    label_dict = gifti.labeltable.get_labels_as_dict()
    labels = list(label_dict.values())
    return labels
