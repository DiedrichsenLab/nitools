"""Nifti and Volume-based tools for sampling and deforming
"""
import numpy as np
import nibabel as nb
import sys

def affine_transform(x1, x2, x3, M):
    """
    Returns affine transform of x in vector

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
    Returns affine transform of x in matrix form

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

def coords_to_voxelidxs(coords,volDef):
    """
    Maps coordinates to voxel indices

    INPUT:
        coords (3*N matrix or 3xPxQ array):
            (x,y,z) coordinates
        voldef (nibabel object):
            Nibabel object with attributes .affine (4x4 voxel to coordinate transformation matrix from the images to be sampled (1-based)) and shape (1x3 volume dimension in voxels)

    OUTPUT:
        linidxsrs (3xN-array or 3xPxQ matrix):
            voxel indices
    """
    mat = np.array(volDef.affine)

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

    ijk = affine_transform_mat(coords,np.linalg.inv(mat)).astype(int)
    # Now set the indices out of range to -1
    for d in range(3):
        ijk[d,ijk[d,:]>=volDef.shape[d]]=-1
        ijk[d,ijk[d,:]<0]=-1
    return ijk


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
        elif D.ndim ==5:
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
    if value.dtype==np.dtype('float'):
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

def deform_image(source,deform,interpolation):
    """ This function resamples an image into an atlas space,
    using a non-linear deformation map

    Args:
        source (NiftiImage): Source nifti imate
        deform (NiftiImage): A (x,y,z,1,3) deformation image
        interpolation (int): 0: Nearest Nieghbour 1:trilinear interpolation

    Returns:
        NiftiImage: Resampled source imahe - has the same shape as deform
    """
    XYZ = deform.get_fdata()
    data = sample_image(source,
                    XYZ[:,:,:,0,0],
                    XYZ[:,:,:,0,1],
                    XYZ[:,:,:,0,2],interpolation)
    outimg = nb.Nifti1Image(data,deform.affine)
    return outimg

def change_nifti_numformat(infile,
                           outfile,
                           new_numformat='uint16',
                           typecast_data=True):
    """Changes the number format of a nifti file and saves it.

    Args:
        infile (str): file name of input file 
        outfile (str): file name output file 
        new_numformat (str): New number format. Defaults to 'uint16'.
        type_cast_data (bool): If true, typecasts the data as well. 
    Note:
        typecast_data changes the data format and can lead to and wrap-around of values, as the data is typecasted as well. 
        Do only when correcting a previous mistake in processing. 
    """
    A = nb.load(infile)
    X = A.get_fdata()
    if typecast_data: 
        X = X.astype(new_numformat)
    head = A.header.copy()
    head.set_data_dtype(new_numformat)
    B = nb.Nifti1Image(X,A.affine,header=head)
    nb.save(B,outfile)

def sample_images(imgs,coords,use_dataobj=True):
    """
    Samples a list of images at a set of voxels (coordinates nearest neighbor)
    3d images are being returned as a single value per voxel. 4d images are being returned as a vector per voxel. 
    The function respects ths SPM-convention of specifying time slices 'img.nii,1' 
    Note that either all or none of the image names need to follow the 'img.nii,1' convention. 
    Args:
        imgs (list):
            List of Nifti file names (SPM convention for time slices is respected) 
        coords (np-array):
            3xN matrix of coordinates of to be sampled voxels 
    Returns:
        values (np-array):
            Array contains all values in the image
    """
    n = len(imgs)
    if ',' in imgs[0]: # SPM-convention of indicating time slice with ',x' notation 
        fnames=[f.split(',')[0] for f in imgs]
        slice=np.array([f.split(',')[1] for f in imgs],dtype=int)-1
        unique_fnames,img_indx = np.unique(fnames,return_inverse=True)
        input_vols = [nb.load(f) for f in unique_fnames]
    else:       # Simply concatenate all the images
        input_vols = [nb.load(f) for f in imgs]
        slice = []
        img_indx = []
        for i,v in enumerate(input_vols):
            if v.ndim==4:
                img_indx.append(np.ones(v.shape[3],dtype=int)*i)
                slice.append(np.arange(v.shape[3],dtype=int))
            else:
                img_indx.append(np.array([i],dtype=int))
                slice.append(np.array([0],dtype=int))
        slice = np.concatenate(slice)
        img_indx = np.concatenate(img_indx) 
    N = len(slice)
    # Load the header from all the input files
    voxels = coords_to_voxelidxs(coords,input_vols[0])
    data = np.empty((N,coords.shape[1]))
    if use_dataobj:
        for i,v in enumerate(input_vols):
            dindx = (img_indx == i)
            data_block = v.dataobj[min(voxels[0]):max(voxels[0])+1,min(voxels[1]):max(voxels[1])+1,min(voxels[2]):max(voxels[2])+1]
            D = data_block[voxels[0]-min(voxels[0]),voxels[1]-min(voxels[1]),voxels[2]-min(voxels[2])]
            if v.ndim==3:
                data[dindx,:]=D 
            else:
                data[dindx,:]=D[:,slice[dindx]].T
    else:
        for i,v in enumerate(input_vols):
            dindx = (img_indx == i)
            if v.ndim==3:
                data[dindx,:]=v.get_fdata()[voxels[0],voxels[1],voxels[2]]
            else:
                D = v.get_fdata()[voxels[0],voxels[1],voxels[2]]
                data[dindx,:]=D[:,slice[dindx]].T
    return data


def get_mask_coords(mask):
    """ Returns the coordinates of the mask voxels

    Args:
        mask (str or NiftiImage or 3xP np-array):
            Mask file name or NiftiImage or np-array
    Returns:
        coords (np-array):
            3xP array of coordinates
    """
    if isinstance(mask, str):
        mask = nb.load(mask)
    if isinstance(mask, nb.nifti1.Nifti1Image):
        [i,j,k]=np.where(mask.get_fdata())
        coords = affine_transform_mat(np.array([i,j,k]),mask.affine)
    elif isinstance(mask, np.ndarray) and (mask.shape[0]!=3):
        coords = mask
    else:
        raise ValueError('Mask should be a 3xP array or coordinates, a nifti1image, or nifti file name')
    return coords