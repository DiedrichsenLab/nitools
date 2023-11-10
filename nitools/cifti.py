""" General tools for manipulating Cifti2Images
"""
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import nitools as nt

def make_label_cifti(data, bm_axis,
                       labels=None,
                       label_names=None,
                       column_names=None,
                       label_RGBA=None):
    """Generates a label Cifti2Image from a numpy array

    Args:
        data (np.array):
            num_vert x num_col data
        bm_axis:
            The corresponding brain model axis (voxels or vertices)
        labels (list): Numerical values in data indicating the labels -
            defaults to np.unique(data)
        label_names (list):
            List of strings for names for labels
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors for labels
    Returns:
        cifti (GiftiImage): Label gifti image 
    """
    if data.ndim == 1:
        # reshape to (1, num_vertices)
        data = data.reshape(-1, 1)

    num_verts, num_cols = data.shape
    if labels is None:
        labels = np.unique(data)
    num_labels = len(labels)

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if column_names is None:
        column_names = []
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i + 1))

    # Determine color scale if empty
    if label_RGBA is None:
        label_RGBA = [(0.0, 0.0, 0.0, 0.0)]
        if 0 in labels:
            num_labels -= 1
        hsv = plt.cm.get_cmap('hsv', num_labels)
        color = hsv(np.linspace(0, 1, num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        for i in range(num_labels):
            label_RGBA.append((color[i][0],
                               color[i][1],
                               color[i][2],
                               color[i][3]))

    # Create label names from numerical values
    if label_names is None:
        label_names = ['???']
        for i in labels:
            if i == 0:
                pass
            else:
                label_names.append("label-{:02d}".format(i))

    assert len(label_RGBA) == len(label_names), \
        "The number of parcel labels must match the length of colors!"
    labelDict = []
    for i, nam in enumerate(label_names):
        labelDict.append((nam, label_RGBA[i]))

    labelAxis = nb.cifti2.LabelAxis(column_names, dict(enumerate(labelDict)))
    header = nb.Cifti2Header.from_axes((labelAxis, bm_axis))
    img = nb.Cifti2Image(dataobj=data.T, header=header)
    return img

def join_giftis_to_cifti(giftis,mask=[None,None],seperate_labels=False,join_zero=True):
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
            True: Offsets the label for the right hemisphere, and prepend a L/R to label (default = False)
        join_zero (bool, optional):
            If separate labels, and joint_zero set to true, the zero-label will be joined for the two hemispheres, resulting label less overall. (default = True)

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
        if nt.get_gifti_anatomical_struct(GIF[h])!=hem_name_gifti[h]:
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

def split_cifti_to_giftis(cifti_img, type = "label", column_names = None):
    """ Splits a Cifti files with cortical data into two gifti objects

    Args:
        cifti_img (nb.CiftiImage): C
        type (str): Type of data "label"/"func"
        column_names (list, optional): Column names for Gifti header. Defaults to None.

    Returns:
        gii (list): List of two GiftiImages
    """
    img = surf_from_cifti(cifti_img,
                struct_names=['CIFTI_STRUCTURE_CORTEX_LEFT',
                                'CIFTI_STRUCTURE_CORTEX_RIGHT'])

    gii = []
    for h, hem_name in enumerate(['CortexLeft', 'CortexRight']):

        if type == "label":
            # Extract the label information from the cifti
            label_axis = cifti_img.header.get_axis(0)
            label_dict = label_axis.get_element(0)[1]
            label_values = list(label_dict.keys())
            [label_names, label_RGBA] = zip(*label_dict.values())

            gii.append(nt.make_label_gifti(
                                            img[h].T,
                                            anatomical_struct=hem_name,
                                            label_names=label_names,
                                            column_names=column_names,
                                            label_RGBA=label_RGBA,
                                            labels=label_values
                                            ))
        elif type == "func":
            gii.append(nt.make_func_gifti(
                                        img[h].T,
                                        anatomical_struct=hem_name,
                                        column_names=column_names
                                        ))
    return gii

def volume_from_cifti(cifti, struct_names=[]):
        """ Gets the 4D nifti object containing the data
        for all subcortical (volume-based) structures

        Args:
            cifti (ciftiImage):
                cifti object containing the data
            struct_names (list or None):
                List of structure names that are included
                defaults to None
        Returns:
            nii_vol(niftiImage):
                nifti object containing the data
        """
        # get brain axis models
        bmf = cifti.header.get_axis(1)
        # get the data array with all the time points, all the structures
        d_array = cifti.get_fdata(dtype=np.float32)

        struct_names = [nb.cifti2.BrainModelAxis.to_cifti_brain_structure_name(n) for n in struct_names]

        # initialize a matrix representing 4D data (x, y, z, time_point)
        vol = np.zeros(bmf.volume_shape + (d_array.shape[0],))
        for idx, (nam,slc,bm) in enumerate(bmf.iter_structures()):

            if (any(s in nam for s in struct_names)):
                ijk = bm.voxel
                bm_data = d_array[:, slc]
                i  = (ijk[:,0] > -1)

                # fill in data
                vol[ijk[i, 0], ijk[i, 1], ijk[i, 2], :]=bm_data[:,i].T

        # save as nii
        nii_vol_4d = nb.Nifti1Image(vol,bmf.affine)
        return nii_vol_4d

def surf_from_cifti(cifti,
                    struct_names=['cortex_left','cortex_right']):
        """ Gets the data for cortical surface vertices (Left and Right)
        from normal cifti or parcellated cifti

        Args:
            cifti (cifti2Image):
                Input cifti that contains surface data
                Dimension 0 is data, Dimension 1 is brain model or parcel
            struct_names (list):
                Names of anatomical structures (in cifti format).
                Defaults to left and right hemisphere
        Returns:
            list of ndarrays:
                Data for all surface, with data x numVert for each surface
        """
        # get brain model or parcel axis
        bmf = cifti.header.get_axis(1)
        # print(dir(bmf))
        # get the data array for  all the structures
        data_array = cifti.get_fdata()
        data_list = []
        # Ensure that structure names are in CIFTI format
        struct_names = [nb.cifti2.BrainModelAxis.to_cifti_brain_structure_name(n) for n in struct_names]

        # For Brain model axis
        if isinstance(bmf, nb.cifti2.BrainModelAxis):
            for idx, (nam,slc,bm) in enumerate(bmf.iter_structures()):
                # just get the cortical surfaces
                if (any(s in nam for s in struct_names)):
                    values = np.full((data_array.shape[0],bmf.nvertices[nam]),np.nan)
                    # get the values corresponding to the brain model
                    values[:,bm.vertex] = data_array[:, slc]
                    data_list.append(values)
        elif isinstance(bmf, nb.cifti2.ParcelsAxis):
            parcels = bmf.vertices
            for s in struct_names:
                values = np.full((data_array.shape[0],bmf.nvertices[s]),np.nan)
                for i,p in enumerate(parcels):
                    if s in p.keys():
                        values[:,p[s]]=data_array[:,i:i+1]
                data_list.append(values)
        return data_list

def smooth_cifti(cifti_input,
                 cifti_output,
                 left_surface,
                 right_surface,
                 surface_sigma = 2.0,
                 volume_sigma = 0.0,
                 direction = "COLUMN",
                 ):
    """
    smoothes a cifti file on the direction specified by "direction"

    Args:
        cifti_input (str): path to the input cifti file
        cifti_output (str): path to the output cifti file
        left_surface (str): path to the left surface
        right_surface (str): path to the right surface
        surface_sigma (float): sigma for surface smoothing
        volume_sigma (float): sigma for volume smoothing
        direction (str): direction of smoothing, either "ROW" or "COLUMN"
    """
    # to discard NaNs they will be converted to 0s before smoothing
    # otherwise Nans will spread as a result of smoothing
    cifti = nb.load(cifti_input)
    ## create a mask for NaNs
    mask = np.isnan(cifti.get_fdata())
    ## create a temp cifti object and replace NaNs with 0s
    cifti_tmp = nb.Cifti2Image(dataobj=np.nan_to_num(cifti.get_fdata()), header=cifti.header)
    nb.save(cifti_tmp, 'temp.dscalar.nii')


    # make up the command
    # overwrite the temp file created
    smooth_cmd = f"wb_command -cifti-smoothing 'temp.dscalar.nii' {surface_sigma} {volume_sigma} {direction} {cifti_output} -left-surface {left_surface} -right-surface {right_surface} -fix-zeros-surface"
    subprocess.run(smooth_cmd, shell=True)

    # remove the temp file
    os.remove("temp.dscalar.nii")

    # Replace 0s back to NaN (we don't want the 0s impact model learning)
    C = nb.load(cifti_output)
    data = C.get_fdata()
    data[mask] = np.nan
    C = nb.Cifti2Image(dataobj=data, header=C.header)
    nb.save(C, cifti_output)

    return
