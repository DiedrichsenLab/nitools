"""Handeling SPM (Statistical Parametric Mapping) fMRI data

Utility object that helps to extract time series data, beta coefficients, and residuals from a GLM stored in a SPM.mat file.

## Usage
```
betas = rsatoolbox.io.spm.load_betas('/dir/imgs/')
betas.save2combo() ## stores /dir/imgs.nii.gz and /dir/imgs.csv
betas.to_dataset() ## not implemented yet
```
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Callable
from os.path import join, normpath
import nibabel as nb
import numpy as np
import nitools as nt
from pandas import DataFrame
from numpy import stack
from scipy.io import loadmat

if TYPE_CHECKING:
    from numpy.typing import NDArray

class SpmGlm:
    """class for handling first-levels GLMs estimated in SPM

    Attributes:
        path (str):
            paths to directory containing SPM files
    """

    def __init__(self, path: str, nibabelMock=None, globMock=None):
        self.path = normpath(path)
        # self.nibabel = import_nibabel(nibabelMock)
        # self.glob = globMock or glob.glob

    def get_info_from_spm_mat(self):
        """Initializes information for SPM.mat file

        Args:
            spm_mat_path (str): _description_
        """
        SPM = loadmat(f"{self.path}/SPM.mat", simplify_cells=True)['SPM']
        # Get basic information from SPM.mat
        self.nscans = SPM['nscan']
        self.nruns = len(self.nscans)
        # Get the name and information on all the beta files``
        self.beta_files = [v['fname'] for v in SPM['Vbeta']]
        self.beta_names = []
        self.run_number = []
        # Extract run number and condition name from SPM names
        for reg_name in SPM['xX']['name']:
            s=reg_name.split(' ')
            self.run_number.append(int(s[0][3:-1]))
            self.beta_names.append(s[1])
        # Get the raw data file names
        self.rawdata_files = SPM['xY']['P']
        # Get the necesssary matrices to reestimate the GLM for getting the residuals
        self.filter_matrices = [k['X0'] for k in SPM['xX']['K']]
        self.reg_of_interest = SPM['xX']['iC']
        self.design_matrix = SPM['xX']['X']
        self.eff_df = SPM['xX']['erdf'] # Effective degrees of freedom
        self.weight = SPM['xX']['W'] # Weight matrix for whitening
        self.pinvX = SPM['xX']['pKX'] # Pseudo-inverse of (filtered and weighted) design matrix

    def get_betas(self,mask):
        """
        Samples the beta images of an estimated SPM GLM at the mask locations
        also returns the ResMS values, and the obseration descriptors (run and condition) name

        Args:
            mask (ndarray or nibabel nifti1image):
                Indicates which voxels to extract
                Could be a binary 3d-array or a nifti image of the same size as the data
                Or a 3xP array of coordinates to extract (in mm space)
        Returns:
            data (ndarray): N x P array of beta coefficients
            resms (ndarray): 1d array of ResMS values
            obs_descriptors (dict): with lists reg_name and run_number (N long)
        """
        if isinstance(mask, str):
            mask = nb.load(mask)
        if isinstance(mask, nb.nifti1.Nifti1Image):
            [i,j,k]=np.where(mask.get_fdata())
            coords = nt.affine_transform_mat(np.array([i,j,k]),mask.affine)
        elif isinstance(mask, np.ndarray) and (mask.shape[0]!=3):
            coords = mask
        else:
            raise ValueError('Mask should be a 3xP array or coordinates, a nifti1image, or nifti file name')

        return self

    def get_residuals(self):
        """
        Collects 3d images of a range of GLM residuals
        (typical SPM GLM results) and corresponding metadata
        (scan number + run info) into respective lists

        Args:
            res_range (range): range of to be saved residual images per run
        """

        residual_array_superset = []
        self.dim4_descriptors = []
        self.n_res = len(self.glob(join(self.path, "Res_*")))
        if isinstance(res_range, range):
            assert self.n_res >= max(res_range), \
                "res_range outside of existing residuals"
        else:
            res_range = range(1, self.n_res+1)

        run_counter = 0

        run_counter += 1
        for res in res_range:
            res_image_path = join(self.path, "Res_" +
                                            str(res).zfill(4))
            res_image = self.nibabel.nifti1.load(res_image_path)
            residual_array_superset.append(res_image.get_fdata())
            self.dim4_descriptors.append("res_" + str(res).zfill(4) + "_run_" +
                                    str(run_counter).zfill(2))
        # Get affine matrix
        self.subject_affine = res_image.affine.copy()
        self.pooled_data_array = stack(residual_array_superset, axis=3)
        return self

    def save2nifti(self, fpath: Optional[str]=None):
        """
        Converts 4d array to subject-specific 4d NIfTi image
        of beta coeffients or residuals and saves it to your OS

        Args:
            fpath (str):
                path to which you want to save your data
        """
        fpath = fpath or (self.path + '.nii.gz')
        pooled_data = self.nibabel.nifti1.Nifti1Image(
            self.pooled_data_array,
            self.affine
        )
        self.nifti_filename = join(fpath)
        self.nibabel.nifti1.save(pooled_data, self.nifti_filename)

    def save2csv(self, fpath: Optional[str]=None):
        """
        Saves subject-specific 4d NIfTi image descriptors to a csv file

        Args:
            descriptors (list of str):
                descriptors of fourth a NIfTi file's 4th dimension
            fpath (str):
                path to which you want to save your data
        """
        fpath = fpath or (self.path + '.csv')
        df = DataFrame({'descriptor': self.dim4_descriptors})
        df.to_csv(fpath, header=False)

    def save2combo(self, fpath: Optional[str]=None):
        """
        Combined saving of fmri data and descriptors
        """
        fpath_niigz = fpath or (self.path + '.nii.gz')
        fpath_csv = fpath_niigz.replace('.nii.gz', '.csv')
        self.save2nifti(fpath_niigz)
        self.save2csv(fpath_csv)
