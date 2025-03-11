"""Handeling SPM (Statistical Parametric Mapping) fMRI data

Utility object that helps to extract time series data, beta coefficients, and residuals from a GLM stored in a SPM.mat file.

## Usage
```
spm = SpmGlm('path/to/spm')
spm.get_info_from_spm_mat()
[residuals, beta, info] = spm.get_residuals('my_ROI_Mask.nii')
```
"""
from __future__ import annotations
from os.path import normpath, dirname
import numpy as np
import nibabel as nb
from nibabel import cifti2
import nitools as nt
from scipy.io import loadmat
import pandas as pd


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
        try:
            # SPM = loadmat(f"{self.path}/SPM.mat", simplify_cells=True)['SPM']
            SPM = loadmat(f"{self.path}/SPM.mat", simplify_cells=True)['SPM']
            spm_file_loaded_with_scipyio = True
        except Exception as e:
            print(
                f"Error loading SPM.mat file. The file was saved as mat-file version 7.3 (see https://www.mathworks.com/help/matlab/import_export/mat-file-versions.html). Try loading the mat-file with Matlab, and saving it as mat-file version 7.0 intead. Use this command:  ")
            print(f"cp {self.path}/SPM.mat {self.path}/SPM.mat.backup")
            print(
                f"matlab -nodesktop -nosplash -r \"load('{self.path}/SPM.mat'); save('{self.path}/SPM.mat', '-struct', 'SPM', '-v7'); exit\"")

        # Get basic information from SPM.mat
        self.nscans = SPM['nscan']
        self.nruns = len(self.nscans)
        # Get the name and information on all the beta files``
        self.beta_files = [v['fname'] for v in SPM['Vbeta']]
        self.beta_names = []
        self.run_number = []
        # Extract run number and condition name from SPM names
        for reg_name in SPM['xX']['name']:
            s = reg_name.split(' ')
            self.run_number.append(int(s[0][3:-1]))
            self.beta_names.append(s[1])
        self.run_number = np.array(self.run_number)
        self.beta_names = np.array(self.beta_names)
        # Get the raw data file name
        self.rawdata_files = [self.relocate_file(p) for p in SPM['xY']['P']]
        # Get the necesssary matrices to reestimate the GLM for getting the residuals
        self.filter_matrices = [k['X0'] for k in SPM['xX']['K']]
        self.reg_of_interest = SPM['xX']['iC']
        self.design_matrix = SPM['xX']['xKXs']['X']  # Filtered and whitened design matrix
        self.eff_df = SPM['xX']['erdf']  # Effective degrees of freedom
        self.weight = SPM['xX']['W']  # Weight matrix for whitening
        self.pinvX = SPM['xX']['pKX']  # Pseudo-inverse of (filtered and weighted) design matrix

    def relocate_file(self, fpath: str) -> str:
        """SPM file entries to current project directory and OS.

        These are file paths suffixed with two spaces and an index number.
        They are stored as absolute paths on the computer where it is
        generated, meaning the project directory, as well as OS-dependent
        path separator may have changed. This method fixes these.

        Args:
            fpath (str): SPM-style file path and number from unknown OS

        Returns:
            str: SPM-style file path
        """
        norm_fpath = fpath.replace('\\', '/')
        base_path = dirname(self.path).replace('\\', '/')
        c = norm_fpath.find('imaging_data')
        g = base_path.find('glm')
        return base_path[:g] + norm_fpath[c:]
        # return norm_fpath

    def get_betas(self, mask):
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
        if isinstance(mask, nb.Nifti1Image):
            coords = nt.get_mask_coords(mask)
        elif isinstance(mask, np.ndarray) and (mask.shape[0] == 3):
            coords = mask
        else:
            raise ValueError('Mask should be a 3xP array or coordinates, a nifti1image, or nifti file name')

        # Generate the list of relevant beta images:
        indx = self.reg_of_interest - 1
        beta_files = [f'{self.path}/{self.beta_files[i]}' for i in indx]
        # Get the data from beta and ResMS files
        rms_file = [f'{self.path}/ResMS.nii']
        data = nt.sample_images(beta_files + rms_file, coords, use_dataobj=False)
        # Return the data and the observation descriptors
        info = {'reg_name': self.beta_names[indx], 'run_number': self.run_number[indx]}
        return data[:-1, :], data[-1, :], info

    def get_residuals(self, mask):
        """
        Collects 3d images of a range of GLM residuals
        (typical SPM GLM results) and corresponding metadata
        (scan number + run info) into respective lists

        Args:
            res_range (range): range of to be saved residual images per run
        """
        # Sample the relevant time series data
        if isinstance(mask, str):
            mask = nb.load(mask)
        if isinstance(mask, nb.Nifti1Image):
            coords = nt.get_mask_coords(mask)
        elif isinstance(mask, np.ndarray) and (mask.shape[0] == 3):
            coords = mask
        else:
            raise ValueError('Mask should be a 3xP array or coordinates, a nifti1image, or nifti file name')

        data = nt.sample_images(self.rawdata_files, coords, use_dataobj=True)

        # Filter and temporal pre-whiten the data
        fdata = self.spm_filter(self.weight @ data)  # spm_filter

        # Estimate the beta coefficients and residuals
        beta = self.pinvX @ fdata
        residuals = fdata - self.design_matrix @ beta

        # Return the regressors of interest
        indx = self.reg_of_interest - 1
        info = {'reg_name': self.beta_names[indx], 'run_number': self.run_number[indx]}
        return residuals, beta[indx, :], info

    def spm_filter(self, data):
        """
        Does high pass-filtering and temporal weighting of the data (indentical to spm_filter)

        Args:
            data (ndarray): 2d array of time series data (TxP)
        Returns:
            data (ndarray): 2d array of time series data (TxP)
        """
        scan_bounds = self.nscans.cumsum()
        scan_bounds = np.insert(scan_bounds, 0, 0)

        fdata = data.copy()
        for i in range(self.nruns):
            Y = fdata[scan_bounds[i]:scan_bounds[i + 1], :]
            if self.filter_matrices[i].size > 0:
                # Only apply with filter matrices are not empty
                Y = Y - self.filter_matrices[i] @ (self.filter_matrices[i].T @ Y)
        return fdata

    def rerun_glm(self, data):
        """
        Re-estimate the GLM (without hyperparameter estimation) on new data

        Args:
            data (ndarray): 2d array of time series data (TxP)
        Returns:
            beta (ndarray): 2d array of beta coefficients (PxQ)
            info (dict): with lists reg_name and run_number (Q long)
            data_filt (ndarray): 2d array of filtered time series data (TxP)
            data_hat (ndarray): 2d array of predicted time series data (TxP) - 
                This is predicted only using regressors of interest (without the constant or other nuisance regressors)
            data_adj (ndarray): 2d array of adjusted time series data (TxP)
                This is filtered timeseries with constants and other nuisance regressors substrated out
            residuals (ndarray): 2d array of residuals (TxP)
        """

        # Filter and temporal pre-whiten the data
        data_filt = self.spm_filter(self.weight @ data)  # spm_filter

        # Estimate the beta coefficients and residuals
        beta = self.pinvX @ data_filt
        residuals = data_filt - self.design_matrix @ beta
        # Get estimated (predicted) timeseries (regressors of interest without the constant)
        data_hat = self.design_matrix[:, self.reg_of_interest[:-1]] @ beta[self.reg_of_interest[:-1], :]

        # Get adjusted timeseries
        data_adj = data_hat + residuals

        # Return the regressors of interest (apart from the constant)
        indx = self.reg_of_interest - 1
        info = pd.DataFrame({'reg_name': self.beta_names[indx], 'run_number': self.run_number[indx]})

        return beta[indx, :], info, data_filt, data_hat, data_adj, residuals


def cut(X, pre, at, post, padding='last'):
    """
    Cut segment from signal X.

    Parameters:
    X : np.ndarray
        Input data (rows: voxels, cols: time samples)
    pre : int
        N samples before `at`
    at : int or None
        Sample index at which to cut. If None or NaN, returns NaNs.
    post : int
        N samples after `at`
    padding : str, optional
        Padding strategy when time is not available ('nan', 'zero', 'last'). Default is 'last'.

    Returns:
    np.ndarray
        The extracted segment of the trajectory.
    """
    if at is None or np.isnan(at):
        return np.full((pre + post + 1, X.shape[1]), np.nan)

    rows, cols = X.shape
    start, end = max(0, at - pre), min(at + post + 1, rows)
    y0 = X[:, start:end]

    pad_before, pad_after = max(0, pre - (at - start)), max(0, post - (rows - end))

    if padding == 'nan':
        y = np.pad(y0, ((pad_before, pad_after), (0, 0)), mode='constant', constant_values=np.nan)
    elif padding == 'zero':
        y = np.pad(y0, ((pad_before, pad_after), (0, 0)), mode='constant', constant_values=0)
    elif padding == 'last':
        y = np.pad(y0, ((pad_before, pad_after), (0, 0)), mode='edge')
    else:
        raise ValueError("Unknown padding option. Use: 'nan', 'last', 'zero'")

    return y


def avg_cut(X, pre, at, post, padding='last', stats='mean', axis=0, ResMS=None):
    """
    Takes a vector of sample locations (at) and returns the signal (X) aligned and averaged around those locations
    Args:
        X (np.array):
            Signal to cut.
        pre (int):
            N samples before cut
        at (np.array or list):
            Cut locations in samples
        post (int):
            N samples after cut
        padding (str, optional):
            Padding strategy when time is not available ('nan', 'zero', 'last'). Default is 'last'.
        stats (str):
            Sufficient statistic used for the area
               mean:       Mean of the voxels
               whitemean:  Mean of the voxel, spatially weighted by noise
        axis (int):
            Axis along which the sufficient statistic is applied. If X.shape = (n_voxels, n_samples) then axis=0
        ResMS (np.array):
            Residual mean squares of shape (n_voxels,) only used if stats='prewhiten' for spatial prewhitening

    Returns:

        y (np.array):
            Signal cut and averaged around locations at.

    """

    if stats == 'mean':
        X_avg = X.mean(axis=axis)
    elif stats == 'whiten':
        X_avg = (X / np.sqrt(ResMS)).mean(axis=axis)
    else:
        raise ValueError('Wrong argument for stats (permitted: mean, whiten)')

    y_tmp = []
    for a in at:
        y_tmp.append(cut(X_avg, pre, a, post, padding))

    y = np.vstack(y_tmp).mean(axis=0)

    return y
